"""
pipeline.py — Disaster classification evaluation pipeline.

Recursively discovers every image-containing leaf folder under datasets/,
treats each folder name as the ground-truth class label, runs CLIP inference
on every image, and reports per-category and overall accuracy.

Skipped automatically:
  - Any folder named 'history'
  - Any file that is not a recognised image format (.h5, .json, …)
  - Empty folders
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Ensure src/ is on the path regardless of the CWD the script is launched from.
sys.path.insert(0, str(Path(__file__).parent))

from models.clip_model import predict_disaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_DIR  = Path(__file__).parent.parent / "datasets"
OUTPUTS_DIR  = Path(__file__).parent.parent / "outputs"
CSV_PATH     = OUTPUTS_DIR / "results.csv"

# Maximum images sampled per class during a run.
# Lower values speed up debugging; set to 0 for no limit.
MAX_IMAGES_PER_CLASS: int = 5

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
)

# Folder names that are skipped entirely (case-insensitive).
# Their entire sub-tree is also excluded because rglob won't descend into them
# once we filter on this set.
SKIP_FOLDER_NAMES: frozenset[str] = frozenset({"history"})

# Maps a normalised folder name (lowercase, underscores) to the list of
# keywords that must appear somewhere in the CLIP label string for the
# prediction to count as correct.
#
# Rules:
#   non-empty list  → category has a CLIP analogue; any keyword hit = correct
#   empty list []   → no CLIP analogue exists (Drought, Non-damage, …);
#                     predictions are logged but never scored as correct
#   key absent      → unknown category; fuzzy word-overlap fallback is used
#
# Both the leaf folder name (e.g. "land_slide") and the parent category name
# (e.g. "land_disaster") are mapped so the pipeline works whether images are
# placed at the leaf level or directly inside the category folder.
FOLDER_TO_CLIP_KEYWORDS: dict[str, list[str]] = {
    # ── DisasterModel  (train / test / validation splits) ────────────────────
    "cyclone":                          ["cyclone"],
    "earthquake":                       ["earthquake"],
    "flood":                            ["flooding"],
    "wildfire":                         ["wildfire"],

    # ── comprehensive_disaster_dataset / Damaged_Infrastructure ──────────────
    "damaged_infrastructure":           ["earthquake", "collapsed", "cyclone"],
    "infrastructure":                   ["earthquake", "collapsed"],
    # earthquake leaf reuses the entry above

    # ── comprehensive_disaster_dataset / Fire_Disaster ────────────────────────
    "fire_disaster":                    ["wildfire"],
    "urban_fire":                       ["wildfire"],
    "wild_fire":                        ["wildfire"],

    # ── comprehensive_disaster_dataset / Land_Disaster ────────────────────────
    "land_disaster":                    ["landslide"],
    "land_slide":                       ["landslide"],
    "drought":                          [],          # no CLIP label for drought

    # ── comprehensive_disaster_dataset / Water_Disaster ───────────────────────
    "water_disaster":                   ["flooding", "tsunami"],
    "sea":                              ["tsunami"],

    # ── comprehensive_disaster_dataset / Human_Damage & Non_Damage ───────────
    "human_damage":                     [],          # no CLIP analogue
    "non_damage":                       [],          # no CLIP analogue
    "human":                            [],
    "non_damage_buildings_street":      [],
    "non_damage_wildlife_forest":       [],
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    image_path:      str    # full path to the source image file
    filename:        str    # basename only — used in progress output
    actual_label:    str    # normalised folder name used as ground truth
    predicted_label: str    # full CLIP label string
    confidence:      float  # 0–100 %
    correct:         bool
    mapped:          bool   # True = category has a known CLIP analogue


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Return a lowercase, underscore-separated version of a folder name."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def discover_image_folders(root: Path) -> list[tuple[Path, str]]:
    """
    Walk *root* recursively and return (folder, normalised_label) for every
    directory that directly contains at least one recognised image file.

    Folders in SKIP_FOLDER_NAMES are pruned along with their entire sub-tree.
    """
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    found: list[tuple[Path, str]] = []

    for folder in sorted(root.rglob("*")):
        if not folder.is_dir():
            continue
        # Prune unwanted folder trees by checking every ancestor up to root.
        if any(part.lower() in SKIP_FOLDER_NAMES for part in folder.parts):
            continue
        # Only include folders that directly contain image files (leaf check).
        images = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            found.append((folder, _normalise(folder.name)))

    return found


def iter_images(folder: Path) -> Iterator[Path]:
    """Yield every image file that lives directly inside *folder*."""
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def build_work_queue(
    folders: list[tuple[Path, str]],
    max_per_class: int = MAX_IMAGES_PER_CLASS,
) -> list[tuple[Path, str]]:
    """
    Return a flat list of (image_path, label) capped at *max_per_class* images
    per class label, aggregated across all folders that share the same label.

    Pass max_per_class=0 to process every image without a cap.
    """
    per_class: dict[str, int] = defaultdict(int)
    queue: list[tuple[Path, str]] = []

    for folder, label in folders:
        for image_path in iter_images(folder):
            if max_per_class and per_class[label] >= max_per_class:
                break   # class quota reached; skip remaining images in this folder
            queue.append((image_path, label))
            per_class[label] += 1

    return queue


# ---------------------------------------------------------------------------
# Discovery report  (printed before inference starts)
# ---------------------------------------------------------------------------

def print_discovered_classes(
    folders: list[tuple[Path, str]],
    max_per_class: int = MAX_IMAGES_PER_CLASS,
) -> None:
    """
    Print a scan-plan table showing, per class:
      - total images available on disk
      - how many will actually be processed (capped by max_per_class)
      - CLIP label mapping status

    Finishes with the grand total of images that will enter inference.
    """
    available: dict[str, int] = defaultdict(int)
    for folder, label in folders:
        available[label] += sum(
            1 for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    cap_label = f"max {max_per_class} / class" if max_per_class else "all images"
    width = 76
    print("\n" + "=" * width)
    print(f"{'DATASET SCAN PLAN  (' + cap_label + ')':^{width}}")
    print("=" * width)
    print(
        f"  {'Class (folder name)':<32}{'Available':>11}{'Selected':>10}"
        f"  CLIP match"
    )
    print("-" * width)

    total_available = 0
    total_selected  = 0

    for label in sorted(available):
        avail    = available[label]
        selected = min(avail, max_per_class) if max_per_class else avail
        total_available += avail
        total_selected  += selected

        keywords = FOLDER_TO_CLIP_KEYWORDS.get(label)
        if keywords is None:
            clip_status = "auto-detect"
        elif keywords:
            clip_status = f"yes  [{', '.join(keywords)}]"
        else:
            clip_status = "no analogue"

        print(
            f"  {label:<32}{avail:>11}{selected:>10}  {clip_status}"
        )

    print("-" * width)
    print(
        f"  {'TOTAL':<32}{total_available:>11}{total_selected:>10}"
        f"  {len(available)} classes"
    )
    print(f"\n  Images to process: {total_selected}")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Prediction & correctness
# ---------------------------------------------------------------------------

def is_correct_prediction(label: str, predicted_label: str) -> tuple[bool, bool]:
    """
    Return (correct, mapped).

    *mapped* is True when the category has a known CLIP analogue (non-empty
    keyword list).  *correct* is True when any expected keyword appears in the
    predicted label string.

    For unknown labels not present in FOLDER_TO_CLIP_KEYWORDS, a fuzzy
    word-overlap fallback is applied so new categories work without code changes.
    """
    keywords = FOLDER_TO_CLIP_KEYWORDS.get(label)

    if keywords is None:
        # Unknown label: check whether any word from the folder name (≥4 chars)
        # appears in the predicted CLIP string.
        words = label.replace("_", " ").split()
        correct = any(w in predicted_label.lower() for w in words if len(w) >= 4)
        return correct, False   # not in our map → treat as unmapped

    mapped = bool(keywords)
    correct = any(kw in predicted_label.lower() for kw in keywords)
    return correct, mapped


def classify_image(image_path: Path, ground_truth: str) -> PredictionRecord | None:
    """
    Run CLIP inference on a single image.

    Returns None and logs a warning on any failure (corrupt file, OOM, etc.)
    so the pipeline continues with the remaining images.
    """
    try:
        result = predict_disaster(str(image_path))
    except Exception as exc:
        logger.warning(
            "Skipping '%s' — %s: %s", image_path.name, type(exc).__name__, exc
        )
        return None

    predicted  = result["prediction"]
    confidence = result["confidence"]
    correct, mapped = is_correct_prediction(ground_truth, predicted)

    return PredictionRecord(
        image_path=str(image_path),
        filename=image_path.name,
        actual_label=ground_truth,
        predicted_label=predicted,
        confidence=confidence,
        correct=correct,
        mapped=mapped,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(dataset_dir: Path = DATASET_DIR) -> list[PredictionRecord]:
    """
    Full pipeline: discover → plan → infer → time → return records.

    Prints a pre-inference scan plan, one progress line per image during
    inference, and a timing summary at the end.
    """
    folders = discover_image_folders(dataset_dir)

    if not folders:
        logger.error("No image-containing folders found under %s", dataset_dir)
        return []

    print_discovered_classes(folders)

    work_queue     = build_work_queue(folders)
    unique_classes = {label for _, label in work_queue}

    logger.info(
        "Pipeline started — %d images across %d classes (MAX_IMAGES_PER_CLASS=%d)",
        len(work_queue),
        len(unique_classes),
        MAX_IMAGES_PER_CLASS,
    )

    results:         list[PredictionRecord] = []
    inference_times: list[float]            = []
    total_seen = 0
    pipeline_start = time.perf_counter()

    for image_path, label in work_queue:
        total_seen += 1
        t0     = time.perf_counter()
        record = classify_image(image_path, label)
        inference_times.append(time.perf_counter() - t0)

        if record is None:
            continue

        results.append(record)

        if record.correct:
            status = "✓"
        elif not record.mapped:
            status = "?"   # unmapped category — cannot score
        else:
            status = "✗"

        print(
            f"[{total_seen:>5}] {status}  {record.actual_label:<30}"
            f"->  {record.predicted_label:<50}  ({record.confidence:.1f}%)"
        )

    elapsed   = time.perf_counter() - pipeline_start
    avg_ms    = (sum(inference_times) / len(inference_times) * 1000) if inference_times else 0.0
    skipped   = total_seen - len(results)

    logger.info(
        "Pipeline complete — %d processed, %d skipped | "
        "total %.2fs | avg %.0f ms/image",
        len(results), skipped, elapsed, avg_ms,
    )

    print(f"\n  ── Timing ────────────────────────────────")
    print(f"  Total runtime    : {elapsed:.2f} s")
    print(f"  Images processed : {len(results)}  (skipped: {skipped})")
    print(f"  Avg per image    : {avg_ms:.0f} ms")
    print(f"  ──────────────────────────────────────────\n")

    return results


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "image_path",
    "actual_label",
    "predicted_label",
    "confidence",
    "mapped_or_unmapped",
    "correct_or_wrong",
]


def export_csv(
    records: list[PredictionRecord],
    output_path: Path = CSV_PATH,
) -> Path:
    """
    Write every prediction record to *output_path* as a UTF-8 CSV.

    Creates the parent directory automatically if it does not exist.
    Returns the resolved output path so callers can print or log it.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "image_path":         r.image_path,
                "actual_label":       r.actual_label,
                "predicted_label":    r.predicted_label,
                "confidence":         r.confidence,
                "mapped_or_unmapped": "mapped"   if r.mapped   else "unmapped",
                "correct_or_wrong":   "correct"  if r.correct  else "wrong",
            })

    logger.info("Results exported → %s  (%d rows)", output_path, len(records))
    return output_path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def calculate_accuracy(results: list[PredictionRecord]) -> float:
    """Overall accuracy across all records (0–100 %)."""
    if not results:
        return 0.0
    return round(sum(r.correct for r in results) / len(results) * 100, 2)


def calculate_mapped_accuracy(results: list[PredictionRecord]) -> float:
    """Accuracy restricted to records whose category has a CLIP analogue."""
    mapped = [r for r in results if r.mapped]
    if not mapped:
        return 0.0
    return round(sum(r.correct for r in mapped) / len(mapped) * 100, 2)


def print_summary(results: list[PredictionRecord]) -> None:
    """Print a per-category breakdown table and overall / mapped accuracy rows."""
    if not results:
        print("\nNo results — verify that dataset folders contain images.")
        return

    # Accumulate per-category stats
    stats: dict[str, dict] = {}
    for r in results:
        s = stats.setdefault(
            r.actual_label,
            {"correct": 0, "total": 0, "mapped": r.mapped},
        )
        s["total"] += 1
        if r.correct:
            s["correct"] += 1

    width = 74
    print("\n" + "=" * width)
    print(f"{'CLASSIFICATION SUMMARY':^{width}}")
    print("=" * width)
    print(
        f"  {'Category':<30}{'Correct':>9}{'Total':>8}"
        f"{'Accuracy':>10}  CLIP match"
    )
    print("-" * width)

    for category, s in sorted(stats.items()):
        acc = s["correct"] / s["total"] * 100
        tag = "mapped" if s["mapped"] else "no analogue"
        print(
            f"  {category:<30}{s['correct']:>9}{s['total']:>8}"
            f"{acc:>9.1f}%  {tag}"
        )

    print("-" * width)

    total_correct  = sum(r.correct for r in results)
    overall_acc    = calculate_accuracy(results)
    mapped_acc     = calculate_mapped_accuracy(results)
    mapped_total   = sum(1 for r in results if r.mapped)
    mapped_correct = sum(1 for r in results if r.mapped and r.correct)

    print(
        f"  {'OVERALL  (all classes)':<30}"
        f"{total_correct:>9}{len(results):>8}{overall_acc:>9.1f}%"
    )
    print(
        f"  {'MAPPED   (CLIP classes only)':<30}"
        f"{mapped_correct:>9}{mapped_total:>8}{mapped_acc:>9.1f}%"
    )
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    records = run_pipeline()
    print_summary(records)
    if records:
        saved = export_csv(records)
        print(f"  CSV exported → {saved}")
