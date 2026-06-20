#!/usr/bin/env python3
"""
create_filtered_dataset.py — Build a CLIP-aligned disaster evaluation dataset.

Scans the comprehensive disaster dataset, maps each leaf folder to one of five
CLIP-supported categories, and copies the images into a clean flat structure.

Category mapping (formatted folder name → target class):
  Water Disaster  → flood        (1035 images available)
  Wild Fire       → fire         ( 514 images available)
  Urban Fire      → fire         ( 419 images available)
  Earthquake      → earthquake   (  36 images available)
  Land Slide      → landslide    ( 456 images available)
  Cyclone         → cyclone      (   0 — not in this dataset)

Skipped automatically (not in CLIP label space):
  Infrastructure, Drought, Human Damage, Sea, Human,
  Non Damage Buildings Street, Non Damage Wildlife Forest

Output structure:
  dataset_filtered/
  ├── flood/
  ├── fire/
  ├── earthquake/
  ├── landslide/
  └── cyclone/

Usage:
    # Copy all available images
    python scripts/create_filtered_dataset.py

    # Cap at 100 images per target class (recommended for quick evaluation)
    python scripts/create_filtered_dataset.py --limit 100

    # Custom source / destination
    python scripts/create_filtered_dataset.py \\
        --source datasets/comprehensive_disaster_dataset \\
        --dest   dataset_filtered

Then run evaluation:
    python scripts/batch_evaluate.py --dataset dataset_filtered
    python scripts/batch_evaluate.py --dataset dataset_filtered --limit 50
"""

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE  = PROJECT_ROOT / "datasets" / "comprehensive_disaster_dataset"
DEFAULT_DEST    = PROJECT_ROOT / "dataset_filtered"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# The five CLIP-aligned target classes (folder names inside dataset_filtered/)
TARGET_CLASSES = ["flood", "fire", "earthquake", "landslide", "cyclone"]

# Mapping: formatted leaf-folder name → target class
# Formatted = folder_name.replace("_", " ").strip().title()
# Matches exactly how batch_evaluate.py derives labels.
CATEGORY_MAP: dict[str, str] = {
    "Water Disaster": "flood",
    "Wild Fire":      "fire",
    "Urban Fire":     "fire",
    "Earthquake":     "earthquake",
    "Land Slide":     "landslide",
    "Cyclone":        "cyclone",
}

SEP = "─" * 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def _format_label(folder_name: str) -> str:
    """Convert a folder name to a human-readable label (mirrors batch_evaluate.py)."""
    return folder_name.replace("_", " ").strip().title()


def _safe_copy(src: Path, dest_dir: Path) -> Path:
    """Copy src into dest_dir, renaming if a file with the same name already exists.

    Returns the destination path that was actually written.
    """
    dest = dest_dir / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
        return dest

    # Rename: stem_1.ext, stem_2.ext, ...
    counter = 1
    while True:
        dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        if not dest.exists():
            shutil.copy2(src, dest)
            return dest
        counter += 1


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_mappings(
    source_dir: Path,
    limit: int,
) -> tuple[list[tuple[Path, str]], dict[str, list[str]]]:
    """Walk source_dir recursively and build a list of (image_path, target_class) pairs.

    Returns:
        mappings   — list to copy, capped at `limit` per target class (0 = no cap)
        skipped    — {formatted_label: [example_path]} for categories not in CATEGORY_MAP
    """
    if not source_dir.exists():
        sys.exit(
            f"\n[ERROR] Source dataset not found: {source_dir}\n"
            f"  Point --source at the root of your disaster dataset."
        )

    # Collect ALL (image, target_class) pairs first, respecting mapping
    all_entries: list[tuple[Path, str]] = []
    skipped: dict[str, list[str]] = defaultdict(list)

    for folder in sorted(source_dir.rglob("*")):
        if not folder.is_dir():
            continue

        images = sorted(f for f in folder.iterdir() if _is_image(f))
        if not images:
            continue  # not a leaf folder

        label = _format_label(folder.name)

        if label in CATEGORY_MAP:
            target = CATEGORY_MAP[label]
            for img in images:
                all_entries.append((img, target))
        else:
            skipped[label].append(str(folder.relative_to(source_dir)))

    # Apply per-class limit
    if limit > 0:
        class_counters: dict[str, int] = defaultdict(int)
        limited: list[tuple[Path, str]] = []
        for img, target in all_entries:
            if class_counters[target] < limit:
                limited.append((img, target))
                class_counters[target] += 1
        all_entries = limited

    return all_entries, dict(skipped)


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def create_filtered_dataset(
    source_dir: Path,
    dest_dir: Path,
    limit: int,
) -> None:
    """Main routine: discover, copy, and report."""

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VLM Disaster Analyzer — Filtered Dataset Creator")
    print(SEP)
    print(f"  Source  : {source_dir}")
    print(f"  Dest    : {dest_dir}")
    if limit:
        print(f"  Limit   : {limit} images per target class")
    print(SEP)

    # ── Print mapping table ───────────────────────────────────────────────────
    print("\n  Category mapping:")
    print()
    for src_label, tgt_class in CATEGORY_MAP.items():
        arrow = "→"
        print(f"    {src_label:<25} {arrow}  {tgt_class}")
    print()

    # ── Discover ──────────────────────────────────────────────────────────────
    print("  Scanning source dataset...")
    entries, skipped = discover_mappings(source_dir, limit)

    if not entries:
        sys.exit(
            "\n[ERROR] No images found to copy.\n"
            "  Check that --source points to a valid dataset root."
        )

    # ── Create destination structure ──────────────────────────────────────────
    for cls in TARGET_CLASSES:
        (dest_dir / cls).mkdir(parents=True, exist_ok=True)

    # ── Copy images ───────────────────────────────────────────────────────────
    total         = len(entries)
    idx_width     = len(str(total))
    class_counts: dict[str, int] = defaultdict(int)

    print(f"  Copying {total:,} images...\n")

    for idx, (src_path, target) in enumerate(entries, start=1):
        dest_class_dir = dest_dir / target
        _safe_copy(src_path, dest_class_dir)
        class_counts[target] += 1

        if idx % 100 == 0 or idx == total:
            print(
                f"  [{idx:>{idx_width}}/{total}]  "
                f"flood:{class_counts['flood']:>5}  "
                f"fire:{class_counts['fire']:>5}  "
                f"earthquake:{class_counts['earthquake']:>4}  "
                f"landslide:{class_counts['landslide']:>5}  "
                f"cyclone:{class_counts['cyclone']:>4}"
            )

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Copied images by class:")
    print(SEP)
    total_copied = 0
    for cls in TARGET_CLASSES:
        count = class_counts.get(cls, 0)
        total_copied += count
        status = "" if count > 0 else "  ← not found in source dataset"
        print(f"  {cls:<15} : {count:>6,} images{status}")

    if skipped:
        print(f"\n  Skipped categories (not in CLIP label space):")
        for label in sorted(skipped):
            print(f"    {label}")

    print(f"\n  Total copied : {total_copied:,} images")
    print(f"  Destination  : {dest_dir}")

    # ── Next steps ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Next steps:")
    print(SEP)
    print()
    print("  1. Start the backend (separate terminal):")
    print("       uvicorn backend.main:app --port 8000")
    print()
    print("  2. Run evaluation on the filtered dataset:")
    print(f"       python scripts/batch_evaluate.py --dataset {dest_dir.name}")
    print()
    print("  3. Generate analytics plots:")
    print("       python scripts/generate_plots.py")
    print()
    print(SEP)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a CLIP-aligned disaster evaluation dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help="Root of the source disaster dataset",
    )
    p.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help="Destination root for the filtered dataset",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Max images per target class (0 = copy all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    create_filtered_dataset(
        source_dir=args.source,
        dest_dir=args.dest,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
