"""
preprocess_videos.py — Stage 5 of the VIDI pipeline.

Reads verification_report.csv (valid clips only) and:
  1. Extracts a thumbnail (center-frame JPEG) for each clip
     → thumbnails/{split}/{label}/{clip_stem}.jpg
  2. (Optional) Re-encodes each clip to a standardised resolution and FPS
     → in-place update in processed_videos/{split}/{label}/
     Use --normalize flag to enable this (slower, requires more disk I/O).

Standardisation targets (from config.py):
    Resolution : TARGET_WIDTH × TARGET_HEIGHT (default 256×256, centre-pad)
    Frame-rate : TARGET_FPS          (default 16 fps)
    Codec      : libx264, CRF=22, preset=fast
    Container  : MP4 (faststart for streaming)

Outputs:
    thumbnails/{split}/{label}/{clip_stem}.jpg
    metadata/processed_metadata.csv   — per-clip paths + thumbnail status

Usage:
    python preprocess_videos.py               # thumbnails only (fast)
    python preprocess_videos.py --normalize   # also re-encode resolution/FPS
    python preprocess_videos.py --workers 4   # parallel workers
"""

import sys
import logging
import argparse
import subprocess
import concurrent.futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    METADATA_DIR, REPORTS_DIR, THUMBNAILS_BASE,
    TARGET_FPS, TARGET_HEIGHT, TARGET_WIDTH,
    VIDEO_CODEC, AUDIO_CODEC, CRF,
    DOWNLOAD_WORKERS,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Thumbnail extraction
# ---------------------------------------------------------------------------

def extract_thumbnail(clip_path: Path, thumb_path: Path) -> bool:
    """
    Extract a single JPEG frame from the middle of *clip_path*.
    Scales to TARGET_WIDTH × TARGET_HEIGHT with letterbox padding.
    Returns True on success.
    """
    if thumb_path.exists():
        return True

    thumb_path.parent.mkdir(parents=True, exist_ok=True)

    # Seek to ~0.5 s (good approximation for clips 3-20 s long)
    vf = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}"
        f":force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    )
    cmd = [
        "ffmpeg",
        "-ss", "0.5",
        "-i", str(clip_path),
        "-vframes", "1",
        "-vf", vf,
        "-q:v", "2",       # JPEG quality 2 ≈ ~95%
        "-y", str(thumb_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        return r.returncode == 0 and thumb_path.exists()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Video normalisation
# ---------------------------------------------------------------------------

def normalize_clip(src: Path) -> bool:
    """
    Re-encode *src* in-place to TARGET_WIDTH × TARGET_HEIGHT at TARGET_FPS.
    Uses a temp file to avoid overwriting the source before encoding is done.
    Returns True on success.
    """
    tmp = src.with_suffix(".tmp.mp4")
    vf = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}"
        f":force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={TARGET_FPS}"
    )
    cmd = [
        "ffmpeg",
        "-i", str(src),
        "-vf", vf,
        "-c:v", VIDEO_CODEC,
        "-preset", "fast",
        "-crf", str(CRF),
        "-c:a", AUDIO_CODEC,
        "-movflags", "+faststart",
        "-y", str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1024:
            tmp.replace(src)   # atomic rename
            return True
        tmp.unlink(missing_ok=True)
        return False
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# Per-clip worker
# ---------------------------------------------------------------------------

def _process_clip(args: tuple) -> dict:
    row, do_normalize = args
    clip_path = Path(row["expected_path"])
    safe_label = row["label"].replace("/", "_").replace(" ", "_")
    split      = row["split"]
    clip_stem  = clip_path.stem

    thumb_path = THUMBNAILS_BASE / split / safe_label / f"{clip_stem}.jpg"

    thumb_ok   = extract_thumbnail(clip_path, thumb_path)
    norm_status = "skipped"

    if do_normalize:
        ok = normalize_clip(clip_path)
        norm_status = "ok" if ok else "failed"

    return {
        "clip_id":      row.get("clip_id", ""),
        "youtube_id":   row.get("youtube_id", ""),
        "label":        row["label"],
        "split":        split,
        "clip_path":    str(clip_path),
        "thumbnail":    str(thumb_path) if thumb_ok else None,
        "thumb_status": "ok" if thumb_ok else "failed",
        "norm_status":  norm_status,
    }


# ---------------------------------------------------------------------------
# Main preprocessing loop
# ---------------------------------------------------------------------------

def run_preprocessing(normalize: bool = False, workers: int = DOWNLOAD_WORKERS) -> pd.DataFrame:
    """
    Process all valid clips: extract thumbnails and optionally normalise video.

    Args:
        normalize : Re-encode to standard resolution/FPS (slow, ~3× slower).
        workers   : Number of parallel ffmpeg processes.

    Returns:
        processed_metadata DataFrame.
    """
    THUMBNAILS_BASE.mkdir(parents=True, exist_ok=True)

    ver_csv = REPORTS_DIR / "verification_report.csv"
    if not ver_csv.exists():
        raise FileNotFoundError(
            "Run verify_videos.py first.\n"
            f"Expected: {ver_csv}"
        )

    report = pd.read_csv(ver_csv)
    valid  = report[report["status"] == "valid"].copy()

    log.info(f"Processing {len(valid):,} valid clips")
    log.info(f"  Thumbnails : enabled")
    log.info(f"  Normalize  : {'enabled (slow)' if normalize else 'disabled'}")
    log.info(f"  Workers    : {workers}")

    work = [(row, normalize) for _, row in valid.iterrows()]
    records = []

    with tqdm(total=len(work), desc="Preprocessing", unit="clip", dynamic_ncols=True) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_process_clip, work):
                records.append(result)
                pbar.update(1)

    proc_df = pd.DataFrame(records)
    out_csv = METADATA_DIR / "processed_metadata.csv"
    proc_df.to_csv(out_csv, index=False)

    thumb_ok   = (proc_df["thumb_status"] == "ok").sum()
    thumb_fail = len(proc_df) - thumb_ok
    log.info(f"\nThumbnails : {thumb_ok:,} ok, {thumb_fail} failed")
    if normalize:
        norm_ok   = (proc_df["norm_status"] == "ok").sum()
        norm_fail = (proc_df["norm_status"] == "failed").sum()
        log.info(f"Normalize  : {norm_ok:,} ok, {norm_fail} failed")
    log.info(f"Metadata   : {out_csv}")

    return proc_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract thumbnails and optionally normalise VIDI clips"
    )
    parser.add_argument("--normalize", action="store_true",
                        help="Re-encode clips to standard 256×256 @ 16fps (slow)")
    parser.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS,
                        help="Parallel ffmpeg workers")
    args = parser.parse_args()
    run_preprocessing(normalize=args.normalize, workers=args.workers)
