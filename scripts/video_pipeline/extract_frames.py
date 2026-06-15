"""
extract_frames.py — Stage 2: uniform frame extraction from 75 research videos.

For each video in raw_videos/{category}/, samples FRAMES_PER_VIDEO frames
uniformly across the clip duration and saves them as JPEGs to
extracted_frames/{category}/{video_stem}/frame_001.jpg … frame_008.jpg

Outputs
-------
    datasets/video_dataset/
        extracted_frames/
            flood/
                {yt_id}_{s}_{e}/
                    frame_001.jpg … frame_008.jpg
            wildfire/ … (same structure)
        metadata/
            frame_manifest.csv   — one row per frame

Usage
-----
    python extract_frames.py                  # extract 8 frames per video
    python extract_frames.py --frames 12      # extract 12 frames per video
    python extract_frames.py --overwrite      # re-extract even if already done
"""

from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_VIDEOS_ROOT, FRAMES_ROOT, METADATA_DIR,
    TARGET_CATEGORIES, FRAMES_PER_VIDEO, FRAME_FORMAT, FRAME_QUALITY,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ENCODE_PARAMS    = [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY] if FRAME_FORMAT == "jpg" else []


# ── Core extraction helpers ────────────────────────────────────────────────────

def sample_frames_uniform(video_path: Path, n: int) -> list[tuple[int, any]]:
    """
    Open a video file and sample n frames uniformly across its duration.

    Returns a list of (original_frame_index, BGR_array) tuples.
    Returns [] if the file cannot be opened or has fewer than 1 readable frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"  Cannot open: {video_path.name}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        log.warning(f"  Zero frames reported for: {video_path.name}")
        return []

    n       = min(n, total)
    indices = (
        [int(i * (total - 1) / (n - 1)) for i in range(n)]
        if n > 1 else [total // 2]
    )

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))

    cap.release()
    return frames


def extract_video(
    video_path: Path,
    category:   str,
    n:          int  = FRAMES_PER_VIDEO,
    overwrite:  bool = False,
) -> list[dict]:
    """
    Extract frames from a single video file.

    Returns list of metadata dicts (one per saved frame).
    Skips extraction and returns cached metadata if frames already exist
    and overwrite=False.
    """
    stem    = video_path.stem
    out_dir = FRAMES_ROOT / category / stem

    # Return cached results without re-extracting
    if out_dir.exists() and not overwrite:
        existing = sorted(out_dir.glob(f"*.{FRAME_FORMAT}"))
        if existing:
            return [
                {
                    "video_stem":    stem,
                    "category":      category,
                    "video_path":    str(video_path),
                    "frame_path":    str(p),
                    "frame_filename": p.name,
                    "frame_number":  i + 1,
                    "frame_index":   int(p.stem.split("_")[-1]) if "_" in p.stem else i,
                    "status":        "cached",
                }
                for i, p in enumerate(existing)
            ]

    out_dir.mkdir(parents=True, exist_ok=True)
    sampled = sample_frames_uniform(video_path, n)

    if not sampled:
        log.warning(f"  No frames extracted from {video_path.name}")
        return [{
            "video_stem":    stem,
            "category":      category,
            "video_path":    str(video_path),
            "frame_path":    None,
            "frame_filename": None,
            "frame_number":  0,
            "frame_index":   0,
            "status":        "failed",
        }]

    records = []
    for frame_num, (frame_idx, frame) in enumerate(sampled, start=1):
        frame_name = f"frame_{frame_num:03d}.{FRAME_FORMAT}"
        frame_path = out_dir / frame_name

        if cv2.imwrite(str(frame_path), frame, ENCODE_PARAMS):
            records.append({
                "video_stem":    stem,
                "category":      category,
                "video_path":    str(video_path),
                "frame_path":    str(frame_path),
                "frame_filename": frame_name,
                "frame_number":  frame_num,
                "frame_index":   frame_idx,
                "status":        "extracted",
            })
        else:
            log.warning(f"  Failed to write {frame_path}")

    return records


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_extraction(
    n:         int  = FRAMES_PER_VIDEO,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Extract frames from all 75 research videos.

    Args:
        n         : Number of frames to extract per video.
        overwrite : Re-extract even if frames already exist.

    Returns frame manifest DataFrame (also saved to metadata/frame_manifest.csv).
    """
    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    all_records:   list[dict] = []
    videos_seen:   int        = 0
    videos_skipped: int       = 0

    log.info("=" * 60)
    log.info(f"  Frame Extraction — {n} frames per video")
    log.info("=" * 60)

    for category in TARGET_CATEGORIES:
        vid_dir = RAW_VIDEOS_ROOT / category
        if not vid_dir.exists():
            log.warning(f"  [{category}] raw_videos folder missing: {vid_dir}")
            continue

        videos = sorted(
            p for p in vid_dir.iterdir()
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )

        if not videos:
            log.warning(f"  [{category}] No videos found in {vid_dir}")
            continue

        log.info(f"  [{category}] {len(videos)} videos")
        (FRAMES_ROOT / category).mkdir(parents=True, exist_ok=True)

        for vpath in tqdm(videos, desc=f"{category:12}", unit="video", leave=False):
            records = extract_video(vpath, category, n=n, overwrite=overwrite)
            all_records.extend(records)
            videos_seen += 1
            if records and records[0]["status"] == "cached":
                videos_skipped += 1

    manifest = pd.DataFrame(all_records)
    manifest_path = METADATA_DIR / "frame_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    extracted  = (manifest["status"] == "extracted").sum() if not manifest.empty else 0
    cached     = (manifest["status"] == "cached").sum()    if not manifest.empty else 0
    failed     = (manifest["status"] == "failed").sum()    if not manifest.empty else 0

    log.info("\n── Extraction summary ────────────────────────────────────")
    log.info(f"  Videos processed  : {videos_seen}")
    log.info(f"  Frames extracted  : {extracted}")
    log.info(f"  Frames cached     : {cached}")
    log.info(f"  Videos failed     : {failed}")
    log.info(f"  Frame manifest    → {manifest_path}")

    return manifest


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract frames from 75 research videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames",    type=int,  default=FRAMES_PER_VIDEO,
                   help="Number of frames to extract per video")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract frames even if they already exist")
    args = p.parse_args()
    run_extraction(n=args.frames, overwrite=args.overwrite)
