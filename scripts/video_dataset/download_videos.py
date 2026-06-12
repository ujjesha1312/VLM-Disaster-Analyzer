"""
download_videos.py — Stage 3 of the VIDI pipeline.

Two-phase download strategy:
  Phase 1 — Download each unique YouTube video once at ≤720p to raw_videos/
  Phase 2 — Trim each annotated clip from the raw video with ffmpeg

This approach is significantly more efficient than VIDI's original download.py
which downloads the video once per clip (re-downloading the same source video
dozens of times for categories with many clips from the same YouTube ID).

Requires:
    pip install yt-dlp tqdm pandas
    apt-get install -y ffmpeg   (Linux / Google Colab)

Outputs:
    raw_videos/{youtube_id}.mp4              — full downloaded source video
    processed_videos/{split}/{label}/
        {youtube_id}_{start}_{end}.mp4       — trimmed clip

    logs/download_{timestamp}.jsonl          — JSONL log of every clip result

Usage:
    python download_videos.py                          # all splits
    python download_videos.py --splits train val       # specific splits
    python download_videos.py --labels wildfire flood  # specific categories
    python download_videos.py --workers 4              # reduce parallelism
    python download_videos.py --english-only           # EN clips only (~47%)
    python download_videos.py --dry-run                # preview without downloading
"""

import sys
import json
import time
import logging
import argparse
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    METADATA_DIR, RAW_VIDEOS, PROCESSED_BASE, LOGS_DIR,
    MAX_RESOLUTION, DOWNLOAD_WORKERS, RETRY_ATTEMPTS, RETRY_DELAY_BASE,
    MIN_FILE_SIZE_KB, VIDEO_CODEC, AUDIO_CODEC, CRF,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup (file + console)
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"download_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return log_file


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def raw_video_path(yt_id: str) -> Optional[Path]:
    """
    Return the existing raw video file path for a YouTube ID, or None.
    Checks common extensions in order of preference.
    """
    for ext in ("mp4", "webm", "mkv"):
        p = RAW_VIDEOS / f"{yt_id}.{ext}"
        if p.exists() and p.stat().st_size >= MIN_FILE_SIZE_KB * 1024:
            return p
    return None


def clip_output_path(yt_id: str, label: str, time_start: int, time_end: int, split: str) -> Path:
    safe_label = label.replace("/", "_").replace(" ", "_")
    return PROCESSED_BASE / split / safe_label / f"{yt_id}_{time_start}_{time_end}.mp4"


# ---------------------------------------------------------------------------
# Phase 1: Download raw video
# ---------------------------------------------------------------------------

def download_raw_video(yt_id: str, dry_run: bool = False) -> tuple[bool, str]:
    """
    Download the full YouTube video at ≤MAX_RESOLUTION to raw_videos/.
    Returns (success: bool, message: str).
    Skips if a valid file already exists.
    """
    existing = raw_video_path(yt_id)
    if existing:
        return True, f"Cached: {existing.name}"

    if dry_run:
        return True, "DRY-RUN (skipped)"

    RAW_VIDEOS.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={yt_id}"
    out_tmpl = str(RAW_VIDEOS / f"{yt_id}.%(ext)s")

    # Prefer MP4 container for maximum ffmpeg compatibility
    fmt_selector = (
        f"bestvideo[height<={MAX_RESOLUTION}][ext=mp4]+bestaudio[ext=m4a]"
        f"/bestvideo[height<={MAX_RESOLUTION}]+bestaudio[ext=m4a]"
        f"/best[height<={MAX_RESOLUTION}][ext=mp4]"
        f"/best[height<={MAX_RESOLUTION}]"
    )

    cmd = [
        "yt-dlp",
        "--format",               fmt_selector,
        "--merge-output-format",  "mp4",
        "--no-playlist",
        "--no-warnings",
        "--quiet",
        "--no-overwrites",
        "--socket-timeout",       "30",
        "--retries",              "3",
        "-o",                     out_tmpl,
        url,
    ]

    last_error = ""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0 and raw_video_path(yt_id):
                return True, "Downloaded"
            last_error = (result.stderr or result.stdout).strip()[-200:]
        except subprocess.TimeoutExpired:
            last_error = "Timeout (600s)"
        except FileNotFoundError:
            return False, "yt-dlp not found — run: pip install yt-dlp"
        except Exception as e:
            last_error = str(e)

        if attempt < RETRY_ATTEMPTS:
            delay = RETRY_DELAY_BASE * (2 ** (attempt - 1))   # 5s, 10s, 20s
            log.debug(f"  [{yt_id}] retry {attempt}/{RETRY_ATTEMPTS} in {delay}s")
            time.sleep(delay)

    return False, f"Failed after {RETRY_ATTEMPTS} attempts: {last_error}"


# ---------------------------------------------------------------------------
# Phase 2: Trim clip with ffmpeg
# ---------------------------------------------------------------------------

def trim_clip(
    raw: Path,
    out: Path,
    time_start: int,
    time_end: int,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """
    Trim [time_start, time_end] from raw video and encode to output path.
    Uses fast input-seeking (-ss before -i) with duration flag (-t).
    Returns (success: bool, message: str).
    """
    if out.exists() and out.stat().st_size >= MIN_FILE_SIZE_KB * 1024:
        return True, "Already exists"

    if dry_run:
        return True, "DRY-RUN (skipped)"

    out.parent.mkdir(parents=True, exist_ok=True)
    duration = max(time_end - time_start, 1)

    cmd = [
        "ffmpeg",
        "-ss",  str(time_start),   # input seek (fast approximate)
        "-i",   str(raw),
        "-t",   str(duration),     # duration from seek point
        "-c:v", VIDEO_CODEC,
        "-preset", "fast",
        "-crf", str(CRF),
        "-c:a", AUDIO_CODEC,
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        "-y",
        str(out),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and out.exists() and out.stat().st_size >= MIN_FILE_SIZE_KB * 1024:
            return True, "Trimmed"
        err = (result.stderr or "").strip()[-200:]
        return False, f"ffmpeg error: {err}"
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timeout (120s)"
    except FileNotFoundError:
        return False, "ffmpeg not found — install ffmpeg"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Worker: process one YouTube video and all its annotated clips
# ---------------------------------------------------------------------------

def _process_video_group(
    args: tuple[str, pd.DataFrame, object, bool],
) -> dict:
    """
    Worker function executed in ThreadPoolExecutor.
    Downloads one YouTube video, then trims all clips from it.
    Returns a summary dict with per-clip results.
    """
    yt_id, clips_df, log_fh, dry_run = args
    results = {"youtube_id": yt_id, "clips": []}

    raw_ok, raw_msg = download_raw_video(yt_id, dry_run=dry_run)
    raw = raw_video_path(yt_id)

    if not raw_ok or (raw is None and not dry_run):
        log.warning(f"✗ {yt_id} — {raw_msg}")
        for _, row in clips_df.iterrows():
            entry = _make_log_entry(row, status="download_failed", message=raw_msg)
            results["clips"].append(entry)
            _write_log(log_fh, entry)
        return results

    log.info(f"✓ {yt_id} ({len(clips_df)} clips) — {raw_msg}")

    for _, row in clips_df.iterrows():
        out = clip_output_path(
            yt_id, row["label"], int(row["time_start"]), int(row["time_end"]), row["split"]
        )
        raw_path = raw if raw else RAW_VIDEOS / f"{yt_id}.mp4"  # dry-run placeholder
        trim_ok, trim_msg = trim_clip(
            raw_path, out, int(row["time_start"]), int(row["time_end"]), dry_run=dry_run
        )

        entry = _make_log_entry(
            row,
            status="success" if trim_ok else "trim_failed",
            message=trim_msg,
            output_path=str(out) if trim_ok else None,
        )
        results["clips"].append(entry)
        _write_log(log_fh, entry)

    return results


def _make_log_entry(
    row: pd.Series,
    status: str,
    message: str,
    output_path: str | None = None,
) -> dict:
    return {
        "clip_id":      row.get("clip_id", ""),
        "youtube_id":   row["youtube_id"],
        "label":        row["label"],
        "split":        row.get("split", ""),
        "time_start":   int(row["time_start"]),
        "time_end":     int(row["time_end"]),
        "output_path":  output_path,
        "status":       status,
        "message":      message,
        "timestamp":    datetime.now().isoformat(),
    }


def _write_log(fh, entry: dict) -> None:
    fh.write(json.dumps(entry) + "\n")
    fh.flush()


# ---------------------------------------------------------------------------
# Main download orchestrator
# ---------------------------------------------------------------------------

def download_all(
    splits_filter: list[str] | None = None,
    labels_filter: list[str] | None = None,
    lang_filter: list[str] | None = None,
    workers: int = DOWNLOAD_WORKERS,
    dry_run: bool = False,
) -> dict:
    """
    Download and trim all VIDI video clips.

    Args:
        splits_filter : Only process clips from these splits (e.g. ["train"]).
        labels_filter : Only process these fine-grained labels.
        lang_filter   : Only process these language codes (e.g. ["EN"]).
        workers       : Number of parallel download threads.
        dry_run       : If True, simulate without actually downloading.

    Returns:
        Summary dict with total_ok / total_fail counts.
    """
    log_file = _setup_logging(LOGS_DIR)
    log.info("=" * 60)
    log.info("VIDI Download Pipeline")
    log.info(f"Dry-run: {dry_run}")
    log.info("=" * 60)

    # ── Load split manifest ───────────────────────────────────────────────────
    all_splits_csv = METADATA_DIR / "all_splits.csv"
    if not all_splits_csv.exists():
        raise FileNotFoundError(
            "Run generate_metadata.py and create_splits.py first.\n"
            f"Expected: {all_splits_csv}"
        )

    df = pd.read_csv(all_splits_csv)

    if splits_filter:
        df = df[df["split"].isin(splits_filter)]
    if labels_filter:
        df = df[df["label"].isin(labels_filter)]
    if lang_filter:
        df = df[df["lang"].str.upper().isin([l.upper() for l in lang_filter])]

    log.info(f"Clips to process : {len(df):,}")
    log.info(f"Unique YouTube IDs: {df['youtube_id'].nunique():,}")
    log.info(f"Parallel workers : {workers}")

    if len(df) == 0:
        log.warning("No clips match the given filters. Nothing to do.")
        return {"total_ok": 0, "total_fail": 0}

    # ── Group by YouTube ID ────────────────────────────────────────────────────
    groups = list(df.groupby("youtube_id"))
    total_ok = total_fail = 0

    jsonl_path = LOGS_DIR / f"download_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as log_fh:
        work = [(yt_id, group_df, log_fh, dry_run) for yt_id, group_df in groups]

        with tqdm(total=len(groups), desc="Videos", unit="vid", dynamic_ncols=True) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_id = {
                    executor.submit(_process_video_group, w): w[0] for w in work
                }
                for fut in concurrent.futures.as_completed(future_to_id):
                    yt_id = future_to_id[fut]
                    try:
                        result = fut.result()
                        ok   = sum(1 for c in result["clips"] if c["status"] == "success")
                        fail = len(result["clips"]) - ok
                        total_ok   += ok
                        total_fail += fail
                    except Exception as exc:
                        log.error(f"Unexpected error for {yt_id}: {exc}")
                        total_fail += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=total_ok, fail=total_fail, refresh=False)

    log.info("\n" + "=" * 60)
    log.info(f"Download complete")
    log.info(f"  Succeeded : {total_ok:,} clips")
    log.info(f"  Failed    : {total_fail:,} clips")
    log.info(f"  JSONL log : {jsonl_path}")
    log.info("=" * 60)

    return {"total_ok": total_ok, "total_fail": total_fail, "log": str(jsonl_path)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and trim VIDI video clips",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--splits",  nargs="+", metavar="SPLIT",
                        help="Splits to process: train val test")
    parser.add_argument("--labels",  nargs="+", metavar="LABEL",
                        help="Fine-grained labels to process (subset download)")
    parser.add_argument("--lang",    nargs="+", metavar="LANG",
                        help="Language codes to include (e.g. EN ES)")
    parser.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS,
                        help="Number of parallel download workers")
    parser.add_argument("--english-only", action="store_true",
                        help="Shortcut: download only EN clips (~47% of dataset)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate pipeline without downloading")
    args = parser.parse_args()

    lang = ["EN"] if args.english_only else args.lang
    download_all(
        splits_filter=args.splits,
        labels_filter=args.labels,
        lang_filter=lang,
        workers=args.workers,
        dry_run=args.dry_run,
    )
