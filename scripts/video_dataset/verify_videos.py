"""
verify_videos.py — Stage 4 of the VIDI pipeline.

For every clip listed in all_splits.csv this script verifies that:
  1. The expected file exists at processed_videos/{split}/{label}/
  2. File size is above the minimum healthy threshold
  3. ffprobe reports a valid video stream (not corrupted)
  4. Actual duration is within ±DURATION_TOLERANCE of the annotated duration

Status codes written to the verification report:
  valid            — file passes all checks
  missing          — file does not exist (not yet downloaded / skipped)
  too_small        — file exists but below MIN_FILE_SIZE_KB
  corrupted        — ffprobe cannot parse the file
  duration_mismatch— actual duration deviates > DURATION_TOLERANCE from expected

Outputs:
    reports/verification_report.csv

Usage:
    python verify_videos.py
    python verify_videos.py --redownload   # re-queue failed clips
"""

import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    METADATA_DIR, PROCESSED_BASE, REPORTS_DIR,
    MIN_FILE_SIZE_KB,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DURATION_TOLERANCE = 2.0   # seconds — allow this much deviation from expected


# ---------------------------------------------------------------------------
# ffprobe wrapper
# ---------------------------------------------------------------------------

def ffprobe_info(path: Path) -> Optional[dict]:
    """
    Run ffprobe on *path* and return parsed JSON, or None on any failure.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass
    return None


# ---------------------------------------------------------------------------
# Per-clip verification
# ---------------------------------------------------------------------------

def verify_clip(row: pd.Series) -> dict:
    """
    Verify a single clip annotation and return a result dict.
    """
    safe_label   = str(row["label"]).replace("/", "_").replace(" ", "_")
    clip_name    = f"{row['youtube_id']}_{row['time_start']}_{row['time_end']}.mp4"
    expected_dur = float(int(row["time_end"]) - int(row["time_start"]))
    path         = PROCESSED_BASE / row["split"] / safe_label / clip_name

    result = {
        "clip_id":          row.get("clip_id", ""),
        "youtube_id":       row["youtube_id"],
        "label":            row["label"],
        "split":            row["split"],
        "expected_path":    str(path),
        "exists":           False,
        "size_kb":          0.0,
        "ffprobe_ok":       False,
        "actual_duration":  None,
        "expected_duration": expected_dur,
        "duration_ok":      False,
        "codec":            None,
        "resolution":       None,
        "status":           "missing",
    }

    # ── Check existence ───────────────────────────────────────────────────────
    if not path.exists():
        return result

    size_kb = path.stat().st_size / 1024
    result.update(exists=True, size_kb=round(size_kb, 2))

    if size_kb < MIN_FILE_SIZE_KB:
        result["status"] = "too_small"
        return result

    # ── ffprobe ────────────────────────────────────────────────────────────────
    info = ffprobe_info(path)
    if info is None:
        result["status"] = "corrupted"
        return result

    result["ffprobe_ok"] = True

    # Video stream metadata
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if vstreams:
        vs = vstreams[0]
        result["codec"]      = vs.get("codec_name")
        result["resolution"] = f"{vs.get('width', '?')}x{vs.get('height', '?')}"

    # Duration
    raw_dur = info.get("format", {}).get("duration", "-1")
    actual_dur = float(raw_dur) if raw_dur and raw_dur != "N/A" else -1.0
    result["actual_duration"] = round(actual_dur, 3)

    if actual_dur > 0:
        if abs(actual_dur - expected_dur) <= DURATION_TOLERANCE:
            result["duration_ok"] = True
            result["status"]      = "valid"
        else:
            result["status"] = "duration_mismatch"
    else:
        result["status"] = "no_duration"

    return result


# ---------------------------------------------------------------------------
# Main verification loop
# ---------------------------------------------------------------------------

def run_verification(redownload_failed: bool = False) -> pd.DataFrame:
    """
    Verify every clip in all_splits.csv and write reports/verification_report.csv.

    Args:
        redownload_failed: If True, trigger re-download of any non-valid clips.

    Returns:
        The full verification report as a DataFrame.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_splits_csv = METADATA_DIR / "all_splits.csv"
    if not all_splits_csv.exists():
        raise FileNotFoundError(
            f"Run generate_metadata.py and create_splits.py first.\n"
            f"Expected: {all_splits_csv}"
        )

    df = pd.read_csv(all_splits_csv)
    log.info(f"Verifying {len(df):,} clips …")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying", unit="clip"):
        records.append(verify_clip(row))

    report = pd.DataFrame(records)
    out_csv = REPORTS_DIR / "verification_report.csv"
    report.to_csv(out_csv, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    counts = report["status"].value_counts()
    total  = len(report)

    log.info("\n── Verification Summary ──────────────────────────────────────")
    for status in ("valid", "missing", "corrupted", "too_small",
                   "duration_mismatch", "no_duration"):
        n = counts.get(status, 0)
        pct = n / total * 100 if total else 0
        flag = "✓" if status == "valid" else "✗"
        log.info(f"  {flag} {status:<20}: {n:>5} ({pct:5.1f}%)")
    log.info(f"\n  Total clips   : {total:,}")
    log.info(f"  Report saved  : {out_csv}")

    # Per-split summary
    if "split" in report.columns:
        log.info("\n── By Split ──────────────────────────────────────────────────")
        for split in ("train", "val", "test"):
            sub = report[report["split"] == split]
            if len(sub) == 0:
                continue
            v = (sub["status"] == "valid").sum()
            log.info(f"  {split:<6}: {v:>5} / {len(sub):>5} valid ({v/len(sub)*100:.1f}%)")

    # ── Optional re-download of failed clips ──────────────────────────────────
    if redownload_failed:
        failed = report[report["status"] != "valid"]
        if len(failed) == 0:
            log.info("All clips valid — nothing to re-download.")
        else:
            failed_ids = failed["youtube_id"].unique().tolist()
            log.info(f"\nRe-downloading {len(failed_ids)} YouTube IDs "
                     f"({len(failed)} clips) …")
            # Lazy import to avoid circular dependency
            from download_videos import download_all
            download_all(
                # Pass only the YouTube IDs that failed
                # (done by writing a temp filtered CSV)
            )
            # Simple re-run: let download_all skip already-valid files
            download_all()

    return report


# ---------------------------------------------------------------------------
# Corruption cleanup helper
# ---------------------------------------------------------------------------

def remove_corrupted(report: pd.DataFrame | None = None, dry_run: bool = False) -> int:
    """
    Delete files flagged as corrupted or too_small from processed_videos/.

    Args:
        report  : DataFrame from run_verification (loads from CSV if None).
        dry_run : If True, print what would be deleted but don't delete.

    Returns:
        Number of files deleted (or that would be deleted in dry-run).
    """
    if report is None:
        ver_csv = REPORTS_DIR / "verification_report.csv"
        if not ver_csv.exists():
            log.error("Run run_verification() first.")
            return 0
        report = pd.read_csv(ver_csv)

    bad = report[report["status"].isin(["corrupted", "too_small"])].copy()
    log.info(f"Found {len(bad)} corrupted/too-small files to remove.")

    removed = 0
    for _, row in bad.iterrows():
        p = Path(row["expected_path"])
        if p.exists():
            if dry_run:
                log.info(f"  [DRY-RUN] Would delete: {p}")
            else:
                p.unlink()
                log.debug(f"  Deleted: {p}")
            removed += 1

    if not dry_run:
        log.info(f"Removed {removed} corrupted files.")
    return removed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify integrity of downloaded VIDI clips"
    )
    parser.add_argument("--redownload", action="store_true",
                        help="Re-download any clips that failed verification")
    parser.add_argument("--remove-corrupted", action="store_true",
                        help="Delete files flagged as corrupted or too_small")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --remove-corrupted: print but don't delete")
    args = parser.parse_args()

    report = run_verification(redownload_failed=args.redownload)
    if args.remove_corrupted:
        remove_corrupted(report=report, dry_run=args.dry_run)
