"""
download_videos.py — Stage 1: targeted 75-video VIDI download.

Two-phase strategy
------------------
1. Fetch only the 14 VIDI annotation CSVs relevant to our 5 research
   categories (instead of all 43) and select 15 high-quality clips per
   category using a priority-ordered label waterfall.
2. Download unique YouTube videos via yt-dlp → temporary cache, trim each
   annotated clip with ffmpeg → raw_videos/{category}/{yt_id}_{s}_{e}.mp4,
   then delete the cached full video to save space.

Outputs
-------
    datasets/video_dataset/
        metadata/
            annotations/{label}.csv       — cached VIDI CSVs (14 files)
            selected_clips.csv            — 75 selected clips with VIDI metadata
            download_report.csv           — per-clip download + trim result
        raw_videos/
            flood/                        — 15 trimmed .mp4 clips
            wildfire/
            earthquake/
            landslide/
            cyclone/

Usage
-----
    # Preview selection without downloading:
    python download_videos.py --dry-run

    # Full run:
    python download_videos.py

    # Reduce workers on slow connections:
    python download_videos.py --workers 2

    # Skip reselection if selected_clips.csv already exists:
    python download_videos.py --skip-selection
"""

from __future__ import annotations

import sys
import json
import logging
import argparse
import subprocess
import concurrent.futures
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    VIDI_REPO_RAW, LABEL_PRIORITY, VIDI_TO_RESEARCH,
    RAW_VIDEOS_ROOT, METADATA_DIR, YT_CACHE_DIR,
    MIN_DURATION_S, MAX_DURATION_S, MAX_CLIPS_PER_YT_ID,
    VIDEOS_PER_CATEGORY, TARGET_CATEGORIES,
    MAX_RESOLUTION, DOWNLOAD_WORKERS, RETRY_ATTEMPTS, RETRY_DELAY_S,
    MIN_FILE_SIZE_KB, YT_TIMEOUT_S,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ── Stage 1A: Fetch VIDI annotation CSVs ──────────────────────────────────────

def fetch_vidi_csv(label: str) -> Optional[pd.DataFrame]:
    """
    Download one VIDI annotation CSV from GitHub and cache it locally.
    Re-uses cached file on subsequent runs.
    Returns parsed DataFrame or None on failure.
    """
    cache_dir = METADATA_DIR / "annotations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / f"{label}.csv"

    if local.exists() and local.stat().st_size > 10:
        try:
            df = pd.read_csv(local)
            log.debug(f"  [{label}] Loaded from cache ({len(df)} rows)")
            return df
        except Exception:
            pass  # fall through to re-download

    url = f"{VIDI_REPO_RAW}/{label}.csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        local.write_bytes(resp.content)
        df = pd.read_csv(local)
        log.info(f"  [{label}] Fetched {len(df)} clips from VIDI")
        return df
    except requests.HTTPError as e:
        log.warning(f"  [{label}] HTTP {e.response.status_code}: {url}")
    except Exception as e:
        log.warning(f"  [{label}] Fetch failed: {e}")
    return None


# ── Stage 1B: Apply quality filters to a VIDI DataFrame ────────────────────────

def _clean_vidi_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Normalise and quality-filter a raw VIDI annotation DataFrame.
    Returns filtered DataFrame or empty DataFrame if required columns are missing.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    required = {"youtube_id", "time_start", "time_end"}
    if not required.issubset(df.columns):
        log.warning(f"  [{label}] Missing columns: {required - set(df.columns)}")
        return pd.DataFrame()

    df["youtube_id"] = df["youtube_id"].astype(str).str.strip()
    df["time_start"] = pd.to_numeric(df["time_start"], errors="coerce").fillna(0).astype(int)
    df["time_end"]   = pd.to_numeric(df["time_end"],   errors="coerce").fillna(0).astype(int)
    df["duration_s"] = df["time_end"] - df["time_start"]
    df["lang"]       = df.get("lang", pd.Series([""] * len(df))).astype(str).str.upper().str.strip()
    df["label"]      = label

    # Quality filters
    mask = (
        (df["youtube_id"].str.len() >= 5)
        & (df["duration_s"] >= MIN_DURATION_S)
        & (df["duration_s"] <= MAX_DURATION_S)
        & (df["time_end"] > df["time_start"])
    )
    return df[mask].reset_index(drop=True)


# ── Stage 1C: Select 15 clips per category ────────────────────────────────────

def select_clips_for_category(category: str, n: int = VIDEOS_PER_CATEGORY) -> pd.DataFrame:
    """
    Select exactly n clips for one research category using a priority waterfall
    over VIDI fine-grained labels.

    Selection criteria (in order):
      1. Valid duration (MIN_DURATION_S to MAX_DURATION_S)
      2. English language preferred (falls back to all languages)
      3. Diversity: at most MAX_CLIPS_PER_YT_ID clips per YouTube ID
      4. Prefer shorter clips within quality range (faster download + processing)

    Returns DataFrame with selected clips and research_category column.
    """
    labels = LABEL_PRIORITY[category]
    collected: list[pd.DataFrame] = []
    yt_id_counts: dict[str, int] = {}
    total = 0

    for label in labels:
        if total >= n:
            break
        raw = fetch_vidi_csv(label)
        if raw is None:
            continue
        filtered = _clean_vidi_df(raw, label)
        if filtered.empty:
            continue

        # Sort: English first, then by duration ascending (prefer concise clips)
        filtered = filtered.copy()
        filtered["_is_en"] = (filtered["lang"] == "EN").astype(int)
        filtered = (
            filtered
            .sort_values(["_is_en", "duration_s"], ascending=[False, True])
            .drop(columns=["_is_en"])
            .reset_index(drop=True)
        )

        for _, row in filtered.iterrows():
            if total >= n:
                break
            yt_id = row["youtube_id"]
            if yt_id_counts.get(yt_id, 0) >= MAX_CLIPS_PER_YT_ID:
                continue

            row = row.copy()
            row["research_category"] = category
            row["vidi_label"]        = label
            yt_id_counts[yt_id]      = yt_id_counts.get(yt_id, 0) + 1
            collected.append(row.to_frame().T)
            total += 1

    if not collected:
        log.warning(f"  [{category}] No clips found across labels: {labels}")
        return pd.DataFrame()

    result = pd.concat(collected, ignore_index=True)
    result["clip_id"] = (
        result["research_category"] + "_"
        + result["youtube_id"] + "_"
        + result["time_start"].astype(str) + "_"
        + result["time_end"].astype(str)
    )
    log.info(
        f"  [{category}] Selected {len(result)}/{n} clips "
        f"from {result['youtube_id'].nunique()} unique YouTube IDs"
    )
    return result


def build_selection() -> pd.DataFrame:
    """
    Build the complete 75-clip selection across all 5 research categories.
    Saves the selection to metadata/selected_clips.csv.

    Returns the selection DataFrame.
    """
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("  VIDI Clip Selection — 5 categories × 15 clips = 75 total")
    log.info("=" * 60)

    frames = []
    for cat in TARGET_CATEGORIES:
        frames.append(select_clips_for_category(cat))

    selection = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    out = METADATA_DIR / "selected_clips.csv"
    selection.to_csv(out, index=False)

    log.info("\n── Selection summary ─────────────────────────────────────")
    for cat in TARGET_CATEGORIES:
        n = (selection["research_category"] == cat).sum()
        yt_unique = selection.loc[selection["research_category"] == cat, "youtube_id"].nunique()
        log.info(f"  {cat:<12} : {n:2d} clips from {yt_unique} YouTube IDs")
    log.info(f"\n  TOTAL       : {len(selection)} clips")
    log.info(f"  Unique YT IDs: {selection['youtube_id'].nunique()}")
    log.info(f"  Saved → {out}")

    return selection


# ── Stage 2: Download YouTube videos ──────────────────────────────────────────

def _cached_raw_path(yt_id: str) -> Optional[Path]:
    """Return path of a cached raw YouTube video, or None."""
    for ext in ("mp4", "webm", "mkv"):
        p = YT_CACHE_DIR / f"{yt_id}.{ext}"
        if p.exists() and p.stat().st_size >= MIN_FILE_SIZE_KB * 1024:
            return p
    return None


def _download_yt_video(yt_id: str, dry_run: bool = False) -> tuple[bool, str]:
    """
    Download full YouTube video at ≤MAX_RESOLUTION to YT_CACHE_DIR.
    Skips silently if already cached.
    Returns (success, message).
    """
    if _cached_raw_path(yt_id):
        return True, "Cached"
    if dry_run:
        return True, "DRY-RUN"

    YT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url      = f"https://www.youtube.com/watch?v={yt_id}"
    out_tmpl = str(YT_CACHE_DIR / f"{yt_id}.%(ext)s")
    fmt = (
        f"bestvideo[height<={MAX_RESOLUTION}][ext=mp4]+bestaudio[ext=m4a]"
        f"/bestvideo[height<={MAX_RESOLUTION}]+bestaudio"
        f"/best[height<={MAX_RESOLUTION}][ext=mp4]"
        f"/best[height<={MAX_RESOLUTION}]"
    )
    cmd = [
        sys.executable, "-m", "yt_dlp", "--format", fmt,
        "--merge-output-format", "mp4",
        "--no-playlist", "--no-warnings", "--quiet",
        "--no-overwrites", "--socket-timeout", "30",
        "--retries", "3", "-o", out_tmpl, url,
    ]
    last_err = ""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=YT_TIMEOUT_S)
            if r.returncode == 0 and _cached_raw_path(yt_id):
                return True, "Downloaded"
            last_err = (r.stderr or r.stdout).strip()[-300:]
        except subprocess.TimeoutExpired:
            last_err = f"Timeout ({YT_TIMEOUT_S}s)"
        except FileNotFoundError:
            return False, f"Python not found at {sys.executable}"
        except Exception as e:
            last_err = str(e)
        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_DELAY_S * (2 ** (attempt - 1)))

    return False, f"Failed after {RETRY_ATTEMPTS} attempts: {last_err}"


# ── Stage 3: Trim clips with ffmpeg ───────────────────────────────────────────

def _clip_output_path(row: pd.Series) -> Path:
    category = str(row["research_category"]).lower()
    yt_id    = str(row["youtube_id"])
    start    = int(row["time_start"])
    end      = int(row["time_end"])
    return RAW_VIDEOS_ROOT / category / f"{yt_id}_{start}_{end}.mp4"


def _trim_clip(
    raw: Path, out: Path, start: int, end: int, dry_run: bool = False
) -> tuple[bool, str]:
    """
    Trim [start, end] seconds from raw video → out using fast ffmpeg seeking.
    Returns (success, message).
    """
    if out.exists() and out.stat().st_size >= MIN_FILE_SIZE_KB * 1024:
        return True, "Already exists"
    if dry_run:
        return True, "DRY-RUN"

    out.parent.mkdir(parents=True, exist_ok=True)
    duration = max(end - start, 1)
    cmd = [
        "ffmpeg",
        "-ss",  str(start),        # fast input-seek
        "-i",   str(raw),
        "-t",   str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        "-y", str(out),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and out.exists() and out.stat().st_size >= MIN_FILE_SIZE_KB * 1024:
            return True, "Trimmed"
        err = (r.stderr or "").strip()[-200:]
        return False, f"ffmpeg error: {err}"
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timeout (120s)"
    except FileNotFoundError:
        return False, "ffmpeg not found — run: apt-get install -y ffmpeg"
    except Exception as e:
        return False, str(e)


# ── Worker: process one YouTube ID and all its selected clips ─────────────────

def _process_yt_group(args: tuple) -> list[dict]:
    """
    Download one YouTube video, trim all its selected clips, then clean up
    the cached raw video to save disk space.
    Returns list of result dicts (one per clip).
    """
    yt_id, group_df, dry_run = args
    results = []

    dl_ok, dl_msg = _download_yt_video(yt_id, dry_run=dry_run)
    raw = _cached_raw_path(yt_id)

    if not dl_ok or (raw is None and not dry_run):
        log.warning(f"  ✗ {yt_id} — {dl_msg}")
        for _, row in group_df.iterrows():
            results.append({**row.to_dict(), "status": "download_failed", "message": dl_msg})
        return results

    log.debug(f"  ✓ {yt_id} ({len(group_df)} clips) — {dl_msg}")

    for _, row in group_df.iterrows():
        out = _clip_output_path(row)
        raw_path = raw if raw else YT_CACHE_DIR / f"{yt_id}.mp4"
        trim_ok, trim_msg = _trim_clip(
            raw_path, out, int(row["time_start"]), int(row["time_end"]), dry_run
        )
        results.append({
            **row.to_dict(),
            "status":      "success" if trim_ok else "trim_failed",
            "message":     trim_msg,
            "output_path": str(out) if trim_ok else None,
            "timestamp":   datetime.now().isoformat(),
        })

    # Delete full YouTube video to reclaim disk space (clips are now trimmed)
    if raw and not dry_run:
        try:
            raw.unlink()
        except Exception:
            pass

    return results


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_download(
    selection: Optional[pd.DataFrame] = None,
    workers: int = DOWNLOAD_WORKERS,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Download and trim all selected clips.

    Args:
        selection : Output of build_selection(). Loads from selected_clips.csv if None.
        workers   : Number of parallel download threads.
        dry_run   : If True, simulate without downloading.

    Returns download report DataFrame (also saved to metadata/download_report.csv).
    """
    if selection is None:
        sel_csv = METADATA_DIR / "selected_clips.csv"
        if not sel_csv.exists():
            raise FileNotFoundError(
                f"selected_clips.csv not found. Run build_selection() first.\n"
                f"Expected: {sel_csv}"
            )
        selection = pd.read_csv(sel_csv)

    # Ensure category output folders exist
    for cat in TARGET_CATEGORIES:
        (RAW_VIDEOS_ROOT / cat).mkdir(parents=True, exist_ok=True)

    groups = list(selection.groupby("youtube_id"))
    log.info("=" * 60)
    log.info(f"  Download: {len(selection)} clips | {len(groups)} YouTube IDs")
    log.info(f"  Workers : {workers}  |  Dry-run: {dry_run}")
    log.info("=" * 60)

    all_results: list[dict] = []
    work = [(yt_id, grp, dry_run) for yt_id, grp in groups]

    with tqdm(total=len(groups), desc="Downloading", unit="video") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(_process_yt_group, w): w[0] for w in work}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    all_results.extend(fut.result())
                except Exception as exc:
                    yt_id = futures[fut]
                    log.error(f"Worker error [{yt_id}]: {exc}")
                pbar.update(1)

    report = pd.DataFrame(all_results)
    report_path = METADATA_DIR / "download_report.csv"
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    ok   = (report["status"] == "success").sum()   if not report.empty else 0
    fail = (report["status"] != "success").sum()   if not report.empty else 0
    log.info("\n── Download complete ─────────────────────────────────────")
    log.info(f"  Succeeded : {ok}")
    log.info(f"  Failed    : {fail}")
    log.info(f"  Report    → {report_path}")

    if fail > 0:
        failed_cats = report.loc[report["status"] != "success", "research_category"].value_counts()
        log.warning(f"  Failed by category: {failed_cats.to_dict()}")

    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download 75 VIDI research videos (5 categories × 15 clips)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--workers",         type=int,  default=DOWNLOAD_WORKERS,
                   help="Parallel download threads")
    p.add_argument("--dry-run",         action="store_true",
                   help="Preview selection and simulate download without fetching")
    p.add_argument("--skip-selection",  action="store_true",
                   help="Use existing selected_clips.csv (skip VIDI CSV fetching)")
    args = p.parse_args()

    if args.skip_selection:
        sel = None
    else:
        sel = build_selection()

    if args.dry_run:
        log.info("\n[DRY-RUN] Selection preview only — no videos downloaded.")
        if sel is not None:
            log.info(sel[["research_category", "vidi_label",
                           "youtube_id", "duration_s", "lang"]].to_string(index=False))
    else:
        run_download(selection=sel, workers=args.workers, dry_run=False)
