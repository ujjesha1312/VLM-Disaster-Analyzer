"""
download_replacements.py — Fill gaps left by failed clips in the first download run.

Reads download_report.csv to identify which categories are short, excludes all
previously-failed YouTube IDs, selects fresh VIDI clips, and downloads them.
Appends results to download_report.csv.
"""

from __future__ import annotations

import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    METADATA_DIR, RAW_VIDEOS_ROOT, TARGET_CATEGORIES,
    VIDEOS_PER_CATEGORY, LABEL_PRIORITY,
    MAX_CLIPS_PER_YT_ID, MIN_FILE_SIZE_KB,
)
from download_videos import (
    fetch_vidi_csv, _clean_vidi_df, _clip_output_path,
    _download_yt_video, _trim_clip, _cached_raw_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("replace")


def _count_good_files(cat: str) -> tuple[int, dict[str, int], set[tuple]]:
    """
    Count clips actually on disk for a category.
    Returns (count, {yt_id: clip_count}, {(yt_id, start, end)} existing keys).
    Files under MIN_FILE_SIZE_KB are ignored (treat as corrupt/incomplete).
    """
    vid_dir = RAW_VIDEOS_ROOT / cat
    yt_id_counts: dict[str, int] = {}
    existing_keys: set[tuple] = set()
    if not vid_dir.exists():
        return 0, yt_id_counts, existing_keys
    for p in vid_dir.glob("*.mp4"):
        if p.stat().st_size < MIN_FILE_SIZE_KB * 1024:
            continue
        # File names are {yt_id}_{start}_{end}.mp4 where yt_id may contain underscores.
        # Split from the right so we always get exactly (yt_id, start, end).
        parts = p.stem.rsplit("_", 2)
        if len(parts) >= 3:
            yt_id = parts[0]
            try:
                start, end = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            yt_id_counts[yt_id] = yt_id_counts.get(yt_id, 0) + 1
            existing_keys.add((yt_id, start, end))
    return len(existing_keys), yt_id_counts, existing_keys


def run_replacements() -> None:
    report_path = METADATA_DIR / "download_report.csv"
    if not report_path.exists():
        log.error("download_report.csv not found — run download_videos.py first")
        return

    report = pd.read_csv(report_path)

    # All YouTube IDs that failed at any point — never retry these
    failed_ids: set[str] = set(
        report.loc[report["status"] != "success", "youtube_id"].tolist()
    )

    log.info("=" * 60)
    log.info("  REPLACEMENT DOWNLOAD — filling gaps (disk-authoritative)")
    log.info(f"  Excluded IDs ({len(failed_ids)}): {sorted(failed_ids)}")
    log.info("=" * 60)

    new_results: list[dict] = []

    for cat in TARGET_CATEGORIES:
        # Use actual on-disk files as the ground truth, not the report
        have, yt_id_counts, existing_keys = _count_good_files(cat)
        need = VIDEOS_PER_CATEGORY - have
        if need <= 0:
            log.info(f"[{cat}] Complete on disk ({have}/15) — skipping")
            continue

        log.info(f"\n[{cat}] On disk: {have}/15 — downloading {need} replacements")

        candidates: list[pd.Series] = []
        for label in LABEL_PRIORITY[cat]:
            raw = fetch_vidi_csv(label)
            if raw is None:
                continue
            filtered = _clean_vidi_df(raw, label)
            if filtered.empty:
                continue
            filtered = filtered.copy()
            filtered["_is_en"] = (filtered["lang"] == "EN").astype(int)
            filtered = (
                filtered
                .sort_values(["_is_en", "duration_s"], ascending=[False, True])
                .reset_index(drop=True)
            )
            for _, row in filtered.iterrows():
                candidates.append(row.copy())

        # Select new clips: skip failed IDs, already-on-disk clips, diversity cap
        selected: list[pd.Series] = []
        for row in candidates:
            if len(selected) >= need:
                break
            yt_id = row["youtube_id"]
            key   = (yt_id, int(row["time_start"]), int(row["time_end"]))
            if yt_id in failed_ids:
                continue
            if key in existing_keys:
                continue  # this exact clip is already on disk
            if yt_id_counts.get(yt_id, 0) >= MAX_CLIPS_PER_YT_ID:
                continue
            row["research_category"] = cat
            row["vidi_label"]        = row.get("label", "unknown")
            yt_id_counts[yt_id]      = yt_id_counts.get(yt_id, 0) + 1
            existing_keys.add(key)
            selected.append(row)

        unique_yt = len({r["youtube_id"] for r in selected})
        log.info(f"  Selected {len(selected)} new candidates from {unique_yt} YouTube IDs")

        if not selected:
            log.warning(f"  [{cat}] No replacement candidates — category will remain short")
            continue

        # Download and trim each selected clip
        for row in selected:
            yt_id = str(row["youtube_id"])
            out   = _clip_output_path(row)

            if out.exists() and out.stat().st_size >= MIN_FILE_SIZE_KB * 1024:
                log.info(f"  [SKIP] {out.name} exists (race condition)")
                continue

            dl_ok, dl_msg = _download_yt_video(yt_id)
            raw = _cached_raw_path(yt_id)

            if not dl_ok or raw is None:
                log.warning(f"  [FAIL] {yt_id} — {dl_msg[:100]}")
                failed_ids.add(yt_id)  # don't retry this ID again this session
                new_results.append({
                    **row.to_dict(),
                    "status": "download_failed", "message": dl_msg,
                    "output_path": None, "timestamp": datetime.now().isoformat(),
                })
                continue

            trim_ok, trim_msg = _trim_clip(
                raw, out, int(row["time_start"]), int(row["time_end"])
            )
            status = "success" if trim_ok else "trim_failed"
            if trim_ok:
                log.info(f"  [OK]   {out.name}")
            else:
                log.warning(f"  [TRIM FAIL] {out.name} — {trim_msg[:80]}")

            new_results.append({
                **row.to_dict(),
                "status": status, "message": trim_msg,
                "output_path": str(out) if trim_ok else None,
                "timestamp": datetime.now().isoformat(),
            })

            if raw and raw.exists():
                try:
                    raw.unlink()
                except Exception:
                    pass

    # Append only genuinely new results to the report
    if new_results:
        new_df   = pd.DataFrame(new_results)
        combined = pd.concat([report, new_df], ignore_index=True)
        combined.to_csv(report_path, index=False)
        ok   = (new_df["status"] == "success").sum()
        fail = (new_df["status"] != "success").sum()
        log.info("\n── Replacement complete ────────────────────────────────")
        log.info(f"  New downloads : {ok} succeeded, {fail} failed")
        log.info(f"  Report updated: {report_path}")
    else:
        log.info("Nothing to download — all categories already complete on disk.")

    # Final tally from disk (authoritative)
    log.info("\n── Final on-disk clip counts ───────────────────────────")
    grand = 0
    for cat in TARGET_CATEGORIES:
        n, _, _ = _count_good_files(cat)
        flag  = "OK" if n >= VIDEOS_PER_CATEGORY else f"SHORT by {VIDEOS_PER_CATEGORY - n}"
        log.info(f"  {cat:<12}: {n:2d}/15  [{flag}]")
        grand += n
    log.info(f"  {'TOTAL':<12}: {grand}/75")


if __name__ == "__main__":
    run_replacements()
