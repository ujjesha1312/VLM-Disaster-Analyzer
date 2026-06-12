"""
generate_metadata.py — Stage 1 of the VIDI pipeline.

Downloads all 43 annotation CSV files directly from the VIDI GitHub repository,
merges them into a single master DataFrame, enriches with derived columns, and
saves the result to metadata/master_annotations.csv.

Outputs:
    annotations/{category}.csv          — raw per-category annotation files
    metadata/master_annotations.csv     — merged, enriched manifest (all 43 categories)
    metadata/category_mapping.csv       — fine-grained → broad category lookup table

Usage:
    python generate_metadata.py
    python generate_metadata.py --categories wildfire earthquake  # subset
"""

import sys
import logging
import argparse
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    VIDI_REPO_RAW, VIDI_CATEGORIES, CATEGORY_MAP,
    ANNOTATIONS_DIR, METADATA_DIR, LANGUAGE_FILTER,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_category_csv(category: str) -> pd.DataFrame | None:
    """
    Download one VIDI category CSV from GitHub and cache it locally.
    Returns the parsed DataFrame, or None on failure.
    """
    local = ANNOTATIONS_DIR / f"{category}.csv"

    # Re-use cached file if already present
    if local.exists() and local.stat().st_size > 10:
        try:
            return pd.read_csv(local)
        except Exception:
            pass  # fall through to re-download

    url = f"{VIDI_REPO_RAW}/{category}.csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        local.write_bytes(resp.content)
        df = pd.read_csv(local)
        log.debug(f"  Fetched {category}: {len(df)} rows")
        return df
    except requests.HTTPError as e:
        log.warning(f"  HTTP {e.response.status_code} for {category}: {url}")
    except Exception as e:
        log.warning(f"  Failed to fetch {category}: {e}")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_master(categories: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch and merge annotation CSVs into a single master DataFrame.

    Args:
        categories: Subset of category slugs to process. Defaults to all 43.

    Returns:
        master DataFrame saved to metadata/master_annotations.csv
    """
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    targets = categories or VIDI_CATEGORIES
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for cat in tqdm(targets, desc="Fetching annotations", unit="cat"):
        df = _fetch_category_csv(cat)
        if df is not None:
            frames.append(df)
        else:
            failed.append(cat)

    if not frames:
        raise RuntimeError(
            "No annotation CSVs could be fetched. Check your internet connection."
        )

    # ── Merge ─────────────────────────────────────────────────────────────────
    master = pd.concat(frames, ignore_index=True)

    # ── Normalise columns ──────────────────────────────────────────────────────
    master.columns = master.columns.str.strip().str.lower()
    for col in ("label", "youtube_id", "lang"):
        master[col] = master[col].astype(str).str.strip()

    master["time_start"] = pd.to_numeric(master["time_start"], errors="coerce").fillna(0).astype(int)
    master["time_end"]   = pd.to_numeric(master["time_end"],   errors="coerce").fillna(0).astype(int)

    # ── Fix known VIDI typos ───────────────────────────────────────────────────
    master["label"] = master["label"].replace({"oil_spil": "oil_spill"})

    # ── Derived columns ────────────────────────────────────────────────────────
    master["duration_sec"] = (master["time_end"] - master["time_start"]).clip(lower=0)

    # Unique clip identifier: ytid_start_end
    master["clip_id"] = (
        master["youtube_id"] + "_"
        + master["time_start"].astype(str) + "_"
        + master["time_end"].astype(str)
    )

    # Broad disaster category (falls back to title-cased fine label if unmapped)
    master["broad_category"] = (
        master["label"]
        .map(CATEGORY_MAP)
        .fillna(master["label"].str.replace("_", " ").str.title())
    )

    # Normalise language code
    master["lang"] = master["lang"].replace({"nan": "", "None": ""}).str.upper()

    # ── Optional language filter ───────────────────────────────────────────────
    if LANGUAGE_FILTER:
        before = len(master)
        master = master[master["lang"].isin([l.upper() for l in LANGUAGE_FILTER])]
        log.info(f"Language filter {LANGUAGE_FILTER}: {before} → {len(master)} clips")

    # ── Drop degenerate clips ─────────────────────────────────────────────────
    # Remove rows where start >= end or duration == 0
    bad = master["duration_sec"] <= 0
    if bad.any():
        log.warning(f"Dropping {bad.sum()} rows with zero/negative duration")
        master = master[~bad]

    # ── Persist ───────────────────────────────────────────────────────────────
    master_csv = METADATA_DIR / "master_annotations.csv"
    master.to_csv(master_csv, index=False)
    log.info(f"Master annotations saved → {master_csv} ({len(master):,} clips)")

    # Category mapping reference table
    mapping_df = pd.DataFrame(
        sorted(CATEGORY_MAP.items()), columns=["fine_category", "broad_category"]
    )
    mapping_df.to_csv(METADATA_DIR / "category_mapping.csv", index=False)

    # ── Summary print ─────────────────────────────────────────────────────────
    log.info("\n── Dataset overview ─────────────────────────────────────────")
    log.info(f"  Total clips          : {len(master):,}")
    log.info(f"  Unique YouTube IDs   : {master['youtube_id'].nunique():,}")
    log.info(f"  Fine-grained labels  : {master['label'].nunique()}")
    log.info(f"  Broad categories     : {master['broad_category'].nunique()}")
    log.info(f"  Total duration       : {master['duration_sec'].sum()/3600:.1f} h")
    log.info(f"  Avg clip duration    : {master['duration_sec'].mean():.1f} s")
    log.info(f"  Languages            : {sorted(master['lang'].unique())}")

    if failed:
        log.warning(f"\n  Failed categories ({len(failed)}): {failed}")

    return master


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch and merge VIDI annotation CSVs"
    )
    parser.add_argument(
        "--categories", nargs="+", metavar="CATEGORY",
        help="Process only these categories (default: all 43)",
    )
    args = parser.parse_args()
    build_master(categories=args.categories)
