"""
create_splits.py — Stage 2 of the VIDI pipeline.

Creates stratified train / val / test splits from master_annotations.csv.

Split strategy
--------------
Splitting is done at the *YouTube video* level, not the clip level.
This prevents data leakage where two clips from the same video end up
in different splits (which would give inflated validation accuracy because
the model has seen the same scene during training).

Algorithm:
  1. Collect unique YouTube IDs per fine-grained label.
  2. For each label, split its video IDs 70 / 15 / 15 using sklearn's
     train_test_split with a fixed random seed.
  3. All clips from each video ID are assigned to the same split.
  4. Clips that belong to a YouTube ID appearing in multiple labels
     are assigned to the split determined by the majority label.

Outputs:
    metadata/train_splits.csv
    metadata/val_splits.csv
    metadata/test_splits.csv
    metadata/all_splits.csv   (union with `split` column)

Usage:
    python create_splits.py
    python create_splits.py --seed 123
"""

import sys
import logging
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    METADATA_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def _split_video_ids(
    video_ids: list[str], seed: int
) -> tuple[set[str], set[str], set[str]]:
    """
    Split a list of YouTube video IDs into train / val / test sets.
    Returns three sets of IDs.
    """
    if len(video_ids) < 3:
        # Too few videos to split — assign all to train
        return set(video_ids), set(), set()

    val_test_ratio = VAL_RATIO + TEST_RATIO
    train_ids, temp_ids = train_test_split(
        video_ids, test_size=val_test_ratio, random_state=seed, shuffle=True
    )

    # Relative val size within the temp pool
    if len(temp_ids) < 2:
        return set(train_ids), set(temp_ids), set()

    val_frac = VAL_RATIO / val_test_ratio
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=(1 - val_frac), random_state=seed, shuffle=True
    )

    return set(train_ids), set(val_ids), set(test_ids)


def create_splits(seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """
    Build train / val / test split CSVs from master_annotations.csv.

    Returns a dict {"train": df, "val": df, "test": df}.
    """
    master_csv = METADATA_DIR / "master_annotations.csv"
    if not master_csv.exists():
        raise FileNotFoundError(
            f"master_annotations.csv not found. Run generate_metadata.py first.\n"
            f"Expected: {master_csv}"
        )

    master = pd.read_csv(master_csv)
    log.info(f"Loaded {len(master):,} clips from {master_csv.name}")

    # ── Per-label video-level split ──────────────────────────────────────────
    # Build a mapping youtube_id → split assignment (handle multi-label IDs
    # with majority vote at the end).
    id_split_votes: dict[str, list[str]] = {}

    for label, label_df in master.groupby("label"):
        video_ids = label_df["youtube_id"].unique().tolist()
        tr_ids, va_ids, te_ids = _split_video_ids(video_ids, seed)

        for yt_id in tr_ids:
            id_split_votes.setdefault(yt_id, []).append("train")
        for yt_id in va_ids:
            id_split_votes.setdefault(yt_id, []).append("val")
        for yt_id in te_ids:
            id_split_votes.setdefault(yt_id, []).append("test")

    # Majority vote for each YouTube ID
    id_to_split: dict[str, str] = {
        yt_id: Counter(votes).most_common(1)[0][0]
        for yt_id, votes in id_split_votes.items()
    }

    # ── Assign split column ───────────────────────────────────────────────────
    master["split"] = master["youtube_id"].map(id_to_split).fillna("train")

    # ── Save individual split files ───────────────────────────────────────────
    splits: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        df = master[master["split"] == split_name].copy().reset_index(drop=True)
        splits[split_name] = df
        out = METADATA_DIR / f"{split_name}_splits.csv"
        df.to_csv(out, index=False)
        log.info(
            f"  {split_name:<6}: {len(df):>5} clips  "
            f"from {df['youtube_id'].nunique():>4} videos  "
            f"across {df['label'].nunique()} categories"
        )

    # ── Save unified file ─────────────────────────────────────────────────────
    all_splits = pd.concat(splits.values(), ignore_index=True)
    all_splits_csv = METADATA_DIR / "all_splits.csv"
    all_splits.to_csv(all_splits_csv, index=False)
    log.info(f"Unified splits saved → {all_splits_csv}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    _validate_no_leakage(splits)

    return splits


def _validate_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Assert that no YouTube video ID appears in more than one split."""
    id_sets = {name: set(df["youtube_id"]) for name, df in splits.items()}
    names = list(id_sets.keys())
    ok = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = id_sets[names[i]] & id_sets[names[j]]
            if overlap:
                log.error(
                    f"Data leakage! {len(overlap)} YouTube IDs appear in "
                    f"both {names[i]} and {names[j]}: {list(overlap)[:5]} ..."
                )
                ok = False
    if ok:
        log.info("  ✓ No data leakage — every YouTube ID appears in exactly one split")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for VIDI"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    args = parser.parse_args()
    create_splits(seed=args.seed)
