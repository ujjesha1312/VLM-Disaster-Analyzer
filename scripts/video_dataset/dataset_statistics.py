"""
dataset_statistics.py — Stage 6 of the VIDI pipeline.

Produces a comprehensive set of reports and visualisations by reading
master_annotations.csv and (optionally) verification_report.csv.

Outputs:
    reports/dataset_summary.md           — Markdown overview (human-readable)
    reports/class_distribution.csv       — Per fine-grained category stats
    reports/broad_class_distribution.csv — Per broad category stats
    reports/storage_report.csv           — Disk usage by split × label
    reports/data_quality_report.csv      — Download status per label
    reports/class_distribution.png       — Dual bar-chart (fine + broad)

Usage:
    python dataset_statistics.py
    python dataset_statistics.py --no-plots   # skip matplotlib (headless safe)
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import METADATA_DIR, REPORTS_DIR, PROCESSED_BASE

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _fmt_dur(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

def _class_distribution(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (fine_dist, broad_dist) DataFrames."""
    fine_dist = (
        master.groupby("label")
        .agg(
            broad_category  = ("broad_category", "first"),
            clip_count      = ("clip_id",         "count"),
            unique_videos   = ("youtube_id",       "nunique"),
            total_duration  = ("duration_sec",     "sum"),
            avg_duration    = ("duration_sec",     "mean"),
            min_duration    = ("duration_sec",     "min"),
            max_duration    = ("duration_sec",     "max"),
        )
        .reset_index()
        .sort_values("clip_count", ascending=False)
    )
    fine_dist["avg_duration"]   = fine_dist["avg_duration"].round(2)
    fine_dist["total_dur_min"]  = (fine_dist["total_duration"] / 60).round(2)

    broad_dist = (
        master.groupby("broad_category")
        .agg(
            clip_count      = ("clip_id",     "count"),
            unique_videos   = ("youtube_id",  "nunique"),
            fine_categories = ("label",       "nunique"),
            total_duration  = ("duration_sec","sum"),
            avg_duration    = ("duration_sec","mean"),
        )
        .reset_index()
        .sort_values("clip_count", ascending=False)
    )
    broad_dist["avg_duration"]  = broad_dist["avg_duration"].round(2)
    broad_dist["total_dur_min"] = (broad_dist["total_duration"] / 60).round(2)

    return fine_dist, broad_dist


def _storage_report(processed_base: Path) -> pd.DataFrame:
    """Walk processed_videos/ and tally file sizes per split × label."""
    records = []
    if not processed_base.exists():
        return pd.DataFrame(
            columns=["split", "label", "clip_count", "size_bytes", "size_human"]
        )

    for split_dir in sorted(processed_base.iterdir()):
        if not split_dir.is_dir():
            continue
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            files  = list(label_dir.glob("*.mp4"))
            total  = sum(f.stat().st_size for f in files)
            records.append({
                "split":      split_dir.name,
                "label":      label_dir.name,
                "clip_count": len(files),
                "size_bytes": total,
                "size_human": _fmt_bytes(total),
            })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["split", "label", "clip_count", "size_bytes", "size_human"]
    )


def _quality_report(ver_csv: Path) -> pd.DataFrame | None:
    if not ver_csv.exists():
        return None
    ver = pd.read_csv(ver_csv)
    pivot = (
        ver.groupby("label")["status"]
        .value_counts()
        .unstack(fill_value=0)
    )
    # Ensure expected columns are present
    for col in ("valid", "missing", "corrupted", "too_small", "duration_mismatch"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total"]        = pivot.sum(axis=1)
    pivot["success_rate"] = (pivot.get("valid", 0) / pivot["total"] * 100).round(1)
    return pivot.reset_index()


def _plot_distributions(
    fine_dist: pd.DataFrame, broad_dist: pd.DataFrame, out_path: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plot generation")
        return

    top_n = min(20, len(fine_dist))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(6, top_n * 0.38)))
    fig.suptitle("VIDI Dataset — Class Distribution", fontsize=14, fontweight="bold", y=1.01)

    # Fine-grained top-N (horizontal bar)
    plot_fine = fine_dist.head(top_n).sort_values("clip_count")
    bars1 = ax1.barh(plot_fine["label"], plot_fine["clip_count"], color="#E65D20", alpha=0.85)
    ax1.bar_label(bars1, padding=3, fontsize=8)
    ax1.set_xlabel("Number of Clips")
    ax1.set_title(f"Top {top_n} Fine-Grained Categories (of {len(fine_dist)})")
    ax1.set_xlim(0, plot_fine["clip_count"].max() * 1.18)
    ax1.tick_params(axis="y", labelsize=8)

    # Broad categories
    plot_broad = broad_dist.sort_values("clip_count")
    bars2 = ax2.barh(plot_broad["broad_category"], plot_broad["clip_count"],
                     color="#2B7BE5", alpha=0.85)
    ax2.bar_label(bars2, padding=3, fontsize=9)
    ax2.set_xlabel("Number of Clips")
    ax2.set_title("Broad Disaster Categories")
    ax2.set_xlim(0, plot_broad["clip_count"].max() * 1.18)
    ax2.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Plot saved  : {out_path}")


# ---------------------------------------------------------------------------
# Markdown summary builder
# ---------------------------------------------------------------------------

def _build_summary_md(
    master: pd.DataFrame,
    fine_dist: pd.DataFrame,
    broad_dist: pd.DataFrame,
    storage_df: pd.DataFrame,
    quality_df: pd.DataFrame | None,
) -> str:
    now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_clips  = len(master)
    total_vids   = master["youtube_id"].nunique()
    total_dur_h  = master["duration_sec"].sum() / 3600
    avg_dur_s    = master["duration_sec"].mean()
    n_fine       = master["label"].nunique()
    n_broad      = master["broad_category"].nunique()
    total_bytes  = storage_df["size_bytes"].sum() if not storage_df.empty else 0

    dl_rate_str = ""
    if quality_df is not None:
        rate = quality_df["success_rate"].mean()
        dl_rate_str = f"| Download Success Rate | **{rate:.1f}%** |"

    # ── Header ────────────────────────────────────────────────────────────────
    lines = [
        "# VIDI Dataset — Summary Report",
        "",
        f"**Generated:** {now}  ",
        "**Source:** https://github.com/vididataset/VIDI  ",
        "**Paper:** Sesver et al. — *VIDI: A Video Dataset of Incidents*, IEEE IVMSP 2022  ",
        "",
        "---",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Video Clips | **{total_clips:,}** |",
        f"| Unique YouTube Sources | **{total_vids:,}** |",
        f"| Fine-Grained Categories | **{n_fine}** |",
        f"| Broad Disaster Categories | **{n_broad}** |",
        f"| Total Annotated Duration | **{_fmt_dur(master['duration_sec'].sum())}** ({total_dur_h:.1f} h) |",
        f"| Average Clip Length | **{avg_dur_s:.1f} s** |",
        f"| Processed Storage | **{_fmt_bytes(total_bytes)}** |",
    ]
    if dl_rate_str:
        lines.append(dl_rate_str)

    # ── Broad category table ──────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Broad Disaster Category Distribution",
        "",
        "| # | Category | Clips | Videos | Sub-categories | Duration (min) |",
        "|---|----------|-------|--------|---------------|----------------|",
    ]
    for i, r in enumerate(broad_dist.itertuples(), 1):
        lines.append(
            f"| {i} | {r.broad_category} | {r.clip_count:,} | {r.unique_videos:,} "
            f"| {r.fine_categories} | {r.total_dur_min:.1f} |"
        )

    # ── Fine-grained top-20 ───────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Top 20 Fine-Grained Categories",
        "",
        "| Category | Broad | Clips | Videos | Avg Duration (s) |",
        "|----------|-------|-------|--------|-----------------|",
    ]
    for r in fine_dist.head(20).itertuples():
        lines.append(
            f"| {r.label} | {r.broad_category} | {r.clip_count:,} "
            f"| {r.unique_videos:,} | {r.avg_duration:.1f} |"
        )

    # ── Language distribution ─────────────────────────────────────────────────
    lang_dist = master["lang"].value_counts()
    lines += [
        "",
        "---",
        "",
        "## Language Distribution",
        "",
        "| Language | Clips | % of Total |",
        "|----------|-------|-----------|",
    ]
    for lang, cnt in lang_dist.items():
        label = lang if lang else "(unspecified)"
        pct   = cnt / total_clips * 100
        lines.append(f"| {label} | {cnt:,} | {pct:.1f}% |")

    # ── Split distribution ────────────────────────────────────────────────────
    all_splits_csv = METADATA_DIR / "all_splits.csv"
    if all_splits_csv.exists():
        splits_df = pd.read_csv(all_splits_csv)
        lines += [
            "",
            "---",
            "",
            "## Dataset Splits",
            "",
            "| Split | Clips | Videos | Categories | % of Total |",
            "|-------|-------|--------|-----------|-----------|",
        ]
        for split in ("train", "val", "test"):
            sub = splits_df[splits_df["split"] == split]
            pct = len(sub) / len(splits_df) * 100
            lines.append(
                f"| {split:<6} | {len(sub):,} | {sub['youtube_id'].nunique():,} "
                f"| {sub['label'].nunique()} | {pct:.1f}% |"
            )

    # ── Storage ───────────────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Storage Requirements",
        "",
        "| Component | Size |",
        "|-----------|------|",
        f"| Processed clips (trimmed, H.264 CRF 22) | {_fmt_bytes(total_bytes)} |",
        f"| Raw downloads (est. ~8× processed) | ~{_fmt_bytes(total_bytes * 8)} |",
        f"| Thumbnails (256×256 JPEG, est.) | ~{_fmt_bytes(total_clips * 50 * 1024)} |",
        "",
        "> **Tip:** English-only subset (~47% of clips) reduces storage proportionally.",
        "",
        "---",
        "",
        "## Integration with VLM Disaster Analyzer",
        "",
        "The VIDI broad categories map directly to the existing image pipeline categories:",
        "",
        "| VIDI Broad Category | Image Pipeline Class |",
        "|--------------------|--------------------|",
        "| Flood | Water Disaster |",
        "| Wildfire | Wild Fire / Urban Fire |",
        "| Earthquake | Earthquake |",
        "| Landslide | Land Slide |",
        "| Infrastructure Damage | Infrastructure Damage |",
        "| Transportation Accident | Human Damage |",
        "",
        "---",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@inproceedings{sesver2022vidi,",
        '  title     = {VIDI: A Video Dataset of Incidents},',
        '  author    = {Sesver, Bahar and others},',
        '  booktitle = {IEEE 14th Image, Video, and Multisignal Processing Workshop (IVMSP)},',
        '  year      = {2022},',
        '  url       = {https://arxiv.org/abs/2205.13277}',
        "}",
        "```",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_reports(plots: bool = True) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    master_csv = METADATA_DIR / "master_annotations.csv"
    if not master_csv.exists():
        raise FileNotFoundError(
            f"Run generate_metadata.py first.\nExpected: {master_csv}"
        )

    master = pd.read_csv(master_csv)
    log.info(f"Loaded {len(master):,} clips from master_annotations.csv")

    # ── Compute tables ────────────────────────────────────────────────────────
    fine_dist, broad_dist = _class_distribution(master)
    storage_df            = _storage_report(PROCESSED_BASE)
    quality_df            = _quality_report(REPORTS_DIR / "verification_report.csv")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    fine_dist.to_csv(REPORTS_DIR / "class_distribution.csv",       index=False)
    broad_dist.to_csv(REPORTS_DIR / "broad_class_distribution.csv", index=False)
    storage_df.to_csv(REPORTS_DIR / "storage_report.csv",          index=False)
    if quality_df is not None:
        quality_df.to_csv(REPORTS_DIR / "data_quality_report.csv", index=False)
    log.info("  CSVs saved  : class_distribution, broad_class_distribution, storage_report, data_quality_report")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plots:
        _plot_distributions(fine_dist, broad_dist, REPORTS_DIR / "class_distribution.png")

    # ── Markdown summary ──────────────────────────────────────────────────────
    md = _build_summary_md(master, fine_dist, broad_dist, storage_df, quality_df)
    summary_out = REPORTS_DIR / "dataset_summary.md"
    summary_out.write_text(md, encoding="utf-8")
    log.info(f"  Summary     : {summary_out}")

    # ── Print key stats to stdout ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VIDI DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total clips         : {len(master):,}")
    print(f"  Unique YouTube IDs  : {master['youtube_id'].nunique():,}")
    print(f"  Fine categories     : {master['label'].nunique()}")
    print(f"  Broad categories    : {master['broad_category'].nunique()}")
    print(f"  Total duration      : {master['duration_sec'].sum()/3600:.1f} h")
    print(f"  Avg clip duration   : {master['duration_sec'].mean():.1f} s")
    total_bytes = storage_df["size_bytes"].sum() if not storage_df.empty else 0
    print(f"  Processed storage   : {_fmt_bytes(total_bytes)}")
    print("=" * 60)
    print("\nTop 10 broad categories:")
    for _, r in broad_dist.head(10).iterrows():
        bar = "█" * (r["clip_count"] // 50)
        print(f"  {r['broad_category']:<30} {r['clip_count']:>5} clips  {bar}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate VIDI dataset statistics and reports"
    )
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib chart generation")
    args = parser.parse_args()
    generate_reports(plots=not args.no_plots)
