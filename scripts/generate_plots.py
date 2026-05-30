#!/usr/bin/env python3
"""
generate_plots.py — Research-quality visual analytics for disaster evaluation results.

Reads outputs/results.csv (produced by batch_evaluate.py) and generates five
publication-ready visualizations saved to outputs/plots/:

  Plot                          File
  ────────────────────────────────────────────────────
  1. Confusion Matrix           confusion_matrix.png
  2. Category Accuracy Chart    category_accuracy.png
  3. Average Confidence Chart   average_confidence.png
  4. Dataset Distribution Pie   dataset_distribution.png
  5. Confidence Histogram       confidence_distribution.png

Usage (from project root, after batch_evaluate.py has run):

    python scripts/generate_plots.py

Custom paths:

    python scripts/generate_plots.py \\
        --results outputs/results.csv \\
        --output  outputs/plots
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency guard — must be checked before any heavy imports
# ---------------------------------------------------------------------------

_missing: list[str] = []

for _pkg, _install in [
    ("pandas",  "pandas"),
    ("numpy",   "numpy"),
    ("seaborn", "seaborn"),
    ("sklearn", "scikit-learn"),
]:
    try:
        __import__(_pkg)
    except ImportError:
        _missing.append(f"pip install {_install}")

try:
    import matplotlib
    # Non-interactive backend — MUST be set before pyplot is imported.
    matplotlib.use("Agg")
except ImportError:
    _missing.append("pip install matplotlib")

if _missing:
    sys.exit(
        "[ERROR] Missing packages. Install with:\n  "
        + "\n  ".join(_missing)
    )

# Safe to import everything now.
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT        = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "outputs" / "results.csv"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "outputs" / "plots"

DPI      = 150      # high enough for presentations, not bloated
TITLE_FS = 15
LABEL_FS = 12
TICK_FS  = 10
ANNOT_FS = 9

# 12 perceptually distinct colours — automatically wraps for larger sets
_PALETTE = [
    "#1565C0",  # deep blue
    "#C62828",  # deep red
    "#EF6C00",  # orange
    "#2E7D32",  # deep green
    "#6A1B9A",  # purple
    "#00838F",  # teal
    "#4E342E",  # brown
    "#37474F",  # slate
    "#AD1457",  # raspberry
    "#827717",  # olive
    "#BF360C",  # rust
    "#00695C",  # sea-green
]

SEP = "─" * 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_style() -> None:
    """Apply whitegrid style regardless of matplotlib/seaborn version."""
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def _color_map(labels: list[str]) -> dict[str, str]:
    """Assign a consistent colour to each label (alphabetically ordered)."""
    return {
        label: _PALETTE[i % len(_PALETTE)]
        for i, label in enumerate(sorted(labels))
    }


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure at DPI, close it to free memory, and print confirmation."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(csv_path: Path) -> pd.DataFrame:
    """Load results.csv, validate columns, and compute a 'Correct' flag."""
    if not csv_path.exists():
        sys.exit(
            f"\n[ERROR] Results file not found: {csv_path}\n"
            f"  Run batch_evaluate.py first:\n"
            f"    python scripts/batch_evaluate.py --limit 50"
        )

    df = pd.read_csv(csv_path)

    required = {"Image Name", "Actual Disaster", "Predicted Disaster", "Confidence"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns in {csv_path}: {missing}")

    if df.empty:
        sys.exit(f"[ERROR] {csv_path} is empty — nothing to plot.")

    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    df = df.dropna(subset=["Confidence"]).copy()
    df["Correct"] = (
        df["Actual Disaster"].str.strip().str.lower()
        == df["Predicted Disaster"].str.strip().str.lower()
    )

    return df


# ---------------------------------------------------------------------------
# Plot 1 — Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(df: pd.DataFrame, out: Path) -> None:
    """Row-normalised heatmap (recall per actual class)."""
    y_true     = df["Actual Disaster"]
    y_pred     = df["Predicted Disaster"]
    all_labels = sorted(set(y_true) | set(y_pred))
    n          = len(all_labels)

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Row-normalise → recall; guard against zero-sum rows
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    # Scale figure size to the number of labels
    cell  = max(0.8, 9.0 / n)
    fig_w = max(10, n * cell + 3)
    fig_h = max(8,  n * cell * 0.85 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    annot_fs = max(6, 11 - n // 3)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=all_labels,
        yticklabels=all_labels,
        linewidths=0.4,
        linecolor="white",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Recall (row normalised)", "shrink": 0.75},
        annot_kws={"size": annot_fs},
        ax=ax,
    )

    ax.set_title(
        "Confusion Matrix — CLIP Disaster Classification",
        fontsize=TITLE_FS, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Predicted Disaster", fontsize=LABEL_FS, labelpad=10)
    ax.set_ylabel("Actual Disaster",    fontsize=LABEL_FS, labelpad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=TICK_FS)
    ax.tick_params(axis="y", rotation=0,  labelsize=TICK_FS)
    ax.text(
        0.99, -0.14,
        f"n = {len(df):,} images   |   Values = row-normalised recall",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color="grey",
    )

    plt.tight_layout()
    _save(fig, out / "confusion_matrix.png")


# ---------------------------------------------------------------------------
# Plot 2 — Category Accuracy Bar Chart
# ---------------------------------------------------------------------------

def plot_category_accuracy(df: pd.DataFrame, out: Path) -> None:
    """Per-category accuracy sorted descending with an overall average line."""
    grp      = df.groupby("Actual Disaster")
    accuracy = (grp["Correct"].sum() / grp["Correct"].count() * 100).round(1)
    accuracy = accuracy.sort_values(ascending=False)
    cats     = accuracy.index.tolist()

    cmap   = _color_map(cats)
    colors = [cmap[c] for c in cats]

    fig_w = max(9, len(cats) * 0.9 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    bars = ax.bar(
        cats, accuracy.values,
        color=colors, edgecolor="white", linewidth=0.8, zorder=3,
    )

    for bar, val in zip(bars, accuracy.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=ANNOT_FS, fontweight="bold",
        )

    overall = df["Correct"].mean() * 100
    ax.axhline(
        overall, color="#D32F2F", linestyle="--", linewidth=2.0,
        label=f"Overall accuracy: {overall:.1f}%", zorder=4,
    )

    ax.set_title(
        "Classification Accuracy by Disaster Category",
        fontsize=TITLE_FS, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Disaster Category", fontsize=LABEL_FS, labelpad=8)
    ax.set_ylabel("Accuracy (%)",      fontsize=LABEL_FS, labelpad=8)
    ax.set_ylim(0, min(120, accuracy.max() + 20))
    ax.tick_params(axis="x", rotation=35, labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    _save(fig, out / "category_accuracy.png")


# ---------------------------------------------------------------------------
# Plot 3 — Average Confidence Bar Chart
# ---------------------------------------------------------------------------

def plot_average_confidence(df: pd.DataFrame, out: Path) -> None:
    """Mean confidence per category sorted descending; ±1 σ error bars."""
    grp  = df.groupby("Actual Disaster")["Confidence"]
    mean = grp.mean().sort_values(ascending=False)
    std  = grp.std().reindex(mean.index).fillna(0)
    cats = mean.index.tolist()

    cmap   = _color_map(cats)
    colors = [cmap[c] for c in cats]

    fig_w = max(9, len(cats) * 0.9 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    bars = ax.bar(
        cats, mean.values,
        color=colors, edgecolor="white", linewidth=0.8,
        yerr=std.values, capsize=5,
        error_kw={"linewidth": 1.2, "ecolor": "#444444", "zorder": 5},
        zorder=3,
    )

    for bar, val, s in zip(bars, mean.values, std.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + s + 1.5,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=ANNOT_FS, fontweight="bold",
        )

    overall_mean = df["Confidence"].mean()
    ax.axhline(
        overall_mean, color="#1565C0", linestyle="--", linewidth=2.0,
        label=f"Overall avg: {overall_mean:.1f}%", zorder=4,
    )

    ax.set_title(
        "Average CLIP Confidence Score by Disaster Category",
        fontsize=TITLE_FS, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Disaster Category",      fontsize=LABEL_FS, labelpad=8)
    ax.set_ylabel("Average Confidence (%)", fontsize=LABEL_FS, labelpad=8)
    ax.set_ylim(0, min(120, mean.max() + std.max() + 20))
    ax.tick_params(axis="x", rotation=35, labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    ax.text(
        0.99, 0.02, "Error bars = ±1 std dev",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="grey",
    )

    plt.tight_layout()
    _save(fig, out / "average_confidence.png")


# ---------------------------------------------------------------------------
# Plot 4 — Dataset Distribution Pie Chart
# ---------------------------------------------------------------------------

def plot_dataset_distribution(df: pd.DataFrame, out: Path) -> None:
    """Pie chart of actual class frequencies with external legend."""
    counts = df["Actual Disaster"].value_counts()
    cats   = counts.index.tolist()
    cmap   = _color_map(cats)
    colors = [cmap[c] for c in cats]

    # Slightly explode very small slices (<3 % of total) for readability
    threshold = counts.sum() * 0.03
    explode   = [0.06 if v < threshold else 0.0 for v in counts.values]

    fig, ax = plt.subplots(figsize=(11, 8))

    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2.0 else "",
        pctdistance=0.78,
        colors=colors,
        explode=explode,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.8},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    legend_labels = [f"{cat}  ({cnt:,})" for cat, cnt in counts.items()]
    ax.legend(
        wedges, legend_labels,
        title="Disaster Category",
        title_fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        framealpha=0.95,
        edgecolor="#cccccc",
    )

    ax.set_title(
        f"Dataset Distribution\n"
        f"({counts.sum():,} images  ·  {len(counts)} categories)",
        fontsize=TITLE_FS, fontweight="bold", pad=14,
    )
    ax.set_aspect("equal")

    plt.tight_layout()
    _save(fig, out / "dataset_distribution.png")


# ---------------------------------------------------------------------------
# Plot 5 — Confidence Distribution
# ---------------------------------------------------------------------------

def plot_confidence_histogram(df: pd.DataFrame, out: Path) -> None:
    """Left: stacked histogram (correct vs incorrect).
       Right: per-category horizontal box plots (top 8 categories)."""
    correct_conf   = df.loc[df["Correct"],  "Confidence"]
    incorrect_conf = df.loc[~df["Correct"], "Confidence"]

    top_n    = 8
    top_cats = df["Actual Disaster"].value_counts().head(top_n).index.tolist()
    subset   = df[df["Actual Disaster"].isin(top_cats)].copy()
    cat_order = sorted(top_cats)
    cmap     = _color_map(top_cats)

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(15, 6))

    # ── Left: histogram ───────────────────────────────────────────────────────
    bins = np.arange(0, 103, 3)

    if not incorrect_conf.empty:
        ax_hist.hist(
            incorrect_conf, bins=bins, alpha=0.55,
            color="#EF5350", edgecolor="white", linewidth=0.4, zorder=3,
            label=f"Incorrect  ({len(incorrect_conf):,})",
        )
    if not correct_conf.empty:
        ax_hist.hist(
            correct_conf, bins=bins, alpha=0.72,
            color="#42A5F5", edgecolor="white", linewidth=0.4, zorder=3,
            label=f"Correct    ({len(correct_conf):,})",
        )

    mean_c   = df["Confidence"].mean()
    median_c = df["Confidence"].median()

    ax_hist.axvline(
        mean_c,   color="#0D47A1", linestyle="--", linewidth=2.0,
        label=f"Mean:   {mean_c:.1f}%", zorder=5,
    )
    ax_hist.axvline(
        median_c, color="#4A148C", linestyle=":",  linewidth=2.0,
        label=f"Median: {median_c:.1f}%", zorder=5,
    )

    ax_hist.set_title(
        "Confidence Distribution\n(Correct vs Incorrect Predictions)",
        fontsize=TITLE_FS - 1, fontweight="bold",
    )
    ax_hist.set_xlabel("Confidence Score (%)", fontsize=LABEL_FS, labelpad=8)
    ax_hist.set_ylabel("Number of Images",     fontsize=LABEL_FS, labelpad=8)
    ax_hist.set_xlim(0, 100)
    ax_hist.legend(fontsize=8.5, loc="upper left")
    ax_hist.tick_params(labelsize=TICK_FS)
    ax_hist.grid(axis="y", alpha=0.35, zorder=0)
    ax_hist.set_axisbelow(True)

    # Annotation: std dev
    ax_hist.text(
        0.98, 0.97,
        f"std dev: {df['Confidence'].std():.1f}%",
        transform=ax_hist.transAxes, ha="right", va="top",
        fontsize=8, color="grey",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.7},
    )

    # ── Right: box plot ───────────────────────────────────────────────────────
    sns.boxplot(
        data=subset,
        x="Confidence",
        y="Actual Disaster",
        order=cat_order,
        palette={c: cmap[c] for c in cat_order},
        orient="h",
        width=0.55,
        linewidth=0.9,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
        ax=ax_box,
    )

    ax_box.axvline(
        mean_c, color="#0D47A1", linestyle="--", linewidth=1.8,
        label=f"Overall mean: {mean_c:.1f}%", zorder=4,
    )

    ax_box.set_title(
        f"Confidence by Category\n(Top {len(cat_order)} categories by image count)",
        fontsize=TITLE_FS - 1, fontweight="bold",
    )
    ax_box.set_xlabel("Confidence Score (%)", fontsize=LABEL_FS, labelpad=8)
    ax_box.set_ylabel("Disaster Category",    fontsize=LABEL_FS, labelpad=8)
    ax_box.set_xlim(0, 105)
    ax_box.tick_params(labelsize=TICK_FS)
    ax_box.grid(axis="x", alpha=0.35, zorder=0)
    ax_box.set_axisbelow(True)
    ax_box.legend(fontsize=8.5, loc="lower right")

    plt.suptitle(
        "CLIP Model Confidence Analysis",
        fontsize=TITLE_FS + 1, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, out / "confidence_distribution.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate research-quality plots from CLIP evaluation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS_CSV,
        help="Path to results.csv",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save plots",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"\n{SEP}")
    print("  VLM Disaster Analyzer — Plot Generator")
    print(SEP)
    print(f"  Reading results from : {args.results}")
    print(f"  Output directory     : {args.output}")
    print(SEP)

    df = load_results(args.results)

    n_actual    = df["Actual Disaster"].nunique()
    n_predicted = df["Predicted Disaster"].nunique()
    overall_acc = df["Correct"].mean() * 100

    print(f"\n  Loaded  {len(df):,} rows")
    print(f"  Actual categories    : {n_actual}")
    print(f"  Predicted categories : {n_predicted}")
    print(f"  Overall accuracy     : {overall_acc:.1f}%\n")

    args.output.mkdir(parents=True, exist_ok=True)
    _set_style()

    print("  Generating plots...")
    print()

    plot_confusion_matrix(df,       args.output)
    plot_category_accuracy(df,      args.output)
    plot_average_confidence(df,     args.output)
    plot_dataset_distribution(df,   args.output)
    plot_confidence_histogram(df,   args.output)

    print(f"\n{SEP}")
    print(f"  All 5 plots saved to: {args.output}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
