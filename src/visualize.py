"""
visualize.py — Plot inference results from the disaster classification pipeline.

Reads:   outputs/results.csv
Saves:   outputs/plots/confusion_matrix.png
         outputs/plots/category_accuracy.png

Only mapped categories (mapped_or_unmapped == "mapped") are included so that
classes with no CLIP analogue do not distort the accuracy or confusion matrix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent.parent
OUTPUT_DIR  = BASE_DIR / "outputs" / "plots"
RESULTS_CSV = BASE_DIR / "outputs" / "results.csv"

# ---------------------------------------------------------------------------
# CLIP verbose label → short display name
# Keys must match the exact strings produced by clip_model.py.
# ---------------------------------------------------------------------------

CLIP_LABEL_SHORT: dict[str, str] = {
    "a photo of urban flooding with submerged roads":       "flooding",
    "a photo of wildfire with flames and smoke":            "wildfire",
    "a photo of collapsed buildings after earthquake":      "earthquake",
    "a photo of landslide on roads or mountains":           "landslide",
    "a photo of cyclone destruction with damaged houses":   "cyclone",
    "a photo of tsunami waves hitting coastal areas":       "tsunami",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mapped_results(csv_path: Path) -> pd.DataFrame:
    """
    Read *csv_path*, keep only rows where mapped_or_unmapped == 'mapped',
    and add a *predicted_short* column with human-readable label names.

    Raises FileNotFoundError if the CSV does not exist, and ValueError if
    no mapped records are present (e.g. pipeline was run with 0 images).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {csv_path}\n"
            "Run  python src/pipeline.py  first to generate inference results."
        )

    df = pd.read_csv(csv_path)

    mapped = df[df["mapped_or_unmapped"] == "mapped"].copy()
    if mapped.empty:
        raise ValueError(
            "No mapped records found in the CSV. "
            "Check that CLIP-compatible images are present in the dataset."
        )

    # Shorten verbose CLIP labels; fall back to the raw string if unknown.
    mapped["predicted_short"] = mapped["predicted_label"].map(
        lambda x: CLIP_LABEL_SHORT.get(x, x)
    )
    return mapped


# ---------------------------------------------------------------------------
# Plot 1 — Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(df: pd.DataFrame, output_path: Path) -> None:
    """
    Build a confusion matrix from *actual_label* vs *predicted_short* and
    save a seaborn heatmap to *output_path*.
    """
    # pd.crosstab produces a label × label frequency table; fill_value=0
    # ensures every (actual, predicted) pair appears even if count is zero.
    cm = pd.crosstab(
        df["actual_label"],
        df["predicted_short"],
        rownames=["Actual"],
        colnames=["Predicted"],
    ).fillna(0).astype(int)

    # Align axes so every label appears on both rows and columns.
    all_labels = sorted(set(cm.index) | set(cm.columns))
    cm = cm.reindex(index=all_labels, columns=all_labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Image count"},
        ax=ax,
    )

    ax.set_title(
        "Confusion Matrix — CLIP Disaster Classification",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Label",    fontsize=12, labelpad=10)
    ax.tick_params(axis="x", rotation=40, labelsize=10)
    ax.tick_params(axis="y", rotation=0,  labelsize=10)

    plt.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 2 — Per-category accuracy bar chart
# ---------------------------------------------------------------------------

def plot_category_accuracy(df: pd.DataFrame, output_path: Path) -> None:
    """
    Compute per-category accuracy from *correct_or_wrong* and save a bar
    chart to *output_path*.  A dashed red line marks the mean accuracy.
    """
    stats = (
        df.groupby("actual_label")["correct_or_wrong"]
        .apply(lambda col: (col == "correct").sum() / len(col) * 100)
        .reset_index(name="accuracy")
        .sort_values("accuracy", ascending=False)
    )

    mean_acc = stats["accuracy"].mean()

    fig, ax = plt.subplots(figsize=(11, 6))

    # Use a sequential palette so bars are visually ranked.
    palette = sns.color_palette("Blues_d", len(stats))
    bars = ax.bar(
        stats["actual_label"],
        stats["accuracy"],
        color=palette,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # Annotate each bar with its exact percentage.
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Mean accuracy reference line.
    ax.axhline(
        y=mean_acc,
        color="crimson",
        linestyle="--",
        linewidth=1.4,
        zorder=4,
        label=f"Mean accuracy: {mean_acc:.1f}%",
    )

    ax.set_title(
        "Per-Category Classification Accuracy (CLIP)",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Disaster Category", fontsize=12, labelpad=10)
    ax.set_ylabel("Accuracy (%)",      fontsize=12, labelpad=10)
    ax.set_ylim(0, 115)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    plt.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    """Resolve to an absolute path, create parent dirs, save, verify, and print."""
    abs_path = path.resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(abs_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if not abs_path.exists():
        raise RuntimeError(f"Save appeared to succeed but file not found: {abs_path}")
    print(f"  Saved → {abs_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(csv_path: Path = RESULTS_CSV, plots_dir: Path = OUTPUT_DIR) -> None:
    """Load results and generate all plots."""
    abs_csv  = csv_path.resolve()
    abs_dir  = plots_dir.resolve()

    print(f"\nReading results from: {abs_csv}")
    df = load_mapped_results(abs_csv)
    print(f"Mapped records loaded: {len(df)}")

    print(f"Output directory    : {abs_dir}")
    print("\nGenerating plots...")
    plot_confusion_matrix(df,  abs_dir / "confusion_matrix.png")
    plot_category_accuracy(df, abs_dir / "category_accuracy.png")
    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
