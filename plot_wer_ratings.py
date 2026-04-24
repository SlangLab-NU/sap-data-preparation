"""
Generate visualizations comparing Average WER and Average Rating per speaker.
Reads pd_train_wer.csv (output of calculate_sap_wer.py).

Usage:
    python plot_wer_ratings.py --csv path/to/pd_train_wer.csv --output-dir path/to/output
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RATING_BINS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
RATING_BIN_LABELS = ["1.0–1.5", "1.5–2.0", "2.0–2.5", "2.5–3.0", "3.0–3.5", "3.5–4.0", "4.0–4.5", "4.5–5.0"]


def get_args():
    parser = argparse.ArgumentParser(description="Plot WER vs Rating visualizations from pd_train_wer.csv")
    parser.add_argument("--csv", type=Path, required=True, help="Path to pd_train_wer.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("wer_rating_plots"),
                        help="Directory to save plots (default: wer_rating_plots/)")
    return parser.parse_args()


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["Average_Rating"] = pd.to_numeric(df["Average_Rating"], errors="coerce")
    df["Average_WER"] = pd.to_numeric(df["Average_WER"], errors="coerce")
    rated = df[df["Average_Rating"].notna()].copy()
    unrated = df[df["Average_Rating"].isna()].copy()
    logger.info(f"Loaded {len(df)} speakers: {len(rated)} rated, {len(unrated)} unrated")
    return df, rated, unrated


def assign_bin(rating):
    for i in range(len(RATING_BINS) - 1):
        if RATING_BINS[i] <= rating < RATING_BINS[i + 1]:
            return RATING_BIN_LABELS[i]
    return RATING_BIN_LABELS[-1]


# ── Plot 1: Scatter with correlation line ─────────────────────────────────────

def plot_scatter(rated, unrated, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Unrated speakers — shown as a rug/strip at y=0 with a distinct marker
    ax.scatter(unrated["Average_WER"], np.zeros(len(unrated)),
               color="gray", alpha=0.4, marker="|", s=80, linewidths=1.2,
               label=f"Unrated (n={len(unrated)})", zorder=2)

    # Rated speakers
    ax.scatter(rated["Average_WER"], rated["Average_Rating"],
               color="steelblue", alpha=0.65, s=50, edgecolors="white", linewidths=0.4,
               label=f"Rated (n={len(rated)})", zorder=3)

    # Correlation line (rated only)
    slope, intercept, r, p, _ = stats.linregress(rated["Average_WER"], rated["Average_Rating"])
    x_line = np.linspace(rated["Average_WER"].min(), rated["Average_WER"].max(), 200)
    ax.plot(x_line, slope * x_line + intercept,
            color="crimson", linewidth=2, label=f"Fit: r={r:.2f}, p={p:.3f}", zorder=4)

    ax.set_xlabel("Average WER", fontsize=12)
    ax.set_ylabel("Average Rating", fontsize=12)
    ax.set_title("Average WER vs. Average Rating per Speaker\n(TRAIN set)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(-0.3, rated["Average_Rating"].max() * 1.1)

    plt.tight_layout()
    out = output_dir / "scatter_wer_vs_rating.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


# ── Plot 2: Binned grouped bar chart ──────────────────────────────────────────

# Consistent bin colours used in both subplots
BIN_COLORS = plt.cm.tab10.colors


def plot_binned_bars(rated, unrated, output_dir):
    rated = rated.copy()
    rated["Bin"] = rated["Average_Rating"].apply(assign_bin)

    bin_groups = rated.groupby("Bin", sort=False)
    bin_order = [b for b in RATING_BIN_LABELS if b in rated["Bin"].values]

    avg_wer_bins = [bin_groups.get_group(b)["Average_WER"].mean() for b in bin_order]
    avg_rating_bins = [bin_groups.get_group(b)["Average_Rating"].mean() for b in bin_order]
    counts_bins = [len(bin_groups.get_group(b)) for b in bin_order]

    all_labels = bin_order + ["Unrated"]
    avg_wer_all = avg_wer_bins + [unrated["Average_WER"].mean()]
    avg_rating_all = avg_rating_bins + [None]
    counts_all = counts_bins + [len(unrated)]

    # Assign a colour per bin (reused in lower subplot)
    bin_color_map = {lbl: BIN_COLORS[i % len(BIN_COLORS)] for i, lbl in enumerate(bin_order)}
    bin_color_map["Unrated"] = "lightgray"

    x = np.arange(len(all_labels))
    width = 0.35

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 14),
        gridspec_kw={"height_ratios": [1, 1.6]}
    )

    # ── Top: binned summary ──
    bars_wer = ax_top.bar(x - width / 2, avg_wer_all, width, label="Avg WER",
                          color="steelblue", alpha=0.85)

    rating_vals = [v if v is not None else 0 for v in avg_rating_all]
    bars_rating = ax_top.bar(x + width / 2, rating_vals, width, label="Avg Rating",
                             color="darkorange", alpha=0.85)
    bars_rating[-1].set_color("lightgray")
    bars_rating[-1].set_edgecolor("gray")
    bars_rating[-1].set_linestyle("--")

    for bar, val in zip(bars_wer, avg_wer_all):
        ax_top.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_rating, avg_rating_all):
        if val is not None:
            ax_top.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"{lbl}\n(n={c})" for lbl, c in zip(all_labels, counts_all)], fontsize=10)
    ax_top.set_ylabel("Value", fontsize=12)
    ax_top.set_title("Average WER and Rating by Rating Bin\n(TRAIN set)", fontsize=14, fontweight="bold")
    ax_top.legend(fontsize=10)
    ax_top.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Bottom: individual speaker WER, coloured by bin ──
    # Build ordered list: rated speakers sorted by bin then WER, then unrated sorted by WER
    speaker_rows = []
    for lbl in bin_order:
        group = bin_groups.get_group(lbl).sort_values("Average_WER")
        for _, row in group.iterrows():
            speaker_rows.append((row["Average_WER"], bin_color_map[lbl], lbl))
    for _, row in unrated.sort_values("Average_WER").iterrows():
        speaker_rows.append((row["Average_WER"], bin_color_map["Unrated"], "Unrated"))

    wers = [r[0] for r in speaker_rows]
    colors = [r[1] for r in speaker_rows]
    bins_seq = [r[2] for r in speaker_rows]

    ax_bot.bar(range(len(wers)), wers, color=colors, alpha=0.8, width=1.0)

    y_top = max(wers) * 1.05

    # Bin boundary lines and labels
    current_bin = bins_seq[0]
    start_idx = 0
    for i, b in enumerate(bins_seq):
        if b != current_bin or i == len(bins_seq) - 1:
            end_idx = i if b != current_bin else i + 1
            mid = (start_idx + end_idx - 1) / 2
            ax_bot.text(mid, y_top, current_bin, ha="center", va="bottom", fontsize=7.5,
                        color=bin_color_map[current_bin], fontweight="bold")
            if b != current_bin:
                ax_bot.axvline(i - 0.5, color="black", linewidth=0.6, alpha=0.4, linestyle="--")
            current_bin = b
            start_idx = i

    ax_bot.set_xlim(-0.5, len(wers) - 0.5)
    ax_bot.set_xticks([])
    ax_bot.set_xlabel("Individual Speakers (grouped by rating bin, sorted by WER within bin)", fontsize=11)
    ax_bot.set_ylabel("Average WER", fontsize=12)
    ax_bot.grid(axis="y", alpha=0.3, linestyle="--")

    # Legend for bin colours
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=bin_color_map[lbl], alpha=0.8, label=lbl)
                      for lbl in all_labels]
    ax_bot.legend(handles=legend_handles, fontsize=8, ncol=5, loc="upper left")

    plt.tight_layout()
    out = output_dir / "binned_bar_wer_rating.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


# ── Plot 3: Sorted dual-axis bar chart ────────────────────────────────────────

def plot_sorted_dual_axis(df, output_dir):
    rated = df[df["Average_Rating"].notna()].sort_values("Average_Rating")
    unrated = df[df["Average_Rating"].isna()].sort_values("Average_WER")
    ordered = pd.concat([rated, unrated], ignore_index=True)

    x = np.arange(len(ordered))
    n_rated = len(rated)

    fig, ax1 = plt.subplots(figsize=(20, 6))

    # WER bars on primary axis
    colors = ["steelblue"] * n_rated + ["gray"] * len(unrated)
    ax1.bar(x, ordered["Average_WER"], color=colors, alpha=0.7, width=1.0, label="Avg WER")
    ax1.set_ylabel("Average WER", fontsize=12, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Rating line on secondary axis (rated speakers only)
    ax2 = ax1.twinx()
    ax2.plot(np.arange(n_rated), rated["Average_Rating"].values,
             color="crimson", linewidth=1.5, alpha=0.85, label="Avg Rating")
    ax2.set_ylabel("Average Rating", fontsize=12, color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    # Divider between rated and unrated
    ax1.axvline(n_rated - 0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.text(n_rated - 0.5 + 1, ax1.get_ylim()[1] * 0.95, "Unrated →",
             fontsize=9, color="gray", va="top")

    ax1.set_xlabel("Speakers (sorted by rating, then unrated)", fontsize=11)
    ax1.set_xticks([])
    ax1.set_title("Per-Speaker WER and Rating (sorted by rating)\n(TRAIN set)",
                  fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")
    ax1.grid(axis="y", alpha=0.2, linestyle="--")

    plt.tight_layout()
    out = output_dir / "sorted_dual_axis_wer_rating.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


# ── Plot 4: Box plot by rating bin ────────────────────────────────────────────

def plot_boxplot(rated, unrated, output_dir):
    rated = rated.copy()
    rated["Bin"] = rated["Average_Rating"].apply(assign_bin)

    bin_order = [b for b in RATING_BIN_LABELS if b in rated["Bin"].values]
    wer_by_bin = [rated[rated["Bin"] == b]["Average_WER"].values for b in bin_order]
    counts_bins = [len(v) for v in wer_by_bin]

    wer_unrated = unrated["Average_WER"].values
    all_labels = bin_order + ["Unrated"]
    all_data = wer_by_bin + [wer_unrated]
    all_counts = counts_bins + [len(wer_unrated)]

    fig, ax = plt.subplots(figsize=(14, 7))

    bp = ax.boxplot(all_data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))

    colors = ["steelblue"] * len(bin_order) + ["lightgray"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(all_labels) + 1))
    ax.set_xticklabels([f"{lbl}\n(n={c})" for lbl, c in zip(all_labels, all_counts)], fontsize=10)
    ax.set_ylabel("Average WER", fontsize=12)
    ax.set_title("WER Distribution by Rating Bin\n(TRAIN set)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    out = output_dir / "boxplot_wer_by_rating_bin.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, rated, unrated = load_data(args.csv)

    plot_scatter(rated, unrated, args.output_dir)
    plot_binned_bars(rated, unrated, args.output_dir)
    plot_sorted_dual_axis(df, args.output_dir)
    plot_boxplot(rated, unrated, args.output_dir)

    logger.info(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
