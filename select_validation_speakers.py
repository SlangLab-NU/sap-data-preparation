"""
Select validation speakers from a WER CSV using per-bin percentage sampling.

Each rating bin contributes a fixed fraction of its available speakers to the
validation set. This generalises across etiologies with different pool sizes
and bin distributions without needing a hard-coded target count.

Rated speakers are binned by Average_Rating.
Unrated speakers are assigned a predicted bin using Gaussian likelihood over the
WER distribution of each rated bin — identical to the approach in plot_wer_ratings.py.

Sampling rules per bin:
  - 1 speaker available  -> 0 selected (cannot spare from training)
  - 2+ speakers available -> max(1, round(n * fraction)), capped at n - 1
    (always retains at least 1 speaker in training)

Output CSV contains Speaker_ID and metadata. Compatible with sap.py --val-speakers,
which performs the actual train/val split with a no-leak assertion.

Run once per etiology. Pass all output CSVs to sap.py --val-speakers to combine them.

Usage:
    python select_validation_speakers.py \\
        --wer-csv ../VallE/egs/sap/pd_train_wer.csv \\
        --etiology "Parkinson's Disease" \\
        --output ../VallE/egs/sap/data/pd_val_speakers.csv \\
        [--val-fraction 0.15] \\
        [--min-utterances 50] \\
        [--seed 42]
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RATING_BINS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
RATING_BIN_LABELS = ["1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", "4.0-4.5", "4.5-5.0"]


def get_args():
    parser = argparse.ArgumentParser(
        description="Per-bin percentage validation speaker selection from a WER CSV"
    )
    parser.add_argument("--wer-csv", type=Path, required=True,
                        help="Path to WER CSV produced by calculate_sap_wer.py")
    parser.add_argument("--etiology", type=str, default=None,
                        help="Filter to this etiology before selection (e.g. \"Parkinson's Disease\"). "
                             "Required when the WER CSV contains multiple etiologies.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output CSV path (Speaker_ID column required by sap.py --val-speakers)")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of each bin to hold out for validation (default: 0.15)")
    parser.add_argument("--min-utterances", type=int, default=50,
                        help="Exclude speakers with fewer than this many utterances (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip saving the distribution plot")
    return parser.parse_args()


def assign_bin(rating):
    for i in range(len(RATING_BINS) - 1):
        if RATING_BINS[i] <= rating < RATING_BINS[i + 1]:
            return RATING_BIN_LABELS[i]
    return RATING_BIN_LABELS[-1]


def fit_bin_gaussians(rated):
    """Fit a Gaussian to the WER distribution of each populated rating bin."""
    params = {}
    for lbl in RATING_BIN_LABELS:
        wers = rated[rated["Bin"] == lbl]["Average_WER"].values
        if len(wers) == 0:
            continue
        mu = wers.mean()
        sigma = wers.std() if wers.std() > 0 else 1e-6
        params[lbl] = (mu, sigma)
    return params


def predict_bin(wer_val, bin_params):
    likelihoods = {lbl: stats.norm.pdf(wer_val, mu, sigma)
                   for lbl, (mu, sigma) in bin_params.items()}
    return max(likelihoods, key=likelihoods.get)


def bin_sample_size(n_available, fraction):
    """
    Number of speakers to select from a bin of size n_available.

      n_available == 1  -> 0  (cannot spare from training)
      n_available >= 2  -> max(1, round(n * fraction)), capped at n - 1
    """
    if n_available < 2:
        return 0
    n = max(1, round(n_available * fraction))
    return min(n, n_available - 1)


def plot_distribution(pool, selected, fraction, output_path):
    """
    Stacked bar chart per rating bin showing:
      - Train speakers (rated vs predicted), stacked
      - Val speakers (rated vs predicted), stacked on top
    """
    val_ids = set(selected["Speaker_ID"])
    populated_bins = [b for b in RATING_BIN_LABELS if b in pool["Effective_Bin"].values]

    train_rated   = []
    train_pred    = []
    val_rated     = []
    val_pred      = []

    for b in populated_bins:
        group = pool[pool["Effective_Bin"] == b]
        in_val = group["Speaker_ID"].isin(val_ids)
        is_rated = group["Bin"].notna()

        val_rated.append((in_val & is_rated).sum())
        val_pred.append((in_val & ~is_rated).sum())
        train_rated.append((~in_val & is_rated).sum())
        train_pred.append((~in_val & ~is_rated).sum())

    x = np.arange(len(populated_bins))
    width = 0.55

    fig, ax = plt.subplots(figsize=(11, 6))

    # Train bars (bottom half of stack)
    b_train_rated = ax.bar(x, train_rated, width, label="Train – rated",
                           color="steelblue", alpha=0.85)
    b_train_pred  = ax.bar(x, train_pred, width, bottom=train_rated,
                           label="Train – predicted bin", color="steelblue",
                           alpha=0.4, hatch="//")

    # Val bars stacked on top of train
    train_total = [r + p for r, p in zip(train_rated, train_pred)]
    b_val_rated = ax.bar(x, val_rated, width, bottom=train_total,
                         label="Val – rated", color="darkorange", alpha=0.85)
    val_bottom  = [t + r for t, r in zip(train_total, val_rated)]
    b_val_pred  = ax.bar(x, val_pred, width, bottom=val_bottom,
                         label="Val – predicted bin", color="darkorange",
                         alpha=0.4, hatch="//")

    # Annotate val count on top of each bar
    for i, (vr, vp, tt) in enumerate(zip(val_rated, val_pred, train_total)):
        n_val = vr + vp
        n_total = tt + n_val
        if n_val > 0:
            ax.text(i, n_total + 0.4, f"{n_val}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="darkorange")

    # Annotate total pool count at top
    for i, (vr, vp, tt) in enumerate(zip(val_rated, val_pred, train_total)):
        n_total = tt + vr + vp
        ax.text(i, n_total + 2.2, f"n={n_total}", ha="center", va="bottom",
                fontsize=8, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels(populated_bins, fontsize=10)
    ax.set_xlabel("Rating Bin (actual or predicted)", fontsize=12)
    ax.set_ylabel("Number of Speakers", fontsize=12)
    etiology_label = pool["Etiology"].iloc[0] if "Etiology" in pool.columns else ""
    title = f"Speaker Distribution by Rating Bin"
    if etiology_label:
        title += f" — {etiology_label}"
    title += (
        f"\nVal fraction: {fraction:.0%}  |  "
        f"Val total: {sum(val_rated) + sum(val_pred)}  |  "
        f"Train total: {sum(train_rated) + sum(train_pred)}"
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Distribution plot saved: {output_path}")


def stratified_sample(df, fraction, seed):
    rng = np.random.default_rng(seed)
    selected = []

    logger.info(f"Per-bin selection at {fraction:.0%}:")
    for b in RATING_BIN_LABELS:
        group = df[df["Effective_Bin"] == b]
        n_available = len(group)
        if n_available == 0:
            continue
        n_select = bin_sample_size(n_available, fraction)
        if n_select == 0:
            logger.info(f"  {b}: {n_available} available -> 0 selected (too few to spare)")
            continue
        chosen = group.sample(n=n_select, random_state=int(rng.integers(0, 2**31)))
        selected.append(chosen)
        logger.info(f"  {b}: {n_select} / {n_available} selected ({n_select/n_available:.0%})")

    return pd.concat(selected, ignore_index=True)


def main():
    args = get_args()

    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError(f"--val-fraction must be between 0 and 1, got {args.val_fraction}")

    df = pd.read_csv(args.wer_csv)
    df["Average_Rating"] = pd.to_numeric(df["Average_Rating"], errors="coerce")
    df["Average_WER"] = pd.to_numeric(df["Average_WER"], errors="coerce")

    if args.etiology:
        before = len(df)
        df = df[df["Etiology"] == args.etiology]
        logger.info(f"Etiology filter '{args.etiology}': {len(df)}/{before} speakers retained")
        if len(df) == 0:
            raise ValueError(
                f"No speakers found for etiology '{args.etiology}'. "
                f"Available: {sorted(pd.read_csv(args.wer_csv)['Etiology'].dropna().unique())}"
            )

    before = len(df)
    df = df.dropna(subset=["Average_WER"])
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} speakers with no valid WER")

    before = len(df)
    df = df[df["Num_Utterances"] >= args.min_utterances]
    logger.info(f"Utterance filter (>= {args.min_utterances}): {len(df)}/{before} speakers retained")

    rated = df[df["Average_Rating"].notna()].copy()
    unrated = df[df["Average_Rating"].isna()].copy()
    logger.info(f"{len(rated)} rated, {len(unrated)} unrated speakers in eligible pool")

    rated["Bin"] = rated["Average_Rating"].apply(assign_bin)
    bin_params = fit_bin_gaussians(rated)

    rated["Effective_Bin"] = rated["Bin"]
    unrated = unrated.copy()
    unrated["Bin"] = None
    unrated["Effective_Bin"] = unrated["Average_WER"].apply(lambda w: predict_bin(w, bin_params))

    pool = pd.concat([rated, unrated], ignore_index=True)

    logger.info("Eligible pool by bin:")
    for b in RATING_BIN_LABELS:
        n = (pool["Effective_Bin"] == b).sum()
        if n > 0:
            logger.info(f"  {b}: {n} speakers")

    selected = stratified_sample(pool, args.val_fraction, args.seed)

    out_cols = ["Speaker_ID", "Etiology", "Effective_Bin", "Average_Rating", "Average_WER", "Num_Utterances"]
    selected = selected[out_cols].sort_values(["Effective_Bin", "Average_WER"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)

    logger.info(f"\nSelected {len(selected)} / {len(pool)} speakers ({len(selected)/len(pool):.1%}) -> {args.output}")
    logger.info("Final bin counts:")
    for b, count in selected["Effective_Bin"].value_counts().sort_index().items():
        logger.info(f"  {b}: {count}")

    if not args.no_plot:
        plot_path = args.output.with_name(args.output.stem + "_distribution.png")
        plot_distribution(pool, selected, args.val_fraction, plot_path)


if __name__ == "__main__":
    main()
