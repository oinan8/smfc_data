"""
Visualize the real GIFT-eval solar hourly dataset.

Generates a figure showing:
  - Time-series plots for a sample of solar series
  - Distribution of series lengths
  - Overall value distribution

Usage
-----
    python viz/plotting_real_samples.py
    python viz/plotting_real_samples.py --input data/real/solar_hourly.csv
                                         --output viz/real_samples.png
                                         --n-series 6 --days 30
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

TIME_COL = "timestamp"
TARGET_COL = "target"
ID_COL = "item_id"

# Colour palette for multiple series
PALETTE = [
    "#2563eb", "#16a34a", "#d97706", "#dc2626",
    "#7c3aed", "#0891b2", "#db2777", "#65a30d",
]


def load_real(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=[TIME_COL])
    df = df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    return df


def plot_real_samples(
    input_path: str = "data/real/solar_hourly.csv",
    output_path: str = "viz/real_samples.png",
    n_series: int = 6,
    days: int = 30,
) -> None:
    """
    Plot `n_series` sample solar series (first `days` days each) plus diagnostics.
    """
    df = load_real(input_path)
    all_ids = df[ID_COL].unique()
    n_total = len(all_ids)

    # Pick evenly-spaced series to show diversity
    step = max(1, n_total // n_series)
    sample_ids = all_ids[::step][:n_series]
    n_show = len(sample_ids)

    # ── Layout: n_show time-series plots + 2 diagnostic panels ──────────────
    n_rows = n_show + 1
    fig = plt.figure(figsize=(14, 3 * n_rows))
    fig.suptitle(
        f"GIFT-Eval Solar Hourly — {n_total} series total\n"
        f"Showing {n_show} sample series ({days}-day window)",
        fontsize=13,
        fontweight="bold",
    )

    date_fmt = mdates.DateFormatter("%b %d")
    n_hours = days * 24

    for i, item_id in enumerate(sample_ids):
        series = df[df[ID_COL] == item_id].iloc[:n_hours]
        ax = fig.add_subplot(n_rows, 1, i + 1)
        color = PALETTE[i % len(PALETTE)]
        ax.plot(series[TIME_COL], series[TARGET_COL],
                color=color, linewidth=0.8, label=f"Series: {item_id}")
        ax.set_ylabel("Solar (W/m²)", fontsize=8)
        ax.xaxis.set_major_formatter(date_fmt)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Solar Irradiance Time Series (sample)")
        if i < n_show - 1:
            ax.set_xticklabels([])

    # ── Diagnostic row: series-length distribution + value distribution ──────
    ax_len = fig.add_subplot(n_rows, 2, 2 * n_rows - 1)
    lengths = df.groupby(ID_COL).size()
    ax_len.hist(lengths.values, bins=30, color="#2563eb", edgecolor="white", linewidth=0.4)
    ax_len.set_xlabel("Series length (hours)")
    ax_len.set_ylabel("Count")
    ax_len.set_title("Distribution of Series Lengths")
    ax_len.axvline(lengths.median(), color="red", linestyle="--", linewidth=1,
                   label=f"Median={lengths.median():.0f}")
    ax_len.legend(fontsize=8)
    ax_len.grid(True, alpha=0.3)

    ax_val = fig.add_subplot(n_rows, 2, 2 * n_rows)
    vals = df[TARGET_COL].dropna()
    ax_val.hist(vals, bins=50, color="#16a34a", edgecolor="white", linewidth=0.4)
    ax_val.set_xlabel("Solar irradiance (W/m²)")
    ax_val.set_ylabel("Count")
    ax_val.set_title("Value Distribution (all series)")
    ax_val.axvline(vals.mean(), color="red", linestyle="--", linewidth=1,
                   label=f"Mean={vals.mean():.1f}")
    ax_val.legend(fontsize=8)
    ax_val.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")

    # Summary
    print(f"\nDataset summary:")
    print(f"  Total series : {n_total}")
    print(f"  Total rows   : {len(df):,}")
    print(f"  Length range : {lengths.min()} – {lengths.max()} hours")
    print(f"  Value range  : {vals.min():.2f} – {vals.max():.2f}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize real GIFT-eval solar dataset")
    parser.add_argument("--input", default="data/real/solar_hourly.csv")
    parser.add_argument("--output", default="viz/real_samples.png")
    parser.add_argument("--n-series", type=int, default=6,
                        help="Number of sample series to plot")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days to show per series")
    args = parser.parse_args()
    plot_real_samples(args.input, args.output, args.n_series, args.days)
