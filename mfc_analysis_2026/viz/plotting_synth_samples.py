"""
Visualize the synthetic MFC dataset.

Generates a 4-panel figure showing:
  - MFC voltage
  - Soil moisture
  - Soil conductivity
  - Correlation heatmap between voltage and covariates

Usage
-----
    python viz/plotting_synth_samples.py
    python viz/plotting_synth_samples.py --input data/synthetic/data.csv
                                          --output viz/synth_samples.png
                                          --days 14
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

COVARIATE_COLS = ["soil_moisture", "soil_conductivity", "soil_char"]
TARGET_COL = "voltage"
TIME_COL = "timestamp"


def load_synthetic(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


def plot_synth_samples(
    input_path: str = "data/synthetic/data.csv",
    output_path: str = "viz/synth_samples.png",
    days: int = 21,
) -> None:
    """
    Plot the first `days` days of the synthetic dataset plus a correlation heatmap.
    """
    df = load_synthetic(input_path)
    n_hours = days * 24
    df_window = df.iloc[:n_hours].copy()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Synthetic MFC Dataset — first {days} days\n"
        f"(total: {len(df):,} hourly samples)",
        fontsize=13,
        fontweight="bold",
    )

    date_fmt = mdates.DateFormatter("%b %d")
    ts = df_window[TIME_COL]

    # ── Panel 1: MFC Voltage ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(4, 2, (1, 2))
    ax1.plot(ts, df_window[TARGET_COL], color="#2563eb", linewidth=0.9, label="Voltage (V)")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("MFC Output Voltage")
    ax1.xaxis.set_major_formatter(date_fmt)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Soil Moisture ────────────────────────────────────────────────
    ax2 = fig.add_subplot(4, 2, (3, 4))
    ax2.plot(ts, df_window["soil_moisture"], color="#16a34a", linewidth=0.9, label="Soil Moisture")
    ax2.set_ylabel("Moisture [0–1]")
    ax2.set_title("Soil Moisture")
    ax2.xaxis.set_major_formatter(date_fmt)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Soil Conductivity ────────────────────────────────────────────
    ax3 = fig.add_subplot(4, 2, (5, 6))
    ax3.plot(ts, df_window["soil_conductivity"], color="#d97706", linewidth=0.9,
             label="Soil Conductivity (S/m)")
    ax3.set_ylabel("Conductivity (S/m)")
    ax3.set_title("Soil Conductivity")
    ax3.xaxis.set_major_formatter(date_fmt)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Correlation heatmap (full dataset) ───────────────────────────
    ax4 = fig.add_subplot(4, 2, 7)
    cols = [TARGET_COL] + COVARIATE_COLS
    corr = df[cols].corr()
    labels = ["Voltage", "Moisture", "Conductivity", "Soil Char"]
    im = ax4.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax4.set_xticks(range(len(labels)))
    ax4.set_yticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax4.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax4.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     fontsize=7, color="black")
    plt.colorbar(im, ax=ax4, shrink=0.8)
    ax4.set_title("Pearson Correlation\n(full dataset)", fontsize=9)

    # ── Panel 5: Scatter voltage vs moisture ─────────────────────────────────
    ax5 = fig.add_subplot(4, 2, 8)
    ax5.scatter(df["soil_moisture"], df[TARGET_COL], alpha=0.05, s=3, color="#2563eb")
    ax5.set_xlabel("Soil Moisture")
    ax5.set_ylabel("Voltage (V)")
    ax5.set_title("Voltage vs Moisture\n(full dataset)", fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize synthetic MFC dataset")
    parser.add_argument("--input", default="data/synthetic/data.csv")
    parser.add_argument("--output", default="viz/synth_samples.png")
    parser.add_argument("--days", type=int, default=21,
                        help="Number of days to show in time-series panels")
    args = parser.parse_args()
    plot_synth_samples(args.input, args.output, args.days)
