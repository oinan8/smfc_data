"""
Plot forecast comparison: original vs. model predictions for one test series.

Loads the raw forecast CSVs saved by main_forecast_zero_shot.py (or e2e),
overlays all available model predictions on the true test signal,
and saves a PNG with the run datetime in its filename.

Layout
------
  Top panel    : context (grey) + true test signal (black) + model predictions
  Bottom panel : MSE / MAE bar chart from results/results.csv for this run

Usage
-----
    # After running main_forecast_zero_shot.py the script prints the exact command:
    python viz/plotting_forecast_results.py --run-datetime 2026-02-20_143022

    # Show more/less context and test signal
    python viz/plotting_forecast_results.py --run-datetime 2026-02-20_143022
        --context-hours 72 --plot-test-hours 168

    # Specify custom directories
    python viz/plotting_forecast_results.py --run-datetime 2026-02-20_143022
        --forecast-dir results/forecasts/ --splits-dir data/splits/
        --results results/results.csv --output viz/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Colours (consistent across scripts) ──────────────────────────────────────
MODEL_COLOURS = {
    "Chronos-2":   "#2563eb",
    "Moirai-2":    "#16a34a",
    "DLinear":     "#d97706",
    "Transformer": "#dc2626",
    "NBEATSx":     "#7c3aed",
    "ARIMAx":      "#0891b2",
}
CONTEXT_COLOUR = "#9ca3af"   # grey
TRUE_COLOUR    = "#111827"   # near-black


def _load_splits(splits_dir: str, dataset: str = "synthetic") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (trainval_df, test_df) in NF format from pre-split CSVs."""
    root = Path(splits_dir) / dataset
    rename = {"timestamp": "ds", "voltage": "y"}

    def _read(name):
        df = pd.read_csv(root / f"{name}.csv", parse_dates=["timestamp"])
        df = df.rename(columns=rename)
        df["unique_id"] = "mfc_0"
        return df.sort_values("ds").reset_index(drop=True)

    train = _read("train")
    val   = _read("val")
    test  = _read("test")
    return pd.concat([train, val], ignore_index=True), test


def _discover_forecasts(forecast_dir: str, run_datetime: str) -> dict[str, pd.DataFrame]:
    """
    Find all forecast CSVs from a given run and load them.

    Returns dict: model_name → DataFrame[unique_id, ds, <model_name>]
    """
    found = {}
    fc_dir = Path(forecast_dir)
    if not fc_dir.exists():
        return found

    for csv_path in sorted(fc_dir.glob(f"*_{run_datetime}.csv")):
        df = pd.read_csv(csv_path, parse_dates=["ds"])
        # Prediction column is everything except unique_id and ds
        pred_cols = [c for c in df.columns if c not in ("unique_id", "ds")]
        if not pred_cols:
            continue
        model_name = pred_cols[0]
        found[model_name] = df
    return found


def _load_run_metrics(results_path: str, run_datetime: str) -> pd.DataFrame:
    """Load metrics rows for a specific run. Returns empty DataFrame on any failure."""
    p = Path(results_path)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()
    if "run_datetime" not in df.columns:
        return pd.DataFrame()
    return df[df["run_datetime"] == run_datetime].reset_index(drop=True)


def plot_forecast_results(
    run_datetime: str,
    forecast_dir: str = "results/forecasts/",
    splits_dir: str   = "data/splits/",
    results_path: str = "results/results.csv",
    output_dir: str   = "viz/",
    context_hours: int = 72,
    plot_test_hours: int = 168,
    series_id: str = "mfc_0",
) -> None:
    """
    Generate and save the forecast comparison plot for one run.

    Parameters
    ----------
    run_datetime     : The timestamp string from a model run (e.g. "2026-02-20_143022").
    context_hours    : How many hours of trainval context to show before the test boundary.
    plot_test_hours  : How many hours of the test period to show (None = full test set).
    series_id        : Which unique_id to plot (synthetic dataset has only "mfc_0").
    """
    # ── Load ground-truth data ────────────────────────────────────────────────
    trainval_df, test_df = _load_splits(splits_dir)
    trainval_s = trainval_df[trainval_df["unique_id"] == series_id].sort_values("ds")
    test_s     = test_df[test_df["unique_id"] == series_id].sort_values("ds")

    context_s = trainval_s.tail(context_hours)
    if plot_test_hours:
        test_s = test_s.head(plot_test_hours)

    # ── Load forecasts ────────────────────────────────────────────────────────
    forecasts = _discover_forecasts(forecast_dir, run_datetime)
    if not forecasts:
        print(f"No forecast CSVs found for run_datetime='{run_datetime}' in '{forecast_dir}'")
        print("Make sure you ran main_forecast_zero_shot.py (or e2e) first.")
        return

    # Trim forecasts to the same test window
    for model_name, fc_df in forecasts.items():
        fc_s = fc_df[fc_df["unique_id"] == series_id].sort_values("ds")
        if plot_test_hours:
            fc_s = fc_s[fc_s["ds"].isin(test_s["ds"])]
        forecasts[model_name] = fc_s

    # ── Load metrics ──────────────────────────────────────────────────────────
    metrics_df = _load_run_metrics(results_path, run_datetime)

    # ── Build figure ──────────────────────────────────────────────────────────
    has_metrics = len(metrics_df) > 0
    n_rows = 2 if has_metrics else 1
    fig = plt.figure(figsize=(14, 5 * n_rows + 1))
    fig.suptitle(
        f"MFC Voltage Forecast Comparison\n"
        f"Run: {run_datetime}  |  Context: {context_hours}h  |  "
        f"Test window: {len(test_s)}h",
        fontsize=12, fontweight="bold",
    )

    date_fmt = mdates.DateFormatter("%b %d\n%Hh")

    # ── Top panel: time series ────────────────────────────────────────────────
    ax = fig.add_subplot(n_rows, 1, 1)

    # Context shading
    ax.axvspan(
        context_s["ds"].iloc[0], context_s["ds"].iloc[-1],
        color=CONTEXT_COLOUR, alpha=0.08, label="_nolegend_",
    )

    # Vertical boundary line
    boundary = test_s["ds"].iloc[0]
    ax.axvline(boundary, color="#6b7280", linewidth=1.2, linestyle="--", label="Train/Test boundary")

    # Context signal
    ax.plot(
        context_s["ds"], context_s["y"],
        color=CONTEXT_COLOUR, linewidth=1.0, label=f"Context (last {context_hours}h)",
    )

    # True test signal
    ax.plot(
        test_s["ds"], test_s["y"],
        color=TRUE_COLOUR, linewidth=1.4, label="True (test)", zorder=5,
    )

    # Model predictions
    for model_name, fc_s in forecasts.items():
        colour = MODEL_COLOURS.get(model_name, "#f59e0b")
        ax.plot(
            fc_s["ds"], fc_s[model_name],
            color=colour, linewidth=1.1, linestyle="--",
            alpha=0.85, label=model_name, zorder=4,
        )

    ax.set_ylabel("MFC Voltage (V)", fontsize=10)
    ax.set_title("Voltage: context → test predictions", fontsize=10)
    ax.xaxis.set_major_formatter(date_fmt)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(context_s["ds"].iloc[0], test_s["ds"].iloc[-1])

    # ── Bottom panel: MSE / MAE bar chart ─────────────────────────────────────
    if has_metrics:
        ax2 = fig.add_subplot(n_rows, 2, 3)
        ax3 = fig.add_subplot(n_rows, 2, 4)

        models  = metrics_df["model"].tolist()
        colours = [MODEL_COLOURS.get(m, "#f59e0b") for m in models]
        x       = np.arange(len(models))
        bar_w   = 0.5

        # MSE
        bars = ax2.bar(x, metrics_df["MSE"].values, width=bar_w, color=colours, edgecolor="white")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax2.set_ylabel("MSE")
        ax2.set_title("Mean Squared Error")
        ax2.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, metrics_df["MSE"].values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.4f}", ha="center", va="bottom", fontsize=7)

        # MAE
        bars = ax3.bar(x, metrics_df["MAE"].values, width=bar_w, color=colours, edgecolor="white")
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax3.set_ylabel("MAE")
        ax3.set_title("Mean Absolute Error")
        ax3.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, metrics_df["MAE"].values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"forecast_comparison_{run_datetime}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot original vs. predicted MFC voltage for a given run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-datetime", required=True,
        help="Datetime string of the run to plot (e.g. 2026-02-20_143022). "
             "Printed by main_forecast_zero_shot.py at the end of each run.",
    )
    parser.add_argument("--forecast-dir",  default="results/forecasts/")
    parser.add_argument("--splits-dir",    default="data/splits/")
    parser.add_argument("--results",       default="results/results.csv")
    parser.add_argument("--output",        default="viz/",
                        help="Directory to save the plot PNG.")
    parser.add_argument("--context-hours", type=int, default=72,
                        help="Hours of trainval context shown before the test boundary.")
    parser.add_argument("--plot-test-hours", type=int, default=168,
                        help="Hours of the test period to show (default 7 days). 0 = full test.")
    parser.add_argument("--series-id",     default="mfc_0",
                        help="Which unique_id to plot.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_forecast_results(
        run_datetime=args.run_datetime,
        forecast_dir=args.forecast_dir,
        splits_dir=args.splits_dir,
        results_path=args.results,
        output_dir=args.output,
        context_hours=args.context_hours,
        plot_test_hours=args.plot_test_hours if args.plot_test_hours > 0 else None,
        series_id=args.series_id,
    )
