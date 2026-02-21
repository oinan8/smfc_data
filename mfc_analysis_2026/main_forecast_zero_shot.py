"""
Zero-shot MFC voltage forecasting.

Runs Chronos-2 and/or Moirai-2 on the synthetic dataset test split
without any training. Logs MSE/MAE to results/results.csv and saves
a comparison plot automatically at the end of each run.

Usage
-----
    python main_forecast_zero_shot.py --model Chronos-2
    python main_forecast_zero_shot.py --model all --mode univariate
    python main_forecast_zero_shot.py --model Chronos-2 --mode covariate --horizon 48
    python main_forecast_zero_shot.py --model Chronos-2 --context-length 336 --device cpu
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from dataloader import load_dataset
from evaluate import evaluate_forecast, save_forecast
from models.zero_shot import ALL_ZERO_SHOT_MODELS, run_zero_shot_model

# ── Plot constants ─────────────────────────────────────────────────────────────
MODEL_COLOURS = {
    "Chronos-2":   "#2563eb",
    "Moirai-2":    "#16a34a",
    "DLinear":     "#d97706",
    "Transformer": "#dc2626",
    "NBEATSx":     "#7c3aed",
    "ARIMAx":      "#0891b2",
}
CONTEXT_COLOUR = "#9ca3af"
TRUE_COLOUR    = "#111827"
CONTEXT_HOURS  = 72    # hours of trainval context shown before test boundary
PLOT_TEST_HOURS = 168  # first 7 days of test period shown in the plot


def _plot_run(
    bundle,
    all_forecasts: dict,   # model_name → forecast DataFrame
    all_metrics: dict,     # model_name → {"MSE": float, "MAE": float}
    run_datetime: str,
    out_dir: str = "results/",
) -> None:
    """
    Generate and save the forecast comparison plot for one run.

    Layout
    ------
    Top    : context (grey) + true test signal (black) + model predictions
    Bottom : MSE and MAE bar charts side by side
    """
    uid = "mfc_0"

    # Ground-truth slices
    ctx_s  = bundle.trainval_df[bundle.trainval_df["unique_id"] == uid].tail(CONTEXT_HOURS)
    test_s = bundle.test_df[bundle.test_df["unique_id"] == uid].head(PLOT_TEST_HOURS)

    has_metrics = bool(all_metrics)
    n_rows = 2 if has_metrics else 1
    fig = plt.figure(figsize=(14, 4 * n_rows + 1))
    fig.suptitle(
        f"Zero-Shot MFC Voltage Forecast  |  mode={bundle.mode}  |  {run_datetime}",
        fontsize=12, fontweight="bold",
    )

    date_fmt = mdates.DateFormatter("%b %d\n%Hh")

    # ── Time-series panel ─────────────────────────────────────────────────────
    ax = fig.add_subplot(n_rows, 1, 1)

    ax.axvspan(ctx_s["ds"].iloc[0], ctx_s["ds"].iloc[-1],
               color=CONTEXT_COLOUR, alpha=0.07)
    ax.axvline(test_s["ds"].iloc[0], color="#6b7280", linewidth=1.2,
               linestyle="--", label="Train/Test boundary")

    ax.plot(ctx_s["ds"], ctx_s["y"],
            color=CONTEXT_COLOUR, linewidth=1.0,
            label=f"Context (last {CONTEXT_HOURS}h)")
    ax.plot(test_s["ds"], test_s["y"],
            color=TRUE_COLOUR, linewidth=1.5, label="True voltage", zorder=5)

    for model_name, fc_df in all_forecasts.items():
        fc_s = (
            fc_df[fc_df["unique_id"] == uid]
            .merge(test_s[["ds"]], on="ds")   # align to plotted window
            .sort_values("ds")
        )
        colour = MODEL_COLOURS.get(model_name, "#f59e0b")
        ax.plot(fc_s["ds"], fc_s[model_name],
                color=colour, linewidth=1.1, linestyle="--",
                alpha=0.85, label=model_name, zorder=4)

    ax.set_ylabel("Voltage (V)")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(ctx_s["ds"].iloc[0], test_s["ds"].iloc[-1])

    # ── Metric bar charts ─────────────────────────────────────────────────────
    if has_metrics:
        models  = list(all_metrics.keys())
        colours = [MODEL_COLOURS.get(m, "#f59e0b") for m in models]
        x       = np.arange(len(models))
        w       = 0.5

        for col_idx, (metric, ax_sub) in enumerate(
            [("MSE", fig.add_subplot(n_rows, 2, 3)),
             ("MAE", fig.add_subplot(n_rows, 2, 4))]
        ):
            vals = [all_metrics[m][metric] for m in models]
            bars = ax_sub.bar(x, vals, width=w, color=colours, edgecolor="white")
            ax_sub.set_xticks(x)
            ax_sub.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
            ax_sub.set_title(metric)
            ax_sub.grid(True, axis="y", alpha=0.3)
            for bar, val in zip(bars, vals):
                ax_sub.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7,
                )

    plt.tight_layout()
    out = Path(out_dir) / f"forecast_zero_shot_{run_datetime}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot MFC forecasting with Chronos-2 and Moirai-2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="Chronos-2",
        choices=ALL_ZERO_SHOT_MODELS + ["all"],
        help="Model to run, or 'all' to run every zero-shot model.",
    )
    parser.add_argument(
        "--mode", default="univariate",
        choices=["univariate", "covariate"],
    )
    parser.add_argument(
        "--horizon", type=int, default=24,
        help="Forecast step size (hours).",
    )
    parser.add_argument(
        "--context-length", type=int, default=512,
        help="Max context rows fed to the model per window.",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--splits-dir", default="data/splits/",
    )
    parser.add_argument(
        "--results", default="results/results.csv",
    )
    parser.add_argument(
        "--forecast-dir", default="results/forecasts/",
        help="Directory to save raw prediction CSVs.",
    )
    parser.add_argument(
        "--plot-dir", default="results/",
        help="Directory to save the comparison plot PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    print(f"\nRun: {run_datetime}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading synthetic/{args.mode} data ...")
    try:
        bundle = load_dataset(
            dataset="synthetic",
            mode=args.mode,
            splits_dir=args.splits_dir,
            horizon=args.horizon,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"  train: {len(bundle.train_df):,}  "
        f"val: {len(bundle.val_df):,}  "
        f"test: {len(bundle.test_df):,}"
    )

    models = ALL_ZERO_SHOT_MODELS if args.model == "all" else [args.model]

    # ── Run each model ────────────────────────────────────────────────────────
    all_metrics   = {}
    all_forecasts = {}

    for model_name in models:
        print(f"\n{'─' * 60}")
        print(f"  {model_name}  |  mode={args.mode}  |  h={args.horizon}  |  ctx={args.context_length}")
        print(f"{'─' * 60}")

        try:
            forecast_df = run_zero_shot_model(
                model_name=model_name,
                bundle=bundle,
                horizon=args.horizon,
                context_length=args.context_length,
                device=args.device,
            )
        except Exception as e:
            print(f"  [ERROR] {model_name} failed: {e}", file=sys.stderr)
            continue

        save_forecast(
            forecast_df=forecast_df,
            model_name=model_name,
            dataset_name="synthetic",
            mode=args.mode,
            horizon=args.horizon,
            run_datetime=run_datetime,
            forecast_dir=args.forecast_dir,
        )

        metrics = evaluate_forecast(
            test_df=bundle.test_df,
            forecast_df=forecast_df,
            model_name=model_name,
            dataset_name="synthetic",
            mode=args.mode,
            horizon=args.horizon,
            run_datetime=run_datetime,
            results_path=args.results,
        )
        all_metrics[model_name]   = metrics
        all_forecasts[model_name] = forecast_df

    # ── Summary + plot ────────────────────────────────────────────────────────
    if all_metrics:
        print(f"\n{'═' * 60}")
        print("  RESULTS SUMMARY")
        print(f"{'═' * 60}")
        for name, m in all_metrics.items():
            print(f"  {name:<16}  MSE={m['MSE']:.6f}  MAE={m['MAE']:.6f}")
        print(f"\nResults → {args.results}")

        _plot_run(
            bundle=bundle,
            all_forecasts=all_forecasts,
            all_metrics=all_metrics,
            run_datetime=run_datetime,
            out_dir=args.plot_dir,
        )


if __name__ == "__main__":
    main()
