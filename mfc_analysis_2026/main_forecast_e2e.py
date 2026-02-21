"""
End-to-end MFC voltage forecasting with Transformer, LSTM, and RNN (GRU).

All models take a sliding window of features as input and predict the
next `horizon` steps of voltage only. Trained with MSE loss + early stopping.

Usage
-----
    python main_forecast_e2e.py --model LSTM
    python main_forecast_e2e.py --model all --mode covariate
    python main_forecast_e2e.py --model Transformer --mode covariate --horizon 48
    python main_forecast_e2e.py --model RNN --hidden-dim 128 --n-epochs 200
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import torch

from dataloader import load_dataset, COVARIATE_COLS
from evaluate import evaluate_forecast, save_forecast
from models.e2e import ALL_E2E_MODELS, build_model, fit_model, predict_model, prepare_arrays

# ── Plot constants ─────────────────────────────────────────────────────────────
MODEL_COLOURS = {
    "Transformer": "#dc2626",
    "LSTM":        "#d97706",
    "RNN":         "#7c3aed",
    "Chronos-2":   "#2563eb",
    "Moirai-2":    "#16a34a",
}
CONTEXT_COLOUR  = "#9ca3af"
TRUE_COLOUR     = "#111827"
CONTEXT_HOURS   = 72
PLOT_TEST_HOURS = 168


def _plot_run(
    bundle,
    all_forecasts: dict,
    all_metrics: dict,
    histories: dict,
    run_datetime: str,
    out_dir: str = "results/",
) -> None:
    """
    3-panel figure:
      Top-left  : time-series comparison (context + true + predictions)
      Top-right : training/val loss curves for each model
      Bottom    : MSE and MAE bar charts
    """
    uid = "mfc_0"
    ctx_s  = bundle.trainval_df[bundle.trainval_df["unique_id"] == uid].tail(CONTEXT_HOURS)
    test_s = bundle.test_df[bundle.test_df["unique_id"] == uid].head(PLOT_TEST_HOURS)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"E2E MFC Voltage Forecast  |  mode={bundle.mode}  |  {run_datetime}",
        fontsize=12, fontweight="bold",
    )
    date_fmt = mdates.DateFormatter("%b %d\n%Hh")

    # ── Top-left: time series ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axvspan(ctx_s["ds"].iloc[0], ctx_s["ds"].iloc[-1],
                color=CONTEXT_COLOUR, alpha=0.07)
    ax1.axvline(test_s["ds"].iloc[0], color="#6b7280", linewidth=1.2,
                linestyle="--", label="Train/Test boundary")
    ax1.plot(ctx_s["ds"], ctx_s["y"],
             color=CONTEXT_COLOUR, linewidth=1.0,
             label=f"Context (last {CONTEXT_HOURS}h)")
    ax1.plot(test_s["ds"], test_s["y"],
             color=TRUE_COLOUR, linewidth=1.5, label="True voltage", zorder=5)

    for model_name, fc_df in all_forecasts.items():
        fc_s = (fc_df[fc_df["unique_id"] == uid]
                .merge(test_s[["ds"]], on="ds")
                .sort_values("ds"))
        colour = MODEL_COLOURS.get(model_name, "#f59e0b")
        ax1.plot(fc_s["ds"], fc_s[model_name],
                 color=colour, linewidth=1.1, linestyle="--",
                 alpha=0.85, label=model_name, zorder=4)

    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("Forecast vs True")
    ax1.xaxis.set_major_formatter(date_fmt)
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(ctx_s["ds"].iloc[0], test_s["ds"].iloc[-1])

    # ── Top-right: training loss curves ──────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    for model_name, hist in histories.items():
        colour = MODEL_COLOURS.get(model_name, "#f59e0b")
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax2.plot(epochs, hist["train_loss"],
                 color=colour, linewidth=1.0, linestyle="-",
                 label=f"{model_name} train")
        ax2.plot(epochs, hist["val_loss"],
                 color=colour, linewidth=1.0, linestyle="--",
                 label=f"{model_name} val", alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE (scaled space)")
    ax2.set_title("Training & Validation Loss")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.25)
    ax2.set_yscale("log")

    # ── Bottom: MSE and MAE bar charts ────────────────────────────────────────
    if all_metrics:
        models  = list(all_metrics.keys())
        colours = [MODEL_COLOURS.get(m, "#f59e0b") for m in models]
        x       = np.arange(len(models))
        w       = 0.5

        for subplot_idx, metric in [(3, "MSE"), (4, "MAE")]:
            ax = fig.add_subplot(2, 2, subplot_idx)
            vals = [all_metrics[m][metric] for m in models]
            bars = ax.bar(x, vals, width=w, color=colours, edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
            ax.set_title(metric)
            ax.grid(True, axis="y", alpha=0.3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = Path(out_dir) / f"forecast_e2e_{run_datetime}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2E MFC forecasting: Transformer, LSTM, RNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="LSTM",
                        choices=ALL_E2E_MODELS + ["all"])
    parser.add_argument("--mode", default="covariate",
                        choices=["univariate", "covariate"],
                        help="univariate: voltage only input. "
                             "covariate: voltage + soil features as input.")
    parser.add_argument("--horizon",    type=int, default=24,
                        help="Forecast steps (hours).")
    parser.add_argument("--input-len",  type=int, default=168,
                        help="Lookback window (hours). Default = 1 week.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers",   type=int, default=2)
    parser.add_argument("--n-heads",    type=int, default=4,
                        help="Transformer attention heads.")
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--n-epochs",   type=int, default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience",   type=int, default=10,
                        help="Early stopping patience (val MSE epochs).")
    parser.add_argument("--device",     default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--splits-dir", default="data/splits/")
    parser.add_argument("--results",    default="results/results.csv")
    parser.add_argument("--forecast-dir", default="results/forecasts/")
    parser.add_argument("--plot-dir",   default="results/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    print(f"\nRun: {run_datetime}")

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

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

    print(f"  train: {len(bundle.train_df):,}  "
          f"val: {len(bundle.val_df):,}  "
          f"test: {len(bundle.test_df):,}")

    # Feature columns: voltage first, then covariates
    feature_cols = ["y"] + bundle.covariate_cols
    n_features   = len(feature_cols)
    print(f"  Input features ({n_features}): {feature_cols}")

    # ── Scale data ────────────────────────────────────────────────────────────
    train_arr, val_arr, test_arr, scaler = prepare_arrays(
        bundle.train_df, bundle.val_df, bundle.test_df, feature_cols,
    )
    trainval_arr = np.concatenate([train_arr, val_arr], axis=0)

    models_to_run = ALL_E2E_MODELS if args.model == "all" else [args.model]

    all_metrics   = {}
    all_forecasts = {}
    histories     = {}

    for model_name in models_to_run:
        print(f"\n{'─' * 60}")
        print(f"  {model_name}  |  mode={args.mode}  |  "
              f"h={args.horizon}  |  input_len={args.input_len}  |  "
              f"hidden={args.hidden_dim}")
        print(f"{'─' * 60}")

        model = build_model(
            name=model_name,
            n_features=n_features,
            horizon=args.horizon,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        print("  Training ...")
        model, history = fit_model(
            model=model,
            train_arr=train_arr,
            val_arr=val_arr,
            input_len=args.input_len,
            horizon=args.horizon,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            device=device,
        )
        histories[model_name] = history

        print("  Predicting on test set ...")
        forecast_df = predict_model(
            model=model,
            trainval_arr=trainval_arr,
            test_arr=test_arr,
            scaler=scaler,
            input_len=args.input_len,
            horizon=args.horizon,
            test_df=bundle.test_df,
            model_name=model_name,
            device=device,
        )

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
            print(f"  {name:<14}  MSE={m['MSE']:.6f}  MAE={m['MAE']:.6f}")
        print(f"\nResults → {args.results}")

        _plot_run(
            bundle=bundle,
            all_forecasts=all_forecasts,
            all_metrics=all_metrics,
            histories=histories,
            run_datetime=run_datetime,
            out_dir=args.plot_dir,
        )


if __name__ == "__main__":
    main()
