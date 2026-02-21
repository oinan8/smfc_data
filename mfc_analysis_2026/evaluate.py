"""
Evaluation utilities: MSE/MAE computation, results logging, and forecast saving.

All model wrappers return forecast DataFrames in NeuralForecast format:
    unique_id | ds | <model_name>

evaluate_forecast() merges with the test targets, computes metrics,
prints a summary, and appends a row to the results CSV (which includes
a run_datetime column so every row is traceable to a specific run).

save_forecast() writes the raw prediction DataFrame to a dated CSV so
the comparison plot script can load it without re-running models.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_COLS = ["run_datetime", "model", "dataset", "mode", "horizon", "MSE", "MAE"]
DEFAULT_RESULTS_PATH = "results/results.csv"
DEFAULT_FORECAST_DIR = "results/forecasts/"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MSE and MAE.

    Parameters
    ----------
    y_true, y_pred : 1-D arrays of the same length.
        NaN values are ignored (dropped pairwise).

    Returns
    -------
    {"MSE": float, "MAE": float}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return {"MSE": float("nan"), "MAE": float("nan")}

    y_true, y_pred = y_true[mask], y_pred[mask]
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"MSE": mse, "MAE": mae}


def evaluate_forecast(
    test_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    mode: str,
    horizon: int,
    run_datetime: str,
    results_path: str = DEFAULT_RESULTS_PATH,
) -> dict:
    """
    Merge test targets with model predictions, compute metrics, log results.

    Parameters
    ----------
    test_df : DataFrame with [unique_id, ds, y, ...]
    forecast_df : DataFrame with [unique_id, ds, <model_name>]
    model_name : column name in forecast_df that holds predictions.
    run_datetime : ISO datetime string stamped on every results row.
    results_path : CSV file to append results to (append-only log).

    Returns
    -------
    {"MSE": float, "MAE": float}
    """
    merged = test_df[["unique_id", "ds", "y"]].merge(
        forecast_df[["unique_id", "ds", model_name]],
        on=["unique_id", "ds"],
        how="left",
    )

    metrics = compute_metrics(merged["y"].values, merged[model_name].values)

    # Console output
    n_missing = merged[model_name].isna().sum()
    suffix = f"  [{n_missing} NaN predictions]" if n_missing else ""
    print(
        f"[{model_name:<14} | {dataset_name:<10} | {mode:<10} | h={horizon:>3}]"
        f"  MSE={metrics['MSE']:.6f}  MAE={metrics['MAE']:.6f}{suffix}"
    )

    # Append to CSV
    out = Path(results_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([{
        "run_datetime": run_datetime,
        "model":        model_name,
        "dataset":      dataset_name,
        "mode":         mode,
        "horizon":      horizon,
        "MSE":          metrics["MSE"],
        "MAE":          metrics["MAE"],
    }])

    if out.exists():
        row.to_csv(out, mode="a", header=False, index=False)
    else:
        row.to_csv(out, mode="w", header=True, index=False)

    return metrics


def save_forecast(
    forecast_df: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    mode: str,
    horizon: int,
    run_datetime: str,
    forecast_dir: str = DEFAULT_FORECAST_DIR,
) -> str:
    """
    Save raw model predictions to a dated CSV for later plotting.

    File name format:
        {forecast_dir}/{model_name}_{dataset}_{mode}_h{horizon}_{run_datetime}.csv

    Returns the path of the saved file.
    """
    safe_model = model_name.replace("-", "").replace(" ", "_")
    filename = f"{safe_model}_{dataset_name}_{mode}_h{horizon}_{run_datetime}.csv"
    out = Path(forecast_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(out, index=False)
    print(f"  Forecast saved → {out}")
    return str(out)


def load_results(results_path: str = DEFAULT_RESULTS_PATH) -> pd.DataFrame:
    """Load the results CSV; return empty DataFrame if it does not exist."""
    p = Path(results_path)
    if not p.exists():
        return pd.DataFrame(columns=RESULTS_COLS)
    return pd.read_csv(p)
