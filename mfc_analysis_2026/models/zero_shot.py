"""
Zero-shot foundation model wrappers: Chronos-2 and Moirai-2.

Both models support univariate and covariate modes via rolling horizon windows.

Covariate strategies
--------------------
Chronos-2 (inherently univariate):
    Linear-residual approach — fit Ridge(covariates → voltage) on the context,
    forecast the voltage residuals with Chronos, then add back the linear
    covariate prediction for the forecast window. This lets Chronos use soil
    covariate information without changing its inference API.

Moirai (moirai-1.1-R-small):
    Pass soil covariates as `past_feat_dynamic_real` in the GluonTS PandasDataset.
    Moirai uses them internally as past-only conditioning features.
    Note: Moirai-2.0 dropped covariate support; moirai-1.1-R-small is used
    for covariate mode.

Rolling evaluation
------------------
At each window the context is extended with ACTUAL test values (not predictions),
giving a fair multi-step evaluation consistent with the e2e model protocol.

Public API
----------
    run_zero_shot_model(model_name, bundle, horizon, context_length, device)
        → forecast_df: [unique_id, ds, <model_name>]
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

COVARIATE_COLS = ["soil_moisture", "soil_conductivity", "soil_char"]

# ── Model IDs ─────────────────────────────────────────────────────────────────
CHRONOS_MODEL_ID  = "amazon/chronos-bolt-small"
# moirai-2.0-R-small is not yet publicly available; moirai-1.1-R-small supports
# both univariate and covariate (past_feat_dynamic_real_dim) modes.
MOIRAI_MODEL_ID   = "Salesforce/moirai-1.1-R-small"

# Module-level singletons — avoid reloading weights between calls
_chronos_pipeline  = None
_moirai_predictor  = None
_moirai_pred_len   = None
_moirai_n_cov      = None   # number of covariate dims the predictor was built for


# ─────────────────────────────────────────────────────────────────────────────
# Shared rolling-window loop
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_forecast(
    predict_window_fn,
    trainval_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    context_length: int,
    pred_col: str,
    covariate_cols: list,
) -> pd.DataFrame:
    """
    Roll a forecast horizon across the full test period.

    predict_window_fn(context_df, future_cov_df, h) -> np.ndarray[h]
        context_df     — last context_length rows from accumulated data,
                         includes covariate columns when covariate_cols is set.
        future_cov_df  — covariate values for the next h test steps
                         (None in univariate mode).
        h              — window size (≤ horizon for the last partial window).

    Context is extended with actual test values after each window, not
    with model predictions, to ensure a fair evaluation protocol.
    """
    uid_col, ds_col, y_col = "unique_id", "ds", "y"
    base_cols = [uid_col, ds_col, y_col]
    all_cols  = base_cols + [c for c in covariate_cols if c not in base_cols]

    records = []

    for uid, test_series in test_df.groupby(uid_col, sort=True):
        test_series = test_series.sort_values(ds_col).reset_index(drop=True)
        n_test = len(test_series)

        # Seed context with trainval for this series
        ctx = (
            trainval_df[trainval_df[uid_col] == uid]
            .sort_values(ds_col)
            .reset_index(drop=True)
            [[c for c in all_cols if c in trainval_df.columns]]
            .copy()
        )

        preds = []
        for start in range(0, n_test, horizon):
            end = min(start + horizon, n_test)
            h   = end - start

            ctx_window = ctx.tail(context_length) if len(ctx) > context_length else ctx

            # Future covariate values for this window (known from test set)
            future_cov = None
            if covariate_cols:
                cov_cols_present = [c for c in covariate_cols if c in test_series.columns]
                future_cov = test_series.iloc[start:end][[uid_col, ds_col] + cov_cols_present].copy()

            try:
                window_preds = predict_window_fn(ctx_window, future_cov, h)
                window_preds = np.asarray(window_preds, dtype=float)
            except Exception as exc:
                warnings.warn(
                    f"Prediction failed for uid={uid}, window=[{start},{end}): {exc}",
                    stacklevel=3,
                )
                window_preds = np.full(h, np.nan)

            preds.append(window_preds)

            # Extend context with ACTUAL test values (not predictions)
            actual = test_series.iloc[start:end][[c for c in all_cols if c in test_series.columns]]
            ctx = pd.concat([ctx, actual], ignore_index=True)

        preds_flat = np.concatenate(preds)
        for i, row in test_series.iterrows():
            records.append({uid_col: uid, ds_col: row[ds_col], pred_col: preds_flat[i]})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Chronos-2 (Chronos-Bolt)
# ─────────────────────────────────────────────────────────────────────────────

def _load_chronos(model_id: str, device: str):
    global _chronos_pipeline
    if _chronos_pipeline is not None:
        return _chronos_pipeline

    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError as e:
        raise ImportError(
            "chronos-forecasting is not installed. Run: pip install chronos-forecasting"
        ) from e

    device_map = (
        device if device != "auto"
        else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    )
    print(f"  Loading Chronos-Bolt from '{model_id}' on {device_map} ...")
    _chronos_pipeline = BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=__import__("torch").float32,
    )
    return _chronos_pipeline


def _chronos_raw_forecast(pipeline, context_df: pd.DataFrame, h: int) -> np.ndarray:
    """
    Run one Chronos window: NF-format context → median forecast of length h.
    Strips all covariate columns before calling the model.
    """
    ctx = (
        context_df[["unique_id", "ds", "y"]]
        .rename(columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"})
    )
    fc_df = pipeline.predict_df(ctx, prediction_length=h, quantile_levels=[0.5])

    # Column may be float 0.5 or string "0.5"
    for key in (0.5, "0.5"):
        if key in fc_df.columns:
            return fc_df[key].values[:h]
    if "mean" in fc_df.columns:
        return fc_df["mean"].values[:h]
    raise ValueError(f"Unexpected Chronos output columns: {fc_df.columns.tolist()}")


def _chronos_predict_univariate(pipeline, ctx_df, future_cov_df, h):
    """Univariate window: ignore covariates, forecast voltage directly."""
    return _chronos_raw_forecast(pipeline, ctx_df, h)


def _chronos_predict_covariate(pipeline, ctx_df, future_cov_df, h):
    """
    Covariate-informed window via linear-residual approach:

    1. Fit Ridge regression: covariates → voltage, on the context window.
    2. Compute residuals = voltage − linear_fit(covariates).
    3. Forecast residuals with Chronos (univariate).
    4. Reconstruct: forecast = residual_forecast + linear_fit(future_covariates).
       Falls back to mean linear prediction if future covariates are missing.
    """
    from sklearn.linear_model import Ridge

    cov_present = [c for c in COVARIATE_COLS if c in ctx_df.columns]
    if not cov_present:
        return _chronos_raw_forecast(pipeline, ctx_df, h)

    X_ctx = ctx_df[cov_present].values
    y_ctx = ctx_df["y"].values

    lr = Ridge(alpha=1.0)
    lr.fit(X_ctx, y_ctx)

    residuals = y_ctx - lr.predict(X_ctx)

    # Build residual context for Chronos
    res_ctx = ctx_df[["unique_id", "ds"]].copy()
    res_ctx["y"] = residuals

    res_forecast = _chronos_raw_forecast(pipeline, res_ctx, h)

    # Linear prediction for the test window
    if future_cov_df is not None and len(future_cov_df) >= h:
        cov_future = [c for c in cov_present if c in future_cov_df.columns]
        if cov_future:
            X_future = future_cov_df[cov_future].values[:h]
            # Fill any missing columns with context means
            if len(cov_future) < len(cov_present):
                means = {c: X_ctx[:, i].mean() for i, c in enumerate(cov_present)}
                X_full = np.column_stack([
                    future_cov_df[c].values[:h] if c in cov_future
                    else np.full(h, means[c])
                    for c in cov_present
                ])
                linear_pred = lr.predict(X_full)
            else:
                linear_pred = lr.predict(X_future)
        else:
            linear_pred = np.full(h, lr.predict(X_ctx).mean())
    else:
        linear_pred = np.full(h, lr.predict(X_ctx).mean())

    return np.clip(res_forecast + linear_pred, 0.0, 0.8)


def run_chronos(
    bundle,
    horizon: int,
    context_length: int = 512,
    device: str = "auto",
    model_id: str = CHRONOS_MODEL_ID,
) -> pd.DataFrame:
    """
    Zero-shot Chronos-2 forecast over the test split.

    In covariate mode, uses linear-residual detrending so that soil
    covariate signals are incorporated without changing the Chronos API.

    Returns
    -------
    DataFrame [unique_id, ds, Chronos-2]
    """
    pipeline = _load_chronos(model_id, device)

    if bundle.covariate_cols:
        print(
            "  [Chronos-2] Covariate mode: linear-residual approach "
            f"({bundle.covariate_cols})"
        )
        predict_fn = lambda ctx, fcov, h: _chronos_predict_covariate(pipeline, ctx, fcov, h)
    else:
        predict_fn = lambda ctx, fcov, h: _chronos_predict_univariate(pipeline, ctx, fcov, h)

    print(
        f"  [Chronos-2] Rolling forecast | mode={bundle.mode} | "
        f"horizon={horizon} | context_length={context_length} | "
        f"test_len={len(bundle.test_df)}"
    )

    return _rolling_forecast(
        predict_window_fn=predict_fn,
        trainval_df=bundle.trainval_df,
        test_df=bundle.test_df,
        horizon=horizon,
        context_length=context_length,
        pred_col="Chronos-2",
        covariate_cols=bundle.covariate_cols,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Moirai
# ─────────────────────────────────────────────────────────────────────────────

def _load_moirai(model_id: str = MOIRAI_MODEL_ID, horizon: int = 24, context_length: int = 512, n_covariates: int = 0, device: str = "auto"):
    """
    Lazy-load and cache the Moirai predictor.
    Reloads if horizon or covariate count changed since last call.
    """
    global _moirai_predictor, _moirai_pred_len, _moirai_n_cov

    if (
        _moirai_predictor is not None
        and _moirai_pred_len == horizon
        and _moirai_n_cov == n_covariates
    ):
        return _moirai_predictor

    try:
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    except ImportError as e:
        raise ImportError("uni2ts is not installed. Run: pip install uni2ts") from e

    device_map = (
        device if device != "auto"
        else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    )
    print(f"  Loading Moirai from '{model_id}' on {device_map} ...")
    if n_covariates > 0:
        print(f"    past_feat_dynamic_real_dim={n_covariates} (soil covariates)")

    module = MoiraiModule.from_pretrained(model_id)
    model = MoiraiForecast(
        module=module,
        prediction_length=horizon,
        context_length=context_length,
        patch_size="auto",
        num_samples=20,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=n_covariates,
    )
    _moirai_predictor = model.create_predictor(batch_size=32)
    _moirai_pred_len  = horizon
    _moirai_n_cov     = n_covariates
    return _moirai_predictor


def _moirai_predict_window(predictor, ctx_df: pd.DataFrame, covariate_cols: list, h: int) -> np.ndarray:
    """
    Run Moirai on one context window via GluonTS PandasDataset.
    Passes covariate columns as past_feat_dynamic_real when available.
    Returns the median forecast, truncated to h steps.
    """
    try:
        from gluonts.dataset.pandas import PandasDataset
    except ImportError as e:
        raise ImportError("gluonts is not installed. Run: pip install gluonts") from e

    ctx = ctx_df.rename(columns={"unique_id": "item_id", "ds": "timestamp"}).copy()

    cov_present = [c for c in covariate_cols if c in ctx.columns]

    try:
        ds = PandasDataset.from_long_dataframe(
            ctx,
            target="y",
            item_id="item_id",
            timestamp="timestamp",
            past_feat_dynamic_real=cov_present if cov_present else None,
            freq="1h",
        )
    except TypeError:
        # Older gluonts versions may not support past_feat_dynamic_real kwarg
        ds = PandasDataset.from_long_dataframe(
            ctx[["item_id", "timestamp", "y"]],
            target="y",
            item_id="item_id",
            timestamp="timestamp",
            freq="1h",
        )

    forecasts = list(predictor.predict(ds))
    if not forecasts:
        return np.full(h, np.nan)

    median = forecasts[0].quantile(0.5)
    return np.asarray(median[:h], dtype=float)


def run_moirai(
    bundle,
    horizon: int,
    context_length: int = 512,
    device: str = "auto",
) -> pd.DataFrame:
    """
    Zero-shot Moirai forecast over the test split.

    In covariate mode: uses moirai-1.1-R-small with past_feat_dynamic_real
    so soil covariates condition the forecast. Moirai-2.0 dropped this
    feature, so the model ID is switched automatically.

    Returns
    -------
    DataFrame [unique_id, ds, Moirai-2]
    """
    n_cov = len(bundle.covariate_cols)

    if n_cov > 0:
        print(
            f"  [Moirai] Covariate mode: past_feat_dynamic_real={bundle.covariate_cols}"
        )

    predictor = _load_moirai(MOIRAI_MODEL_ID, horizon, context_length, n_cov, device)

    print(
        f"  [Moirai] Rolling forecast | mode={bundle.mode} | "
        f"horizon={horizon} | context_length={context_length} | "
        f"test_len={len(bundle.test_df)}"
    )

    def _predict_fn(ctx_df, future_cov_df, h):
        return _moirai_predict_window(predictor, ctx_df, bundle.covariate_cols, h)

    return _rolling_forecast(
        predict_window_fn=_predict_fn,
        trainval_df=bundle.trainval_df,
        test_df=bundle.test_df,
        horizon=horizon,
        context_length=context_length,
        pred_col="Moirai-2",
        covariate_cols=bundle.covariate_cols,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

ALL_ZERO_SHOT_MODELS = ["Chronos-2", "Moirai-2"]


def run_zero_shot_model(
    model_name: str,
    bundle,
    horizon: int,
    context_length: int = 512,
    device: str = "auto",
) -> pd.DataFrame:
    """
    Top-level dispatcher used by main_forecast_zero_shot.py.

    Parameters
    ----------
    model_name     : "Chronos-2" | "Moirai-2"
    bundle         : DataBundle from dataloader.load_dataset()
    horizon        : rolling window step size (hours)
    context_length : max context rows per window
    device         : "auto" | "cpu" | "cuda"

    Returns
    -------
    DataFrame [unique_id, ds, <model_name>]
    """
    if model_name == "Chronos-2":
        return run_chronos(bundle, horizon=horizon, context_length=context_length, device=device)
    elif model_name == "Moirai-2":
        return run_moirai(bundle, horizon=horizon, context_length=context_length, device=device)
    else:
        raise ValueError(
            f"Unknown zero-shot model '{model_name}'. "
            f"Choose from: {ALL_ZERO_SHOT_MODELS}"
        )
