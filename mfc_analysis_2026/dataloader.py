"""
Shared data loader for the MFC forecasting playground.

Reads pre-split CSVs and returns a DataBundle suitable for
NeuralForecast, StatsForecast, and zero-shot model wrappers.

Usage
-----
    from dataloader import load_dataset
    bundle = load_dataset(dataset="synthetic", mode="univariate")
    bundle = load_dataset(dataset="synthetic", mode="covariate")
"""

from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
COVARIATE_COLS = ["soil_moisture", "soil_conductivity", "soil_char"]
FREQ = "1h"
SYNTHETIC_UNIQUE_ID = "mfc_0"

# Raw CSV column → NeuralForecast column
_SYNTHETIC_RENAME = {"timestamp": "ds", "voltage": "y"}


class DataBundle(NamedTuple):
    """
    Unified data container for all model families.

    All DataFrames are in NeuralForecast long-format:
        unique_id (str) | ds (datetime) | y (float) | [covariate cols ...]

    trainval_df is the concatenation of train + val; used as the
    context window for zero-shot models.
    covariate_cols is empty in univariate mode.
    """
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    trainval_df: pd.DataFrame
    covariate_cols: list
    horizon: int
    freq: str
    dataset_name: str
    mode: str


def _load_split(path: Path, mode: str) -> pd.DataFrame:
    """Read one split CSV and normalise to NeuralForecast format."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.rename(columns=_SYNTHETIC_RENAME)
    df["unique_id"] = SYNTHETIC_UNIQUE_ID
    df = df.sort_values("ds").reset_index(drop=True)

    base_cols = ["unique_id", "ds", "y"]
    if mode == "covariate":
        cols = base_cols + COVARIATE_COLS
    else:
        cols = base_cols

    return df[cols]


def load_dataset(
    dataset: str = "synthetic",
    mode: str = "univariate",
    splits_dir: str = "data/splits/",
    horizon: int = 24,
) -> DataBundle:
    """
    Load pre-split data and return a DataBundle.

    Parameters
    ----------
    dataset : "synthetic"
        Only synthetic is supported at this stage.
    mode : "univariate" | "covariate"
        univariate  — only [unique_id, ds, y]
        covariate   — adds soil_moisture, soil_conductivity, soil_char
    splits_dir : str
        Root directory containing {dataset}/{train,val,test}.csv
    horizon : int
        Default forecast horizon (passed through to DataBundle).

    Raises
    ------
    ValueError
        If dataset or mode is unrecognised.
    FileNotFoundError
        If split CSVs are missing (run data/make_splits.py first).
    """
    if dataset != "synthetic":
        raise ValueError(
            f"Dataset '{dataset}' is not yet supported. Only 'synthetic' is available."
        )
    if mode not in ("univariate", "covariate"):
        raise ValueError(f"mode must be 'univariate' or 'covariate', got '{mode}'")

    split_root = Path(splits_dir) / dataset
    for split_name in ("train", "val", "test"):
        p = split_root / f"{split_name}.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: {p}\n"
                "Run: python data/make_splits.py"
            )

    train_df = _load_split(split_root / "train.csv", mode)
    val_df   = _load_split(split_root / "val.csv",   mode)
    test_df  = _load_split(split_root / "test.csv",  mode)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    covariate_cols = COVARIATE_COLS if mode == "covariate" else []

    return DataBundle(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        trainval_df=trainval_df,
        covariate_cols=covariate_cols,
        horizon=horizon,
        freq=FREQ,
        dataset_name=dataset,
        mode=mode,
    )


def to_zero_shot_context(
    bundle: DataBundle,
    context_length: int,
) -> pd.DataFrame:
    """
    Return the last `context_length` rows of trainval_df per series.

    Zero-shot models use this as their input context. The actual test
    targets remain in bundle.test_df for evaluation.
    """
    def _tail(group: pd.DataFrame) -> pd.DataFrame:
        return group.tail(context_length)

    ctx = (
        bundle.trainval_df
        .groupby("unique_id", group_keys=False)
        .apply(_tail)
        .reset_index(drop=True)
    )
    return ctx
