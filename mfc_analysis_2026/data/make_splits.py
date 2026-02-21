"""
Create time-based train/val/test splits for synthetic and real datasets.

Splits are strictly temporal (no shuffling) to prevent data leakage.
Default ratio: 70% train / 15% val / 15% test.

Output
------
    data/splits/synthetic/{train,val,test}.csv
    data/splits/real/{train,val,test}.csv

Usage
-----
    python data/make_splits.py
    python data/make_splits.py --splits-dir data/splits/
    python data/make_splits.py --synthetic data/synthetic/data.csv --skip-real
"""

import argparse
from pathlib import Path

import pandas as pd

SPLIT_RATIOS = (0.70, 0.15, 0.15)

# Column name for the time axis in each dataset
SYNTHETIC_TIME_COL = "timestamp"
REAL_TIME_COL = "timestamp"
REAL_ID_COL = "item_id"


def compute_split_indices(n: int, ratios: tuple = SPLIT_RATIOS) -> tuple[int, int]:
    """
    Return (train_end, val_end) as exclusive right-boundary indices.

    Slices:
        train = df[:train_end]
        val   = df[train_end:val_end]
        test  = df[val_end:]
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "Split ratios must sum to 1"
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))
    return train_end, val_end


def split_single_series(
    df: pd.DataFrame,
    time_col: str,
    ratios: tuple = SPLIT_RATIOS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort a single-series DataFrame by time_col and split into train/val/test.
    Returns (train_df, val_df, test_df).
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    train_end, val_end = compute_split_indices(len(df), ratios)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def split_and_save(
    input_csv: str,
    output_dir: str,
    dataset_name: str,
    time_col: str,
    id_col: str | None = None,
    ratios: tuple = SPLIT_RATIOS,
) -> dict[str, str]:
    """
    Read input_csv, split by time, and write split CSVs.

    For single-series data (id_col=None): splits the whole DataFrame.
    For multi-series data (id_col given): splits each series independently,
    then concatenates results per split to avoid leakage across series.

    Returns dict mapping split name → file path.
    """
    df = pd.read_csv(input_csv, parse_dates=[time_col])
    print(f"\n[{dataset_name}] Loaded {len(df):,} rows from {input_csv}")

    if id_col is None or id_col not in df.columns:
        # Single series (synthetic)
        train, val, test = split_single_series(df, time_col, ratios)
        splits = {"train": train, "val": val, "test": test}
    else:
        # Multi-series (real / GIFT-eval)
        train_parts, val_parts, test_parts = [], [], []
        for item_id, group in df.groupby(id_col, sort=True):
            tr, va, te = split_single_series(group, time_col, ratios)
            train_parts.append(tr)
            val_parts.append(va)
            test_parts.append(te)
        splits = {
            "train": pd.concat(train_parts, ignore_index=True),
            "val":   pd.concat(val_parts,   ignore_index=True),
            "test":  pd.concat(test_parts,  ignore_index=True),
        }

    out_dir = Path(output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split_name, split_df in splits.items():
        out_path = out_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        paths[split_name] = str(out_path)
        print(f"  {split_name:5s}: {len(split_df):>8,} rows → {out_path}")

    return paths


def make_all_splits(
    synthetic_csv: str = "data/synthetic/data.csv",
    real_csv: str = "data/real/solar_hourly.csv",
    splits_dir: str = "data/splits/",
    skip_synthetic: bool = False,
    skip_real: bool = False,
) -> None:
    """
    Orchestrate splits for both datasets.
    Skips a dataset if its source CSV does not exist (with a warning).
    """
    if not skip_synthetic:
        if Path(synthetic_csv).exists():
            split_and_save(
                input_csv=synthetic_csv,
                output_dir=splits_dir,
                dataset_name="synthetic",
                time_col=SYNTHETIC_TIME_COL,
                id_col=None,
            )
        else:
            print(f"[WARNING] Synthetic CSV not found: {synthetic_csv}. "
                  "Run: python data/generate_synthetic.py")

    if not skip_real:
        if Path(real_csv).exists():
            split_and_save(
                input_csv=real_csv,
                output_dir=splits_dir,
                dataset_name="real",
                time_col=REAL_TIME_COL,
                id_col=REAL_ID_COL,
            )
        else:
            print(f"[WARNING] Real CSV not found: {real_csv}. "
                  "Run: python data/download_data.py")

    print("\nDone. Splits written to:", splits_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--synthetic", default="data/synthetic/data.csv")
    parser.add_argument("--real", default="data/real/solar_hourly.csv")
    parser.add_argument("--splits-dir", default="data/splits/")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()
    make_all_splits(
        synthetic_csv=args.synthetic,
        real_csv=args.real,
        splits_dir=args.splits_dir,
        skip_synthetic=args.skip_synthetic,
        skip_real=args.skip_real,
    )
