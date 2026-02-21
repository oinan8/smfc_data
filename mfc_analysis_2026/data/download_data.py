"""
Download the GIFT-eval solar/H dataset from HuggingFace.

The Salesforce/GiftEval dataset stores each time series as a row with
a list-valued 'target' column and a 'start' timestamp. This script
expands it to long-format and writes a flat CSV.

Usage
-----
    python data/download_data.py
    python data/download_data.py --subset solar/H --output-dir data/real/
    python data/download_data.py --token hf_xxx   # if dataset requires auth
"""

import argparse
from pathlib import Path

import pandas as pd

GIFT_EVAL_REPO = "Salesforce/GiftEval"
DEFAULT_SUBSET = "solar/H"


def normalize_gift_eval_df(raw_df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """
    Convert raw GIFT-eval DataFrame (wide-list format) to long-format.

    Raw schema: item_id (str), start (str/datetime), target (list[float])
    Output schema: item_id (str), timestamp (datetime), target (float)

    Each row in the raw data is one time series; 'target' is a list of
    observed values starting at 'start' with the given frequency.
    """
    records = []
    for _, row in raw_df.iterrows():
        item_id = str(row["item_id"])
        start = pd.Timestamp(row["start"])
        target = row["target"]
        n = len(target)
        timestamps = pd.date_range(start=start, periods=n, freq=freq)
        for ts, val in zip(timestamps, target):
            records.append({"item_id": item_id, "timestamp": ts, "target": float(val)})

    return pd.DataFrame(records)


def download_gift_eval_solar(
    output_dir: str = "data/real/",
    subset: str = DEFAULT_SUBSET,
    token: str | None = None,
) -> str:
    """
    Download the GIFT-eval solar subset and write to output_dir/solar_hourly.csv.

    Returns the path to the written CSV.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets huggingface_hub")

    print(f"Downloading {GIFT_EVAL_REPO} subset='{subset}' ...")
    raw = load_dataset(
        GIFT_EVAL_REPO,
        name=subset,
        split="train",
        token=token,
    )
    raw_df = raw.to_pandas()
    print(f"  Raw dataset: {len(raw_df)} series")

    long_df = normalize_gift_eval_df(raw_df, freq="h")
    print(f"  Expanded to {len(long_df):,} rows across {long_df['item_id'].nunique()} series")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "solar_hourly.csv"
    long_df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

    # Summary statistics
    series_lengths = long_df.groupby("item_id").size()
    print(f"  Series lengths: min={series_lengths.min()}, max={series_lengths.max()}, "
          f"median={series_lengths.median():.0f}")

    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GIFT-eval solar dataset")
    parser.add_argument("--subset", default=DEFAULT_SUBSET,
                        help="GIFT-eval subset key (default: solar/H)")
    parser.add_argument("--output-dir", default="data/real/")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (if dataset requires auth)")
    args = parser.parse_args()
    download_gift_eval_solar(args.output_dir, args.subset, args.token)
