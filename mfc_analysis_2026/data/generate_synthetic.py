"""
Synthetic MFC (Microbial Fuel Cell) dataset generator.

Produces 1 year of hourly data with physics-inspired correlations between
soil covariates and MFC output voltage based on a Nernst-equation model.

Usage
-----
    python data/generate_synthetic.py
    python data/generate_synthetic.py --n-hours 8760 --seed 42 --output data/synthetic/data.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── Nernst-inspired MFC voltage parameters ────────────────────────────────────
E0 = 0.40       # baseline open-circuit potential (V), typical range 0.3–0.8 V
ALPHA = 0.15    # moisture sensitivity (Nernst log coefficient)
THETA_M = 8.0   # moisture saturation scale for log term
BETA = 0.10     # conductivity correction amplitude
K_C = 1.50      # conductivity decay constant
GAMMA = 0.05    # soil-characteristic weight
DELTA = 0.02    # amplitude of diurnal bioelectrochemical cycle
NOISE_STD = 0.01  # measurement noise (V)

SOIL_CHAR = 0.70  # constant soil-characteristic scalar (clay-like, range 0–1)


def _ar1(n: int, rng: np.random.Generator, phi: float = 0.85, sigma: float = 0.05) -> np.ndarray:
    """Generate a zero-mean AR(1) process of length n."""
    out = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def generate_covariate_series(
    n_hours: int = 8760,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic covariate time series.

    Returns
    -------
    dict with keys:
        'moisture'       : np.ndarray (n_hours,), range [0, 1]
        'conductivity'   : np.ndarray (n_hours,), range [0.1, 2.0]
        'soil_char'      : float scalar [0, 1]
        'timestamps'     : pd.DatetimeIndex, hourly from 2023-01-01
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours, dtype=float)

    # Soil moisture: 3-day sinusoidal cycle + seasonal drift + AR(1) noise
    moisture = (
        0.50
        + 0.30 * np.sin(2 * np.pi * t / 72)          # 3-day cycle
        + 0.08 * np.sin(2 * np.pi * t / 8760)         # annual seasonal drift
        + _ar1(n_hours, rng, phi=0.85, sigma=0.04)
    )
    moisture = np.clip(moisture, 0.0, 1.0)

    # Soil conductivity: lags moisture by 6 h + small independent noise
    lag = 6
    lagged_moisture = np.concatenate([moisture[:lag][::-1], moisture[:-lag]])
    conductivity = (
        0.80
        + 0.60 * lagged_moisture
        + _ar1(n_hours, rng, phi=0.70, sigma=0.03)
    )
    conductivity = np.clip(conductivity, 0.1, 2.0)

    timestamps = pd.date_range("2023-01-01", periods=n_hours, freq="h")

    return {
        "moisture": moisture,
        "conductivity": conductivity,
        "soil_char": SOIL_CHAR,
        "timestamps": timestamps,
    }


def compute_mfc_voltage(
    moisture: np.ndarray,
    conductivity: np.ndarray,
    soil_char: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Physics-inspired Nernst-equation model for MFC voltage (V).

    V(t) = E0
           + alpha * log(1 + theta_m * moisture(t))   [Nernst log, moisture drives ions]
           - beta  * exp(-k_c * conductivity(t))        [conductivity reduces internal R]
           + gamma * soil_char                          [static organic matter boost]
           + delta * sin(2π * t / 24)                  [diurnal bioelectrochemical cycle]
           + N(0, noise_std)                            [measurement noise]

    Output is clipped to [0.0, 0.8] V (realistic MFC operating range).
    """
    rng = np.random.default_rng(seed + 1)
    n = len(moisture)
    t = np.arange(n, dtype=float)

    voltage = (
        E0
        + ALPHA * np.log1p(THETA_M * moisture)
        - BETA * np.exp(-K_C * conductivity)
        + GAMMA * soil_char
        + DELTA * np.sin(2 * np.pi * t / 24)
        + rng.normal(0, NOISE_STD, n)
    )
    return np.clip(voltage, 0.0, 0.8)


def generate_synthetic_dataset(
    output_path: str = "data/synthetic/data.csv",
    n_hours: int = 8760,
    seed: int = 42,
) -> None:
    """
    Orchestrate covariate and voltage generation, write output CSV.

    Output columns: timestamp, voltage, soil_moisture, soil_conductivity, soil_char
    """
    covariates = generate_covariate_series(n_hours=n_hours, seed=seed)
    voltage = compute_mfc_voltage(
        moisture=covariates["moisture"],
        conductivity=covariates["conductivity"],
        soil_char=covariates["soil_char"],
        seed=seed,
    )

    df = pd.DataFrame({
        "timestamp": covariates["timestamps"],
        "voltage": voltage,
        "soil_moisture": covariates["moisture"],
        "soil_conductivity": covariates["conductivity"],
        "soil_char": covariates["soil_char"],
    })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved synthetic dataset ({len(df)} rows) → {out}")
    print(f"  Voltage: mean={df['voltage'].mean():.4f} V, std={df['voltage'].std():.4f} V")
    print(f"  Moisture: mean={df['soil_moisture'].mean():.3f}, std={df['soil_moisture'].std():.3f}")
    print(f"  Conductivity: mean={df['soil_conductivity'].mean():.3f} S/m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic MFC dataset")
    parser.add_argument("--output", default="data/synthetic/data.csv")
    parser.add_argument("--n-hours", type=int, default=8760)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_synthetic_dataset(args.output, args.n_hours, args.seed)
