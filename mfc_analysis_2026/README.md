# MFC Forecasting Playground

A benchmarking playground for forecasting **Microbial Fuel Cell (MFC) voltage** from soil covariates. Compares zero-shot foundation models against small end-to-end trained neural networks.

---

## Overview

MFCs generate electricity from microbial activity in soil. Voltage output is correlated with environmental conditions:

| Feature | Description |
|---|---|
| `voltage` | MFC output voltage (V) — **forecast target** |
| `soil_moisture` | Volumetric water content [0–1] |
| `soil_conductivity` | Ionic conductivity (S/m) |
| `soil_char` | Soil characteristic index [0–1] (e.g. clay content) |

Two forecasting paradigms are compared:

- **Zero-shot** — Chronos-2 and Moirai, used without any training
- **End-to-end (E2E)** — Transformer, LSTM, RNN trained from scratch with MSE loss

Both support **univariate** (voltage only) and **covariate** (all 4 features) modes.

---

## Setup

```bash
conda env create -f environment.yml
conda activate mfc_2026
```

---

## Data Pipeline

Run once, in order:

```bash
# 1. Generate 8760-hour synthetic MFC dataset (~instant)
python data/generate_synthetic.py

# 2. Create 70/15/15 temporal train/val/test splits
python data/make_splits.py

# 3. Inspect the data visually
python viz/plotting_synth_samples.py        # → viz/synth_samples.png
```

The synthetic dataset uses a Nernst-equation-inspired physics model:

```
V(t) = E₀ + α·log(1 + θ·moisture(t)) − β·exp(−k·conductivity(t))
            + γ·soil_char + δ·sin(2π·t/24) + ε
```

Splits are strictly temporal — no data leakage between train, val, and test.

---

## Zero-Shot Forecasting

No training required. Models are loaded from HuggingFace and run directly on the test split.

```bash
# Chronos-2 (Chronos-Bolt)
python main_forecast_zero_shot.py --model Chronos-2 --mode univariate
python main_forecast_zero_shot.py --model Chronos-2 --mode covariate

# Moirai (moirai-1.1-R-small)
python main_forecast_zero_shot.py --model Moirai-2 --mode univariate
python main_forecast_zero_shot.py --model Moirai-2 --mode covariate

# Run both models
python main_forecast_zero_shot.py --model all --mode covariate

# Custom horizon and context length
python main_forecast_zero_shot.py --model Chronos-2 --horizon 48 --context-length 336
```

### How covariate mode works

| Model | Strategy |
|---|---|
| **Chronos-2** | Linear-residual: fit `Ridge(covariates → voltage)` on context, forecast residuals with Chronos, add back the linear covariate prediction for the test window |
| **Moirai** | Passes soil covariates as `past_feat_dynamic_real` inside GluonTS `PandasDataset`; Moirai conditions on them internally |

### Key options

| Flag | Default | Description |
|---|---|---|
| `--model` | `Chronos-2` | `Chronos-2`, `Moirai-2`, or `all` |
| `--mode` | `univariate` | `univariate` or `covariate` |
| `--horizon` | `24` | Rolling forecast window size (hours) |
| `--context-length` | `512` | Max history rows fed per window |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |

> **First run** downloads model weights from HuggingFace (~1 GB Chronos, ~2 GB Moirai). Cached on subsequent runs.

---

## End-to-End Forecasting

Small PyTorch models trained with MSE loss and early stopping on the val split.

```bash
# Single model
python main_forecast_e2e.py --model LSTM --mode covariate
python main_forecast_e2e.py --model Transformer --mode covariate
python main_forecast_e2e.py --model RNN --mode covariate

# All three models in one run
python main_forecast_e2e.py --model all --mode covariate

# Univariate baseline (voltage-only input)
python main_forecast_e2e.py --model all --mode univariate

# Custom architecture and training
python main_forecast_e2e.py --model Transformer --mode covariate \
    --hidden-dim 128 --n-layers 3 --n-epochs 200 --patience 20 --horizon 48
```

### Model architecture

All models: **multivariate-in → voltage-only-out** (standard MISO setup).

| Model | Architecture | Params (defaults) |
|---|---|---|
| `Transformer` | Input projection → sinusoidal PE → TransformerEncoder → linear head | ~102K |
| `LSTM` | Stacked LSTM → linear head | ~53K |
| `RNN` | Stacked GRU → linear head | ~40K |

Input is standardised with `StandardScaler` fitted on the training split only. Predictions are inverse-transformed back to voltage scale.

### Key options

| Flag | Default | Description |
|---|---|---|
| `--model` | `LSTM` | `Transformer`, `LSTM`, `RNN`, or `all` |
| `--mode` | `covariate` | `univariate` or `covariate` |
| `--horizon` | `24` | Forecast steps (hours) |
| `--input-len` | `168` | Lookback window (hours, default = 1 week) |
| `--hidden-dim` | `64` | Hidden size (d_model for Transformer) |
| `--n-layers` | `2` | Number of stacked layers |
| `--n-heads` | `4` | Transformer attention heads |
| `--n-epochs` | `100` | Max training epochs |
| `--patience` | `10` | Early stopping patience (val MSE) |
| `--lr` | `1e-3` | Adam learning rate |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |

---

## Outputs

Every run produces:

| Output | Location | Description |
|---|---|---|
| Metrics CSV | `results/results.csv` | Append-only log: `run_datetime, model, dataset, mode, horizon, MSE, MAE` |
| Forecast CSV | `results/forecasts/<model>_<dataset>_<mode>_h<H>_<datetime>.csv` | Raw predictions per timestep |
| Plot PNG | `results/forecast_{zero_shot,e2e}_<datetime>.png` | Comparison plot (auto-generated) |

The comparison plot contains:
- **Zero-shot**: context window + true test signal + all model predictions, plus MSE/MAE bar charts
- **E2E**: same time-series panel + training/val loss curves (log scale) + MSE/MAE bar charts

---

## Benchmark Results (synthetic dataset, horizon=24h)

| Model | Type | Mode | MSE | MAE |
|---|---|---|---|---|
| **Chronos-2** | Zero-shot | covariate | **0.000276** | **0.012812** |
| LSTM | E2E | covariate | 0.000735 | 0.020707 |
| Transformer | E2E | covariate | 0.000812 | 0.021544 |
| Chronos-2 | Zero-shot | univariate | 0.002887 | 0.041497 |
| Moirai-2 | Zero-shot | covariate | 0.004988 | 0.052518 |
| Moirai-2 | Zero-shot | univariate | 0.006538 | 0.058667 |

> Results on the held-out test split (last 15% of the 8760-hour synthetic series). Rolling evaluation with horizon-sized windows; actual test values used as context for each subsequent window.

---

## Project Structure

```
mfc_analysis_2026/
├── environment.yml                  # Conda environment
├── requirements.txt                 # Pip dependencies
│
├── data/
│   ├── generate_synthetic.py        # Physics-inspired MFC dataset generator
│   ├── download_data.py             # GIFT-eval solar data downloader (optional)
│   └── make_splits.py              # 70/15/15 temporal train/val/test splitter
│
├── dataloader.py                    # load_dataset() → DataBundle
├── evaluate.py                      # compute_metrics(), evaluate_forecast(), save_forecast()
│
├── models/
│   ├── zero_shot.py                 # Chronos-2 and Moirai wrappers + rolling evaluation
│   └── e2e.py                       # Transformer, LSTM, GRU + training loop
│
├── main_forecast_zero_shot.py       # Zero-shot CLI entry point
├── main_forecast_e2e.py             # E2E training + evaluation CLI entry point
│
└── viz/
    ├── plotting_synth_samples.py    # Data inspection plot
    └── plotting_forecast_results.py # Standalone plot regeneration (optional)
```

---

## Evaluation Protocol

All models use the same rolling evaluation on the held-out test split:

1. Context = last `context_length` rows of train+val (actual data, not predictions)
2. Predict the next `horizon` steps of voltage
3. Extend context with **actual** test values (not model predictions) for the next window
4. Repeat until all test steps are covered
5. Compute MSE and MAE over all predicted steps

This prevents error compounding from corrupting the evaluation.

---

## Notes

- `soil_char` is constant (0.7) in the synthetic dataset — it contributes only a fixed offset to the Nernst voltage model.
- E2E models write no checkpoints to disk; best weights are held in memory during training.
- `results/results.csv` is append-only — re-running a model adds new rows rather than overwriting.
- The `--device auto` flag selects CUDA if available, CPU otherwise.
