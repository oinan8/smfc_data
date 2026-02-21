"""
End-to-end PyTorch forecasting models for MFC voltage prediction.

All models take a lookback window of features as input and predict the
next `horizon` steps of voltage only (multivariate-in, univariate-out).

Models
------
    TransformerForecaster  — multi-head self-attention encoder + linear head
    LSTMForecaster         — stacked LSTM + linear head
    GRUForecaster          — stacked GRU + linear head  (labelled "RNN")

Training protocol
-----------------
    - MSE loss, Adam optimiser
    - Early stopping on val MSE (patience configurable)
    - StandardScaler fitted on train features only, applied to all splits
    - Voltage is always feature index 0; target = predictions[:, 0..horizon]

Public API
----------
    build_model(name, n_features, horizon, **kwargs) → nn.Module
    fit_model(model, trainval_data, train_len, val_len, ...) → (model, history)
    predict_model(model, trainval_arr, test_df, scaler, ...) → forecast_df
"""

import math
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ALL_E2E_MODELS = ["Transformer", "LSTM", "RNN"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """
    Sliding window dataset over a pre-scaled numpy array.

    Parameters
    ----------
    data : np.ndarray, shape (T, n_features)
        Feature matrix in chronological order. Voltage must be column 0.
    input_len : int
        Number of past timesteps fed to the model.
    horizon : int
        Number of future voltage steps to predict.
    start_idx, end_idx : int
        Window indices to include. Windows are:
            X = data[i : i+input_len]
            y = data[i+input_len : i+input_len+horizon, 0]   ← voltage only
        for i in range(start_idx, end_idx - input_len - horizon + 1).
    """

    def __init__(self, data: np.ndarray, input_len: int, horizon: int,
                 start_idx: int = 0, end_idx: Optional[int] = None):
        end_idx = end_idx or len(data)
        X, y = [], []
        for i in range(start_idx, end_idx - input_len - horizon + 1):
            X.append(data[i : i + input_len])
            y.append(data[i + input_len : i + input_len + horizon, 0])
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """Stacked LSTM with a linear projection head."""

    def __init__(self, n_features: int, hidden_dim: int, n_layers: int,
                 horizon: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):                   # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])     # (B, horizon)


class GRUForecaster(nn.Module):
    """Stacked GRU with a linear projection head (labelled 'RNN' in the CLI)."""

    def __init__(self, n_features: int, hidden_dim: int, n_layers: int,
                 horizon: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            n_features, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class TransformerForecaster(nn.Module):
    """
    Lightweight Transformer encoder with a linear forecast head.

    Architecture:
        input_proj  → positional encoding → TransformerEncoder → head
    The head takes the last token's representation and maps to horizon steps.
    """

    def __init__(self, n_features: int, d_model: int, n_heads: int,
                 n_layers: int, horizon: int, dropout: float = 0.1,
                 max_len: int = 1024):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)

        # Sinusoidal positional encoding (fixed, not learnable)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                              # x: (B, T, F)
        x = self.input_proj(x)                        # (B, T, d_model)
        x = self.dropout(x + self.pe[:, :x.size(1)])  # add positional enc
        x = self.encoder(x)
        return self.head(x[:, -1, :])                 # (B, horizon)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    name: str,
    n_features: int,
    horizon: int,
    hidden_dim: int = 64,
    n_layers: int   = 2,
    n_heads: int    = 4,
    dropout: float  = 0.1,
) -> nn.Module:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name       : "Transformer" | "LSTM" | "RNN"
    n_features : number of input channels (1 for univariate, 4 for covariate)
    horizon    : forecast length
    hidden_dim : d_model for Transformer, hidden_size for LSTM/GRU
    n_heads    : Transformer only
    """
    name = name.upper()
    if name == "TRANSFORMER":
        # Ensure d_model is divisible by n_heads
        d_model = max(hidden_dim, n_heads) if hidden_dim % n_heads == 0 \
                  else (hidden_dim // n_heads + 1) * n_heads
        return TransformerForecaster(n_features, d_model, n_heads, n_layers, horizon, dropout)
    elif name == "LSTM":
        return LSTMForecaster(n_features, hidden_dim, n_layers, horizon, dropout)
    elif name == "RNN":
        return GRUForecaster(n_features, hidden_dim, n_layers, horizon, dropout)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from: {ALL_E2E_MODELS}")


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_arrays(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Stack feature columns into numpy arrays and fit a StandardScaler on train.

    Voltage (y) is always placed at column index 0 so target extraction
    in SlidingWindowDataset is consistent.

    Returns
    -------
    train_arr, val_arr, test_arr : np.ndarray, shape (T, n_features)
    scaler : fitted StandardScaler
    """
    def _to_array(df):
        cols = ["y"] + [c for c in feature_cols if c != "y"]
        return df[cols].values.astype(np.float32)

    train_arr = _to_array(train_df)
    val_arr   = _to_array(val_df)
    test_arr  = _to_array(test_df)

    scaler = StandardScaler()
    scaler.fit(train_arr)

    return (
        scaler.transform(train_arr),
        scaler.transform(val_arr),
        scaler.transform(test_arr),
        scaler,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def fit_model(
    model: nn.Module,
    train_arr: np.ndarray,
    val_arr: np.ndarray,
    input_len: int,
    horizon: int,
    n_epochs: int   = 100,
    lr: float       = 1e-3,
    batch_size: int = 32,
    patience: int   = 10,
    device: str     = "cpu",
) -> tuple[nn.Module, dict]:
    """
    Train model with MSE loss and early stopping on val MSE.

    The val dataset is built using the combined (train+val) array so that
    windows starting in the val period can look back into train.

    Returns
    -------
    model      : best model (lowest val MSE)
    history    : {"train_loss": [...], "val_loss": [...]}
    """
    model = model.to(device)
    trainval_arr = np.concatenate([train_arr, val_arr], axis=0)
    train_len    = len(train_arr)

    train_ds = SlidingWindowDataset(trainval_arr, input_len, horizon,
                                    start_idx=0, end_idx=train_len)
    val_ds   = SlidingWindowDataset(trainval_arr, input_len, horizon,
                                    start_idx=train_len, end_idx=len(trainval_arr))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val  = float("inf")
    best_state = deepcopy(model.state_dict())
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):
        # ── train ──
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())

        # ── val ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_losses.append(criterion(model(X), y).item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:>3}/{n_epochs}  "
                  f"train_mse={train_loss:.5f}  val_mse={val_loss:.5f}")

        if val_loss < best_val:
            best_val   = val_loss
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    Early stop at epoch {epoch} (patience={patience})")
                break

    model.load_state_dict(best_state)
    print(f"    Best val MSE: {best_val:.5f}")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Inference (rolling)
# ─────────────────────────────────────────────────────────────────────────────

def predict_model(
    model: nn.Module,
    trainval_arr: np.ndarray,   # scaled, voltage at col 0
    test_arr: np.ndarray,       # scaled, same column order
    scaler: StandardScaler,
    input_len: int,
    horizon: int,
    test_df: pd.DataFrame,      # for unique_id and ds timestamps
    model_name: str,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Rolling forecast over the test period.

    At each step the context is extended with actual (scaled) test values —
    identical protocol to the zero-shot rolling evaluation.
    Predictions are inverse-transformed back to voltage scale.

    Returns
    -------
    DataFrame [unique_id, ds, <model_name>]
    """
    model.eval()
    uid_col, ds_col = "unique_id", "ds"
    records = []

    for uid, test_series in test_df.groupby(uid_col, sort=True):
        test_series = test_series.sort_values(ds_col).reset_index(drop=True)
        n_test = len(test_series)

        # Context starts as the full trainval for this series (single series here)
        ctx = list(trainval_arr)   # list of rows for easy append
        test_scaled = list(test_arr)

        preds = []
        for start in range(0, n_test, horizon):
            end = min(start + horizon, n_test)
            h   = end - start

            window = np.array(ctx[-input_len:], dtype=np.float32)   # (input_len, F)
            X = torch.tensor(window).unsqueeze(0).to(device)        # (1, input_len, F)

            with torch.no_grad():
                pred_scaled = model(X).squeeze(0).cpu().numpy()     # (horizon,)

            preds.append(pred_scaled[:h])

            # Extend context with actual scaled test values
            ctx.extend(test_scaled[start:end])

        preds_flat = np.concatenate(preds)   # (n_test,)

        # Inverse-transform voltage (column 0) only
        dummy = np.zeros((len(preds_flat), scaler.n_features_in_), dtype=np.float32)
        dummy[:, 0] = preds_flat
        voltage_pred = scaler.inverse_transform(dummy)[:, 0]

        for i, row in test_series.iterrows():
            records.append({uid_col: uid, ds_col: row[ds_col], model_name: voltage_pred[i]})

    return pd.DataFrame(records)
