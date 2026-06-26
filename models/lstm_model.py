"""
LSTM neural network model for time series forecasting.

Implements deep learning LSTM networks using PyTorch for time series prediction.
Architecture: Input(look_back) -> LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dense(1)
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42


class _LSTMNet(nn.Module):
    """Two-layer LSTM network with dropout."""

    def __init__(
        self,
        hidden1: int,
        hidden2: int,
        dropout: float,
        horizon: int,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(1, hidden1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc = nn.Linear(hidden2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, look_back, 1)."""
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])  # last timestep only
        return out


class LSTMForecaster:
    """
    PyTorch LSTM forecaster with sliding-window sequences.

    MinMaxScaler is fit on training data only. Early stopping is applied
    on a held-out validation split (last 10% of training data).
    """

    def __init__(
        self,
        look_back: int = 30,
        hidden_size1: int = 64,
        hidden_size2: int = 32,
        dropout: float = 0.2,
        horizon: int = 1,
        lr: float = 0.001,
        max_epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.look_back = look_back
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.dropout = dropout
        self.horizon = horizon
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_seed = random_seed

        self._scaler: Optional[MinMaxScaler] = None
        self._model: Optional[_LSTMNet] = None
        self._train_values: Optional[np.ndarray] = None  # raw unscaled
        self._train_index: Optional[pd.DatetimeIndex] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train: pd.Series) -> "LSTMForecaster":
        """
        Fit on training series.

        MinMaxScaler is fit on TRAIN only. Last 10% of training rows
        are held out as a validation split for early stopping.
        """
        self._set_seeds()

        self._train_values = train.values.astype(np.float32)
        self._train_index = train.index

        # Fit scaler on all training data (before sequence construction)
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self._scaler.fit_transform(
            self._train_values.reshape(-1, 1)
        ).flatten()

        # Split into train/val sequences (80/10 or 90/10 split at row level)
        n = len(scaled)
        val_start = int(n * 0.9)

        train_scaled = scaled[:val_start]
        val_scaled = scaled[val_start:]

        X_train, y_train = self._build_sequences(train_scaled)
        X_val, y_val = self._build_sequences(val_scaled)

        if X_val.shape[0] == 0:
            # Fallback: use train sequences as val when series is too short
            X_val, y_val = X_train, y_train

        self._model = _LSTMNet(
            self.hidden_size1, self.hidden_size2, self.dropout, self.horizon
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._train_loop(
            X_train, y_train, X_val, y_val, optimizer, criterion
        )

        return self

    def forecast(self, steps: int) -> pd.Series:
        """
        Auto-regressive multi-step forecast.

        Returns inverse-scaled pd.Series with DatetimeIndex continuing
        from the end of training data.
        """
        if self._model is None or self._scaler is None:
            raise RuntimeError("Call fit() before forecast().")

        self._model.eval()

        scaled_all = self._scaler.transform(
            self._train_values.reshape(-1, 1)
        ).flatten()

        # Seed window: last look_back values from training
        window = list(scaled_all[-self.look_back :])
        predictions: list[float] = []

        with torch.no_grad():
            for _ in range(steps):
                x = torch.tensor(window[-self.look_back :], dtype=torch.float32)
                x = x.unsqueeze(0).unsqueeze(-1)  # (1, look_back, 1)
                pred = self._model(x)  # (1, horizon)
                next_val = float(pred[0, 0].item())
                predictions.append(next_val)
                window.append(next_val)

        preds_array = np.array(predictions, dtype=np.float32).reshape(-1, 1)
        inv_preds = self._scaler.inverse_transform(preds_array).flatten()

        freq = self._infer_freq()
        last_date = self._train_index[-1]
        future_index = pd.date_range(
            start=last_date, periods=steps + 1, freq=freq
        )[1:]

        return pd.Series(inv_preds, index=future_index, name="forecast")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_sequences(
        self, scaled: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build sliding-window (X, y) pairs.

        X shape: (n, look_back, 1)
        y shape: (n, horizon)
        """
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        end = len(scaled) - self.horizon

        for i in range(self.look_back, end + 1):
            X_list.append(scaled[i - self.look_back : i])
            y_list.append(scaled[i : i + self.horizon])

        if not X_list:
            return np.empty((0, self.look_back, 1)), np.empty((0, self.horizon))

        X = np.array(X_list, dtype=np.float32).reshape(-1, self.look_back, 1)
        y = np.array(y_list, dtype=np.float32)
        return X, y

    def _train_loop(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> None:
        """Run training with early stopping."""
        X_tr = torch.tensor(X_train)
        y_tr = torch.tensor(y_train)
        X_v = torch.tensor(X_val)
        y_v = torch.tensor(y_val)

        best_val_loss = float("inf")
        best_state: dict = {}
        no_improve = 0

        n_samples = X_tr.shape[0]

        for epoch in range(self.max_epochs):
            self._model.train()  # type: ignore[union-attr]
            # Mini-batch training
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                xb = X_tr[batch_idx]
                yb = y_tr[batch_idx]
                optimizer.zero_grad()
                pred = self._model(xb)  # type: ignore[misc]
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            # Validation
            self._model.eval()  # type: ignore[union-attr]
            with torch.no_grad():
                val_pred = self._model(X_v)  # type: ignore[misc]
                val_loss = criterion(val_pred, y_v).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {
                    k: v.clone() for k, v in self._model.state_dict().items()  # type: ignore[union-attr]
                }
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

        if best_state:
            self._model.load_state_dict(best_state)  # type: ignore[union-attr]

    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def _infer_freq(self) -> str:
        """Infer DateOffset string from training index."""
        if len(self._train_index) < 2:
            return "D"
        delta = self._train_index[1] - self._train_index[0]
        days = delta.days
        if days == 1:
            return "D"
        if days == 7:
            return "W"
        if 28 <= days <= 31:
            return "ME"
        if days == 0:
            return "h"
        return f"{days}D"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    values = pd.Series(
        50 + np.cumsum(np.random.randn(200) * 0.5),
        index=dates,
    )

    forecaster = LSTMForecaster(
        look_back=10, max_epochs=5, horizon=1, patience=3
    )
    forecaster.fit(values)
    result = forecaster.forecast(steps=7)

    assert isinstance(result, pd.Series), "forecast must return pd.Series"
    assert len(result) == 7, f"Expected 7 steps, got {len(result)}"
    assert result.index.dtype == "datetime64[ns]", "DatetimeIndex required"

    data_min, data_max = values.min(), values.max()
    margin = (data_max - data_min) * 0.5
    assert result.min() > data_min - margin, f"Forecast too low: {result.min()}"
    assert result.max() < data_max + margin, f"Forecast too high: {result.max()}"

    print(f"Forecast (7 steps): {result.values.round(2)}")
    print("Smoke test passed.")
    sys.exit(0)
