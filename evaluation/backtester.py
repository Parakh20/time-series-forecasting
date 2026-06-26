"""
Backtesting framework for time series models.

Implements walk-forward validation and out-of-sample evaluation.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    from evaluation.metrics import mae, mape, mase, rmse, smape
except ModuleNotFoundError:
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).parent.parent))
    from evaluation.metrics import mae, mape, mase, rmse, smape

RANDOM_SEED = 42


class WalkForwardBacktester:
    """Rolling-origin (expanding-window) walk-forward backtester.

    Starting from an initial training window of 70% of the series, the
    backtester forecasts the next ``test_size_pct`` fraction of data,
    then expands the training set by that window and repeats.

    Args:
        n_splits: Minimum number of folds to produce.
        test_size_pct: Fraction of total series length used as each test window.
    """

    def __init__(self, n_splits: int = 3, test_size_pct: float = 0.10) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not 0.0 < test_size_pct < 1.0:
            raise ValueError("test_size_pct must be in (0, 1)")
        self.n_splits = n_splits
        self.test_size_pct = test_size_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        series: pd.Series,
        forecaster: Any,
        freq: str = "D",
    ) -> pd.DataFrame:
        """Run walk-forward evaluation.

        Args:
            series: Time series with a DatetimeIndex.
            forecaster: Any object with ``fit()`` and ``forecast()`` methods.
            freq: Pandas frequency string passed to Prophet-based forecasters.

        Returns:
            DataFrame with columns: fold, train_size, test_size,
            mape, smape, rmse, mae, mase.  One row per fold.
        """
        n = len(series)
        test_size = max(int(n * self.test_size_pct), 10)
        initial_train_size = int(n * 0.70)

        records = []
        for fold in range(self.n_splits):
            train_end = initial_train_size + fold * test_size
            test_end = train_end + test_size
            if test_end > n:
                break

            train = series.iloc[:train_end]
            test = series.iloc[train_end:test_end]
            steps = len(test)

            try:
                self._fit_forecaster(forecaster, train, freq)
                forecast_values = self._get_forecast_values(forecaster, steps, freq)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Fold {fold}: forecaster raised {type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
                continue

            actual = test.values.astype(float)
            predicted = np.asarray(forecast_values, dtype=float)

            # Trim or pad to match length (safety guard)
            if len(predicted) != steps:
                warnings.warn(
                    f"Fold {fold}: forecaster returned {len(predicted)} steps "
                    f"but test window has {steps}. Truncating to {min(len(predicted), steps)}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]

            records.append(
                {
                    "fold": fold,
                    "train_size": len(train),
                    "test_size": min_len,
                    "mape": mape(actual, predicted),
                    "smape": smape(actual, predicted),
                    "rmse": rmse(actual, predicted),
                    "mae": mae(actual, predicted),
                    "mase": mase(actual, predicted, seasonal_period=1),
                }
            )

        return pd.DataFrame(
            records,
            columns=["fold", "train_size", "test_size", "mape", "smape", "rmse", "mae", "mase"],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_forecaster(self, forecaster: Any, train: pd.Series, freq: str) -> None:
        """Fit a forecaster, handling different fit() signatures."""
        # EnsembleForecaster: fit(train, val, freq=...)
        if hasattr(forecaster, "_arima") and hasattr(forecaster, "_prophet"):
            val_size = max(int(len(train) * 0.10), 1)
            train_part = train.iloc[:-val_size]
            val_part = train.iloc[-val_size:]
            forecaster.fit(train_part, val_part, freq=freq)
            return

        # ProphetForecaster: fit(df_ds_y)
        if _is_prophet(forecaster):
            df = pd.DataFrame({"ds": train.index, "y": train.values})
            forecaster.fit(df)
            return

        # ARIMA / LSTM: fit(pd.Series)
        forecaster.fit(train)

    def _get_forecast_values(
        self, forecaster: Any, steps: int, freq: str
    ) -> np.ndarray:
        """Return a 1-D numpy array of ``steps`` forecast values."""
        # ProphetForecaster: forecast(periods, freq) -> DataFrame with 'yhat'
        if _is_prophet(forecaster):
            result = forecaster.forecast(steps, freq)
            if isinstance(result, pd.DataFrame) and "yhat" in result.columns:
                return result["yhat"].values[-steps:]
            return np.asarray(result, dtype=float).ravel()[:steps]

        # EnsembleForecaster: forecast(steps) -> pd.Series
        if hasattr(forecaster, "_arima") and hasattr(forecaster, "_prophet"):
            result = forecaster.forecast(steps)
            return np.asarray(result, dtype=float).ravel()[:steps]

        # ARIMAForecaster: forecast(steps) -> dict with 'mean' key
        # LSTMForecaster: forecast(steps) -> pd.Series
        result = forecaster.forecast(steps)
        if isinstance(result, dict):
            arr = result.get("mean", next(iter(result.values())))
            return np.asarray(arr, dtype=float).ravel()[:steps]
        return np.asarray(result, dtype=float).ravel()[:steps]


def _is_prophet(forecaster: Any) -> bool:
    """Return True if the forecaster wraps Prophet (duck-type check)."""
    cls_name = type(forecaster).__name__
    return "prophet" in cls_name.lower() or hasattr(forecaster, "_model") and (
        hasattr(forecaster, "_freq") or "Prophet" in cls_name
    )


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    np.random.seed(42)
    dates = pd.date_range("2018-01-01", periods=500, freq="D")
    series = pd.Series(
        50 + np.cumsum(np.random.randn(500) * 0.3),
        index=dates,
    )

    from models.arima import ARIMAForecaster  # noqa: PLC0415

    backtester = WalkForwardBacktester(n_splits=3, test_size_pct=0.10)
    results = backtester.run(series, ARIMAForecaster(seasonal=False))

    assert len(results) >= 3, f"Expected >=3 folds, got {len(results)}"
    assert "mape" in results.columns
    assert "fold" in results.columns

    print(results[["fold", "train_size", "test_size", "mape", "mase"]].to_string())
    print("Backtester smoke test passed.")
