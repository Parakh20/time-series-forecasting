"""
Ensemble model for time series forecasting.

Combines ARIMA and Prophet predictions via learned non-negative weights
optimised on a validation set.
"""

import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import nnls

RANDOM_SEED = 42


class EnsembleForecaster:
    """Weighted average ensemble of ARIMAForecaster and ProphetForecaster.

    Weights are learned by non-negative least squares (NNLS) on a held-out
    validation set, then normalised to sum to 1.

    Args:
        arima_forecaster: A fitted or unfitted ARIMAForecaster instance.
        prophet_forecaster: A fitted or unfitted ProphetForecaster instance.
    """

    def __init__(self, arima_forecaster, prophet_forecaster) -> None:
        self._arima = arima_forecaster
        self._prophet = prophet_forecaster
        self._weights = None  # (w_arima, w_prophet) set after fit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train: pd.Series, val: pd.Series) -> "EnsembleForecaster":
        """Fit both component models on train, learn weights on val.

        Args:
            train: Training time series with DatetimeIndex.
            val: Validation time series with DatetimeIndex (immediately
                follows train).

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If train or val is empty.
        """
        if len(train) == 0:
            raise ValueError("train series must not be empty.")
        if len(val) == 0:
            raise ValueError("val series must not be empty.")

        self._fit_components(train)
        val_preds = self._component_val_forecasts(len(val))
        self._weights = self._learn_weights(val.values, val_preds)
        return self

    def forecast(self, steps: int, freq: str = "D") -> pd.Series:
        """Produce a weighted average forecast.

        Args:
            steps: Number of future steps to forecast.
            freq: Pandas frequency string (used by ProphetForecaster).

        Returns:
            pd.Series with DatetimeIndex of length `steps`.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._weights is None:
            raise RuntimeError("EnsembleForecaster must be fitted before forecasting. Call fit() first.")

        w_arima, w_prophet = self._weights
        arima_fc = self._arima_forecast_series(steps)
        prophet_fc = self._prophet_forecast_series(steps, freq)

        # Align on arima's index (both start from same date)
        combined = w_arima * arima_fc.values + w_prophet * prophet_fc.values
        return pd.Series(combined, index=arima_fc.index, name="ensemble_forecast")

    @property
    def weights(self) -> dict:
        """Return the learned component weights.

        Returns:
            {"arima": float, "prophet": float}

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._weights is None:
            raise RuntimeError("Weights are not available until fit() is called.")
        w_arima, w_prophet = self._weights
        return {"arima": float(w_arima), "prophet": float(w_prophet)}

    def compare_metrics(self, val: pd.Series, metric_fn: Callable) -> dict:
        """Compare component and ensemble metrics on a validation series.

        Args:
            val: Validation time series (actual values).
            metric_fn: Function with signature metric_fn(actual, predicted) -> float.

        Returns:
            {"arima": float, "prophet": float, "ensemble": float}

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._weights is None:
            raise RuntimeError("EnsembleForecaster must be fitted before comparing metrics.")

        steps = len(val)
        val_preds = self._component_val_forecasts(steps)
        arima_pred, prophet_pred = val_preds[:, 0], val_preds[:, 1]

        w_arima, w_prophet = self._weights
        ensemble_pred = w_arima * arima_pred + w_prophet * prophet_pred

        actual = val.values
        return {
            "arima": float(metric_fn(actual, arima_pred)),
            "prophet": float(metric_fn(actual, prophet_pred)),
            "ensemble": float(metric_fn(actual, ensemble_pred)),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_components(self, train: pd.Series) -> None:
        """Fit ARIMA and Prophet on the same training series."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._arima.fit(train)

        train_df = pd.DataFrame({"ds": train.index, "y": train.values})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._prophet.fit(train_df)

    def _arima_forecast_series(self, steps: int) -> pd.Series:
        """Return ARIMA mean forecast as a pd.Series."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._arima.forecast(steps)
        return result["mean"]

    def _prophet_forecast_series(self, steps: int, freq: str = "D") -> pd.Series:
        """Return Prophet yhat forecast as a pd.Series aligned with ARIMA index."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc_df = self._prophet.forecast(periods=steps, freq=freq)

        # make_future_dataframe includes training rows; take the last `steps` rows
        future_rows = fc_df.tail(steps)

        # Build index from the last `steps` rows of Prophet's ds column
        prophet_index = pd.DatetimeIndex(future_rows["ds"].values)
        return pd.Series(future_rows["yhat"].values, index=prophet_index, name="prophet_forecast")

    def _component_val_forecasts(self, steps: int) -> np.ndarray:
        """Return an (steps, 2) array of [arima_pred, prophet_pred]."""
        arima_vals = self._arima_forecast_series(steps).values
        prophet_vals = self._prophet_forecast_series(steps).values
        return np.column_stack([arima_vals, prophet_vals])

    @staticmethod
    def _learn_weights(actual: np.ndarray, preds: np.ndarray) -> tuple:
        """Learn non-negative weights via NNLS, normalised to sum to 1.

        Args:
            actual: 1-D array of actual values (length n).
            preds: (n, 2) array of [arima_pred, prophet_pred].

        Returns:
            (w_arima, w_prophet) tuple summing to 1.
        """
        weights, _ = nnls(preds, actual)
        total = weights.sum()
        if total == 0:
            # Fallback to equal weights if NNLS collapses
            return (0.5, 0.5)
        normalised = weights / total
        return (float(normalised[0]), float(normalised[1]))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from models.arima import ARIMAForecaster
    from models.prophet_model import ProphetForecaster

    try:
        np.random.seed(RANDOM_SEED)

        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        values = pd.Series(
            100
            + np.cumsum(np.random.randn(300) * 0.5)
            + 5 * np.sin(np.arange(300) * 2 * np.pi / 365),
            index=dates,
        )
        train = values[:240]
        val = values[240:270]

        arima = ARIMAForecaster(seasonal=False)
        prophet = ProphetForecaster(seasonality_mode="additive")

        ensemble = EnsembleForecaster(arima, prophet)
        ensemble.fit(train, val)

        weights = ensemble.weights
        assert "arima" in weights and "prophet" in weights, "weights dict must have 'arima' and 'prophet'"
        assert abs(weights["arima"] + weights["prophet"] - 1.0) < 1e-6, (
            f"Weights must sum to 1, got {weights['arima'] + weights['prophet']}"
        )
        assert weights["arima"] >= 0 and weights["prophet"] >= 0, "Weights must be non-negative"

        forecast = ensemble.forecast(steps=10)
        assert isinstance(forecast, pd.Series), "forecast() must return pd.Series"
        assert len(forecast) == 10, f"Expected 10 steps, got {len(forecast)}"
        assert isinstance(forecast.index, pd.DatetimeIndex), "Forecast index must be DatetimeIndex"

        def mse(actual, pred):
            return float(np.mean((actual - pred) ** 2))

        metrics = ensemble.compare_metrics(val, mse)
        assert set(metrics.keys()) == {"arima", "prophet", "ensemble"}, (
            f"compare_metrics must return arima, prophet, ensemble keys; got {set(metrics.keys())}"
        )
        assert all(v >= 0 for v in metrics.values()), "MSE values must be non-negative"

        print(f"Weights: arima={weights['arima']:.3f}, prophet={weights['prophet']:.3f}")
        print(f"Forecast (10 steps): {forecast.values.round(2)}")
        print(
            f"Val MSE — arima={metrics['arima']:.4f}, "
            f"prophet={metrics['prophet']:.4f}, "
            f"ensemble={metrics['ensemble']:.4f}"
        )
        print("Smoke test passed.")
        sys.exit(0)

    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
