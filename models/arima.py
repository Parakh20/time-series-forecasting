"""
ARIMA model for time series forecasting.

Implements ARIMA and auto-ARIMA models using statsmodels and pmdarima.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

import pmdarima as pm

RANDOM_SEED = 42

# Ensure results directory exists
_RESULTS_DIR = Path(__file__).parent.parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)


class ARIMAForecaster:
    """Auto-ARIMA forecaster using pmdarima with AIC model selection.

    Args:
        seasonal: Whether to fit a seasonal ARIMA (SARIMA) model.
        m: Seasonal period (e.g. 12 for monthly, 52 for weekly).
        random_seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        seasonal: bool = False,
        m: int = 1,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.seasonal = seasonal
        self.m = m
        self.random_seed = random_seed
        self._model = None
        self._train_index = None

    def fit(self, train: pd.Series) -> "ARIMAForecaster":
        """Fit using pmdarima.auto_arima (AIC criterion).

        Args:
            train: Training time series with DatetimeIndex.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If train is empty.
        """
        if len(train) == 0:
            raise ValueError("Training series must not be empty.")

        self._train_index = train.index

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = pm.auto_arima(
                train,
                seasonal=self.seasonal,
                m=self.m,
                information_criterion="aic",
                error_action="ignore",
                suppress_warnings=True,
                random_state=self.random_seed,
            )

        return self

    def forecast(self, steps: int) -> dict:
        """Generate point forecast with 95% confidence interval.

        Args:
            steps: Number of steps ahead to forecast.

        Returns:
            Dict with keys:
                - "mean": pd.Series of point forecasts
                - "lower_95": pd.Series of lower CI bound
                - "upper_95": pd.Series of upper CI bound
            All series have a DatetimeIndex continuing from the training end.

        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before forecasting. Call fit() first.")

        mean_vals, conf_int = self._model.predict(n_periods=steps, return_conf_int=True)

        future_index = self._build_future_index(steps)

        return {
            "mean": pd.Series(mean_vals, index=future_index, name="forecast"),
            "lower_95": pd.Series(conf_int[:, 0], index=future_index, name="lower_95"),
            "upper_95": pd.Series(conf_int[:, 1], index=future_index, name="upper_95"),
        }

    def residual_diagnostics(self, name: str = "model") -> dict:
        """Compute residual diagnostics and save a diagnostic plot.

        Runs the Ljung-Box test (lag 10) on model residuals and produces a
        2-panel figure: residuals over time and ACF of residuals.

        Args:
            name: Label used in the output plot filename.

        Returns:
            Dict with keys:
                - "ljung_box_p": float p-value at lag 10
                - "residuals": pd.Series of model residuals
                - "plot_path": str path to the saved figure

        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before diagnostics. Call fit() first.")

        residuals = pd.Series(
            self._model.resid(),
            index=self._train_index[len(self._train_index) - len(self._model.resid()):],
            name="residuals",
        )

        lb_result = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)
        ljung_box_p = float(lb_result["lb_pvalue"].iloc[0])

        plot_path = self._save_residual_plot(residuals, name)

        return {
            "ljung_box_p": ljung_box_p,
            "residuals": residuals,
            "plot_path": plot_path,
        }

    @property
    def order(self) -> tuple:
        """Return the fitted ARIMA order.

        Returns:
            (p, d, q) for non-seasonal; (p, d, q)(P, D, Q, m) for seasonal.

        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before accessing order. Call fit() first.")

        pdq = self._model.order
        if self.seasonal and self._model.seasonal_order is not None:
            return pdq + self._model.seasonal_order
        return pdq

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_future_index(self, steps: int) -> pd.DatetimeIndex:
        """Infer frequency from training index and build a future DatetimeIndex."""
        if self._train_index is None or len(self._train_index) < 2:
            return pd.RangeIndex(steps)

        inferred_freq = pd.infer_freq(self._train_index)
        last_date = self._train_index[-1]

        if inferred_freq is not None:
            return pd.date_range(start=last_date, periods=steps + 1, freq=inferred_freq)[1:]

        # Fallback: use median gap between observations
        gaps = pd.Series(self._train_index).diff().dropna()
        median_gap = gaps.median()
        future_dates = [last_date + median_gap * i for i in range(1, steps + 1)]
        return pd.DatetimeIndex(future_dates)

    def _save_residual_plot(self, residuals: pd.Series, name: str) -> str:
        """Save a 2-panel residual diagnostic plot to results/.

        Panel 1: Residuals over time.
        Panel 2: ACF of residuals.

        Returns:
            Absolute path to the saved figure.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        fig.suptitle(f"ARIMA Residual Diagnostics — {name}", fontsize=13)

        # Panel 1: residuals over time
        axes[0].plot(residuals.values, linewidth=0.8, color="steelblue")
        axes[0].axhline(0, color="red", linewidth=0.8, linestyle="--")
        axes[0].set_title("Residuals over Time")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Residual")

        # Panel 2: ACF of residuals
        plot_acf(residuals.dropna(), ax=axes[1], lags=30, zero=False)
        axes[1].set_title("ACF of Residuals")

        plt.tight_layout()
        out_path = _RESULTS_DIR / f"arima_residuals_{name}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return str(out_path)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    try:
        # Test on a short synthetic series (fast)
        np.random.seed(RANDOM_SEED)
        short_series = pd.Series(
            100 + np.cumsum(np.random.randn(100)),
            index=pd.date_range("2020-01-01", periods=100, freq="D"),
        )

        forecaster = ARIMAForecaster(seasonal=False)
        forecaster.fit(short_series)

        result = forecaster.forecast(steps=10)
        assert "mean" in result and "lower_95" in result and "upper_95" in result, (
            "forecast() must return dict with mean, lower_95, upper_95"
        )
        assert len(result["mean"]) == 10, f"Expected 10 forecast steps, got {len(result['mean'])}"
        assert isinstance(result["mean"].index, pd.DatetimeIndex), (
            "Forecast index must be DatetimeIndex"
        )
        assert result["mean"].index[0] > short_series.index[-1], (
            "Forecast dates must follow training dates"
        )

        diag = forecaster.residual_diagnostics(name="synthetic")
        assert "ljung_box_p" in diag, "residual_diagnostics() must return ljung_box_p"
        assert 0 <= diag["ljung_box_p"] <= 1, (
            f"Ljung-Box p-value out of [0,1]: {diag['ljung_box_p']}"
        )
        assert Path(diag["plot_path"]).exists(), (
            f"Residual plot not found at {diag['plot_path']}"
        )

        print(f"Order selected: {forecaster.order}")
        print(f"Ljung-Box p-value: {diag['ljung_box_p']:.4f}")
        print(f"Forecast range: {result['mean'].index[0].date()} — {result['mean'].index[-1].date()}")
        print(f"Residual plot saved to: {diag['plot_path']}")
        print("Smoke test passed.")
        sys.exit(0)

    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
