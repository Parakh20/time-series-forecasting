"""
Prophet model for time series forecasting.

Implements Facebook Prophet for time series forecasting with automatic seasonality detection.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RANDOM_SEED = 42

# Ensure results directory exists
_RESULTS_DIR = Path(__file__).parent.parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)


class ProphetForecaster:
    """Facebook Prophet forecaster with CV-based changepoint tuning.

    Args:
        seasonality_mode: "additive" (default) or "multiplicative". Use
            "multiplicative" for series with variance proportional to level
            (e.g. commodities).
        changepoint_prior_scale: Flexibility of the trend changepoint prior.
            Larger values allow more flexible trend changes. Default 0.05.
        add_weekly_seasonality: Whether to include a weekly Fourier component.
        add_annual_seasonality: Whether to include a yearly Fourier component.
        add_us_holidays: Whether to add US holiday regressors.
        random_seed: Random seed set via numpy.random.seed before Prophet fitting.
    """

    def __init__(
        self,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        add_weekly_seasonality: bool = True,
        add_annual_seasonality: bool = True,
        add_us_holidays: bool = False,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        if seasonality_mode not in ("additive", "multiplicative"):
            raise ValueError(
                f"seasonality_mode must be 'additive' or 'multiplicative', got '{seasonality_mode}'"
            )
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.add_weekly_seasonality = add_weekly_seasonality
        self.add_annual_seasonality = add_annual_seasonality
        self.add_us_holidays = add_us_holidays
        self.random_seed = random_seed

        self._model = None
        self._train_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ProphetForecaster":
        """Fit Prophet on a DataFrame with columns 'ds' (datetime) and 'y' (float).

        Args:
            df: Training data with columns 'ds' and 'y'.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If df is missing required columns or is empty.
        """
        self._validate_df(df)
        self._train_df = df.copy()

        model = self._build_model(self.changepoint_prior_scale)

        np.random.seed(self.random_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df[["ds", "y"]])

        self._model = model
        return self

    def forecast(self, periods: int, freq: str = "D") -> pd.DataFrame:
        """Generate a forecast for the given number of future periods.

        Args:
            periods: Number of future periods to forecast.
            freq: Pandas frequency string for the forecast horizon (e.g. "D", "B").

        Returns:
            Prophet forecast DataFrame containing at minimum:
            yhat, yhat_lower, yhat_upper columns.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If periods is not a positive integer.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before forecasting. Call fit() first.")
        if periods <= 0:
            raise ValueError(f"periods must be a positive integer, got {periods}.")

        future = self._model.make_future_dataframe(periods=periods, freq=freq)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast_df = self._model.predict(future)

        return forecast_df

    def tune_changepoint_prior(
        self,
        df: pd.DataFrame,
        scales: Optional[List[float]] = None,
        cv_horizon: str = "90 days",
    ) -> float:
        """Select the best changepoint_prior_scale via Prophet cross-validation.

        Fits a separate Prophet model for each candidate scale value, runs
        time-series CV, and returns the scale that minimises mean MAPE.
        Call fit() after this method to apply the tuned scale to the model.

        Args:
            df: Training data with columns 'ds' and 'y'.
            scales: List of candidate changepoint_prior_scale values.
                Defaults to [0.001, 0.01, 0.05, 0.1, 0.5].
            cv_horizon: Horizon string passed to prophet.diagnostics.cross_validation,
                e.g. "90 days".

        Returns:
            Best changepoint_prior_scale (float) — the one with lowest mean MAPE.

        Raises:
            ValueError: If df is missing required columns or is empty.
            RuntimeError: If all CV iterations fail for all changepoint_prior_scale values.
        """
        from prophet.diagnostics import cross_validation, performance_metrics

        self._validate_df(df)

        if scales is None:
            scales = [0.001, 0.01, 0.05, 0.1, 0.5]

        # Use 70% of training length as CV initial period
        n_days = int((df["ds"].max() - df["ds"].min()).days * 0.70)
        initial_str = f"{max(n_days, 1)} days"

        best_scale = scales[0]
        best_mape = float("inf")

        for scale in scales:
            try:
                model = self._build_model(scale)

                np.random.seed(self.random_seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(df[["ds", "y"]])

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_df = cross_validation(
                        model,
                        initial=initial_str,
                        horizon=cv_horizon,
                        parallel=None,
                    )

                metrics_df = performance_metrics(cv_df, rolling_window=1)
                mean_mape = float(metrics_df["mape"].mean())

                if mean_mape < best_mape:
                    best_mape = mean_mape
                    best_scale = scale

            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"CV failed for changepoint_prior_scale={scale}: {exc}",
                    stacklevel=2,
                )

        if best_mape == float("inf"):
            raise RuntimeError(
                f"All CV iterations failed for all changepoint_prior_scale values. "
                f"Data may be too short for cv_horizon='{cv_horizon}'."
            )

        self.changepoint_prior_scale = best_scale
        return best_scale

    def plot_components(self, forecast_df: pd.DataFrame, name: str = "series") -> str:
        """Save a Prophet component plot to results/.

        Args:
            forecast_df: DataFrame returned by forecast().
            name: Label used in the output filename.

        Returns:
            Absolute path to the saved PNG file.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before plotting. Call fit() first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = self._model.plot_components(forecast_df)

        out_path = _RESULTS_DIR / f"prophet_components_{name}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return str(out_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self, changepoint_prior_scale: float):
        """Construct a configured Prophet model instance."""
        from prophet import Prophet

        model = Prophet(
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
        )

        if self.add_weekly_seasonality:
            model.add_seasonality(name="weekly", period=7, fourier_order=3)

        if self.add_annual_seasonality:
            model.add_seasonality(name="yearly", period=365.25, fourier_order=10)

        if self.add_us_holidays:
            model.add_country_holidays(country_name="US")

        return model

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        """Raise ValueError if df is missing required columns or is empty."""
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame must not be empty.")
        missing = {"ds", "y"} - set(df.columns)
        if missing:
            raise ValueError(f"Input DataFrame missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    try:
        np.random.seed(42)

        # Short synthetic series
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        values = (
            100
            + np.cumsum(np.random.randn(200))
            + 10 * np.sin(np.arange(200) * 2 * np.pi / 365)
        )
        train_df = pd.DataFrame({"ds": dates, "y": values})

        forecaster = ProphetForecaster(
            seasonality_mode="additive", changepoint_prior_scale=0.05
        )
        forecaster.fit(train_df)

        forecast_df = forecaster.forecast(periods=30)

        assert "yhat" in forecast_df.columns, "forecast_df must contain 'yhat'"
        assert "yhat_lower" in forecast_df.columns, "forecast_df must contain 'yhat_lower'"
        assert "yhat_upper" in forecast_df.columns, "forecast_df must contain 'yhat_upper'"
        assert len(forecast_df) >= 30, (
            f"Expected at least 30 rows in forecast_df, got {len(forecast_df)}"
        )

        plot_path = forecaster.plot_components(forecast_df, name="smoke_test")
        assert os.path.exists(plot_path), f"Component plot not found at {plot_path}"

        print(f"Forecast shape: {forecast_df.shape}")
        print(f"Component plot: {plot_path}")
        print("Smoke test passed.")
        sys.exit(0)

    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
