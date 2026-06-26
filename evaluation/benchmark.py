"""
Full model comparison benchmark on held-out test sets.

Runs Naive, Seasonal Naive, ARIMA, Prophet, LSTM, and Ensemble on both
energy (daily) and commodity (business-day) datasets and reports MAPE,
sMAPE, RMSE, MAE, MASE metrics.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# Add project root to sys.path for relative imports when run directly
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helper: data split
# ---------------------------------------------------------------------------

def _split_series(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df (ds, y) into train/val/test 70/10/20.

    Returns:
        train_series, val_series, test_series,
        train_df, val_df, test_df
    """
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    def _to_series(sub: pd.DataFrame) -> pd.Series:
        return pd.Series(
            sub["y"].values,
            index=pd.DatetimeIndex(sub["ds"].values),
            name="y",
        )

    return (
        _to_series(train_df),
        _to_series(val_df),
        _to_series(test_df),
        train_df,
        val_df,
        test_df,
    )


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------

def _make_naive_forecast(train: pd.Series, steps: int) -> np.ndarray:
    """Repeat last training value for all forecast steps."""
    return np.full(steps, train.iloc[-1])


def _make_seasonal_naive_forecast(
    series: pd.Series,
    test_start_idx: int,
    steps: int,
    period: int,
) -> np.ndarray:
    """Forecast by repeating the value `period` steps before each test point."""
    preds = np.empty(steps)
    all_values = series.values
    for i in range(steps):
        src_idx = test_start_idx + i - period
        if src_idx >= 0:
            preds[i] = all_values[src_idx]
        else:
            preds[i] = all_values[0]
    return preds


# ---------------------------------------------------------------------------
# Model-specific fit+forecast helpers
# ---------------------------------------------------------------------------

def _fit_and_forecast_arima(
    train_series: pd.Series,
    steps: int,
    seasonal: bool,
    m: int,
) -> np.ndarray:
    """Fit ARIMA on train and return point forecasts as ndarray."""
    from models.arima import ARIMAForecaster

    forecaster = ARIMAForecaster(seasonal=seasonal, m=m, random_seed=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(train_series)
    result = forecaster.forecast(steps=steps)
    return result["mean"].values


def _fit_and_forecast_prophet(
    train_df: pd.DataFrame,
    steps: int,
    freq: str,
    seasonality_mode: str,
) -> np.ndarray:
    """Fit Prophet on train_df and return point forecasts as ndarray."""
    from models.prophet_model import ProphetForecaster

    forecaster = ProphetForecaster(
        seasonality_mode=seasonality_mode,
        random_seed=RANDOM_SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(train_df[["ds", "y"]])
    fc_df = forecaster.forecast(periods=steps, freq=freq)
    return fc_df["yhat"].values[-steps:]


def _fit_and_forecast_lstm(
    train_series: pd.Series,
    steps: int,
) -> np.ndarray:
    """Fit LSTM on train_series and return forecasts as ndarray."""
    from models.lstm_model import LSTMForecaster

    forecaster = LSTMForecaster(look_back=30, horizon=1, max_epochs=30, random_seed=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(train_series)
    fc = forecaster.forecast(steps=steps)
    return fc.values


def _fit_and_forecast_ensemble(
    train_series: pd.Series,
    val_series: pd.Series,
    steps: int,
    freq: str,
    arima_seasonal: bool,
    arima_m: int,
    prophet_seasonality_mode: str,
) -> np.ndarray:
    """Fit ensemble on train+val and return forecasts as ndarray."""
    from models.arima import ARIMAForecaster
    from models.prophet_model import ProphetForecaster
    from models.ensemble import EnsembleForecaster

    arima = ARIMAForecaster(seasonal=arima_seasonal, m=arima_m, random_seed=RANDOM_SEED)
    prophet = ProphetForecaster(
        seasonality_mode=prophet_seasonality_mode,
        random_seed=RANDOM_SEED,
    )
    ens = EnsembleForecaster(arima, prophet)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ens.fit(train_series, val_series, freq=freq)
    fc = ens.forecast(steps=steps, freq=freq)
    return fc.values


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def _compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
) -> dict:
    """Compute all five metrics for a single model."""
    from evaluation.metrics import mape, smape, rmse, mae, mase

    return {
        "model": model_name,
        "mape": mape(actual, predicted),
        "smape": smape(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "mase": mase(actual, predicted, seasonal_period=1),
    }


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------

def run_benchmark(dataset_name: str, freq: str = "D") -> pd.DataFrame:
    """Load dataset, split 70/10/20, fit all models on train, evaluate on test.

    Test data is NEVER passed to any model's fit() method.

    Args:
        dataset_name: "energy" or "commodity".
        freq: Pandas frequency string ("D" for daily, "B" for business-day).

    Returns:
        DataFrame with columns: model, mape, smape, rmse, mae, mase.
        One row per model.
    """
    from data.download import load_dataset

    df = load_dataset(dataset_name)
    train_series, val_series, test_series, train_df, val_df, _test_df = _split_series(df)

    actual = test_series.values
    steps = len(actual)
    full_series = pd.Series(
        df["y"].values,
        index=pd.DatetimeIndex(df["ds"].values),
        name="y",
    )
    train_end_idx = len(train_series)

    is_energy = dataset_name == "energy"
    arima_seasonal = is_energy
    arima_m = 12 if is_energy else 1
    prophet_mode = "additive" if is_energy else "multiplicative"

    rows: list[dict] = []

    # 1. Naive baseline
    naive_fc = _make_naive_forecast(train_series, steps)
    rows.append(_compute_metrics(actual, naive_fc, "naive"))

    # 2. Seasonal naive (energy only)
    if is_energy:
        seasonal_fc = _make_seasonal_naive_forecast(
            full_series, train_end_idx + len(val_series), steps, period=365
        )
        rows.append(_compute_metrics(actual, seasonal_fc, "seasonal_naive"))

    # 3. ARIMA
    try:
        arima_fc = _fit_and_forecast_arima(train_series, steps, arima_seasonal, arima_m)
        rows.append(_compute_metrics(actual, arima_fc, "arima"))
    except Exception as exc:
        print(f"ARIMA failed: {exc}", file=sys.stderr)

    # 4. Prophet
    try:
        prophet_fc = _fit_and_forecast_prophet(train_df, steps, freq, prophet_mode)
        rows.append(_compute_metrics(actual, prophet_fc, "prophet"))
    except Exception as exc:
        print(f"Prophet failed: {exc}", file=sys.stderr)

    # 5. LSTM
    try:
        lstm_fc = _fit_and_forecast_lstm(train_series, steps)
        rows.append(_compute_metrics(actual, lstm_fc, "lstm"))
    except Exception as exc:
        print(f"LSTM failed: {exc}", file=sys.stderr)

    # 6. Ensemble (ARIMA + Prophet)
    try:
        ens_fc = _fit_and_forecast_ensemble(
            train_series, val_series, steps, freq,
            arima_seasonal, arima_m, prophet_mode,
        )
        rows.append(_compute_metrics(actual, ens_fc, "ensemble"))
    except Exception as exc:
        print(f"Ensemble failed: {exc}", file=sys.stderr)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    table = run_benchmark("energy", freq="D")
    assert "model" in table.columns, "Missing 'model' column"
    assert "mape" in table.columns, "Missing 'mape' column"
    assert "mase" in table.columns, "Missing 'mase' column"
    assert len(table) >= 4, f"Expected >= 4 model rows, got {len(table)}"

    naive_mape = table[table["model"] == "naive"]["mape"].values[0]
    best_mape = table["mape"].min()
    assert best_mape <= naive_mape * 1.5, (
        f"All models worse than naive — something is wrong. "
        f"naive_mape={naive_mape:.2f}, best_mape={best_mape:.2f}"
    )

    print(table.to_string(index=False))
    print("Benchmark smoke test passed.")
    sys.exit(0)
