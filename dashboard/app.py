"""
Streamlit dashboard application for time series forecasting.

Interactive UI for model exploration, comparison, and visualization.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.download import load_dataset  # noqa: E402
from evaluation.metrics import mae, mape, mase, rmse, smape  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_MODELS = ["Naive", "ARIMA", "Prophet", "LSTM", "Ensemble"]
HORIZON_OPTIONS = [7, 30, 90]
DATASET_OPTIONS = ["Energy", "Commodity"]
_PERIOD_MAP = {"Energy": 365, "Commodity": 252}
_FREQ_MAP = {"Energy": "D", "Commodity": "B"}


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset…")
def _load_data(dataset_name: str) -> pd.DataFrame:
    """Load and cache dataset by name (lowercase)."""
    return load_dataset(dataset_name.lower())


# ---------------------------------------------------------------------------
# Model fitting helpers (cached by key tuple)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _fit_arima(dataset_name: str, train_pct: int) -> object:
    from models.arima import ARIMAForecaster
    df = _load_data(dataset_name)
    train = _split_series(df, train_pct)
    forecaster = ARIMAForecaster(seasonal=False, random_seed=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(train)
    return forecaster


@st.cache_resource(show_spinner=False)
def _fit_prophet(dataset_name: str, train_pct: int) -> object:
    from models.prophet_model import ProphetForecaster
    df = _load_data(dataset_name)
    train = _split_series(df, train_pct)
    train_df = pd.DataFrame({"ds": train.index, "y": train.values})
    forecaster = ProphetForecaster(random_seed=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(train_df)
    return forecaster


@st.cache_resource(show_spinner=False)
def _fit_lstm(dataset_name: str, train_pct: int) -> object:
    from models.lstm_model import LSTMForecaster
    df = _load_data(dataset_name)
    train = _split_series(df, train_pct)
    forecaster = LSTMForecaster(
        look_back=30, max_epochs=10, patience=5, random_seed=RANDOM_SEED
    )
    forecaster.fit(train)
    return forecaster


@st.cache_resource(show_spinner=False)
def _fit_ensemble(dataset_name: str, train_pct: int) -> object:
    from models.arima import ARIMAForecaster
    from models.ensemble import EnsembleForecaster
    from models.prophet_model import ProphetForecaster
    df = _load_data(dataset_name)
    train = _split_series(df, train_pct)
    # Use last 10% of train as validation for ensemble weight learning
    n = len(train)
    val_start = int(n * 0.90)
    train_part = train.iloc[:val_start]
    val_part = train.iloc[val_start:]
    arima = ARIMAForecaster(seasonal=False, random_seed=RANDOM_SEED)
    prophet = ProphetForecaster(random_seed=RANDOM_SEED)
    freq = _FREQ_MAP.get(dataset_name, "D")
    ensemble = EnsembleForecaster(arima, prophet)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble.fit(train_part, val_part, freq=freq)
    return ensemble


# ---------------------------------------------------------------------------
# Forecast generation helpers (cached by key tuple)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _forecast_naive(dataset_name: str, train_pct: int, horizon: int) -> pd.Series:
    df = _load_data(dataset_name)
    train = _split_series(df, train_pct)
    last_val = float(train.iloc[-1])
    freq = _FREQ_MAP.get(dataset_name, "D")
    future_idx = pd.date_range(train.index[-1], periods=horizon + 1, freq=freq)[1:]
    return pd.Series(np.full(horizon, last_val), index=future_idx, name="Naive")


@st.cache_data(show_spinner=False)
def _forecast_arima(dataset_name: str, train_pct: int, horizon: int) -> dict:
    forecaster = _fit_arima(dataset_name, train_pct)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return forecaster.forecast(horizon)


@st.cache_data(show_spinner=False)
def _forecast_prophet(dataset_name: str, train_pct: int, horizon: int) -> pd.DataFrame:
    forecaster = _fit_prophet(dataset_name, train_pct)
    freq = _FREQ_MAP.get(dataset_name, "D")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc_df = forecaster.forecast(periods=horizon, freq=freq)
    return fc_df.tail(horizon)


@st.cache_data(show_spinner=False)
def _forecast_lstm(dataset_name: str, train_pct: int, horizon: int) -> pd.Series:
    forecaster = _fit_lstm(dataset_name, train_pct)
    return forecaster.forecast(steps=horizon)


@st.cache_data(show_spinner=False)
def _forecast_ensemble(dataset_name: str, train_pct: int, horizon: int) -> pd.Series:
    forecaster = _fit_ensemble(dataset_name, train_pct)
    freq = _FREQ_MAP.get(dataset_name, "D")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return forecaster.forecast(steps=horizon, freq=freq)


# ---------------------------------------------------------------------------
# Walk-forward helpers (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _run_walkforward(dataset_name: str, model_name: str) -> pd.DataFrame:
    from evaluation.backtester import WalkForwardBacktester
    df = _load_data(dataset_name)
    series = df.set_index("ds")["y"]
    backtester = WalkForwardBacktester(n_splits=3, test_size_pct=0.10)
    forecaster = _make_fresh_forecaster(model_name)
    if forecaster is None:
        return pd.DataFrame()
    freq = _FREQ_MAP.get(dataset_name, "D")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return backtester.run(series, forecaster, freq=freq)


def _make_fresh_forecaster(model_name: str):
    """Instantiate a fresh (unfitted) forecaster for walk-forward."""
    if model_name == "Naive":
        return None  # Naive not supported in backtester directly
    if model_name == "ARIMA":
        from models.arima import ARIMAForecaster
        return ARIMAForecaster(seasonal=False, random_seed=RANDOM_SEED)
    if model_name == "Prophet":
        from models.prophet_model import ProphetForecaster
        return ProphetForecaster(random_seed=RANDOM_SEED)
    if model_name == "LSTM":
        from models.lstm_model import LSTMForecaster
        return LSTMForecaster(look_back=30, max_epochs=10, patience=5, random_seed=RANDOM_SEED)
    if model_name == "Ensemble":
        from models.arima import ARIMAForecaster
        from models.ensemble import EnsembleForecaster
        from models.prophet_model import ProphetForecaster
        return EnsembleForecaster(
            ARIMAForecaster(seasonal=False, random_seed=RANDOM_SEED),
            ProphetForecaster(random_seed=RANDOM_SEED),
        )
    return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _split_series(df: pd.DataFrame, train_pct: int) -> pd.Series:
    """Return training portion as a DatetimeIndex Series."""
    series = df.set_index("ds")["y"]
    n = len(series)
    cut = int(n * train_pct / 100)
    return series.iloc[:cut]


def _test_series(df: pd.DataFrame, train_pct: int) -> pd.Series:
    series = df.set_index("ds")["y"]
    n = len(series)
    cut = int(n * train_pct / 100)
    return series.iloc[cut:]


def _compute_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    return {
        "MAPE": mape(actual, predicted),
        "sMAPE": smape(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAE": mae(actual, predicted),
        "MASE": mase(actual, predicted),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _build_forecast_fig(
    df: pd.DataFrame,
    train_pct: int,
    selected_models: list[str],
    horizon: int,
    dataset_name: str,
) -> plt.Figure:
    series = df.set_index("ds")["y"]
    n = len(series)
    cut = int(n * train_pct / 100)
    history = series.iloc[max(0, cut - 180): cut]  # show last 180 hist points
    test_actual = series.iloc[cut: cut + horizon]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history.index, history.values, color="steelblue", linewidth=1.2, label="History")
    if len(test_actual):
        ax.plot(
            test_actual.index,
            test_actual.values,
            color="black",
            linewidth=1.0,
            linestyle="--",
            label="Actual (held-out)",
        )

    colors = {"Naive": "gray", "ARIMA": "darkorange", "Prophet": "green",
              "LSTM": "purple", "Ensemble": "red"}

    for model_name in selected_models:
        color = colors.get(model_name, "blue")
        try:
            if model_name == "Naive":
                fc = _forecast_naive(dataset_name, train_pct, horizon)
                ax.plot(fc.index, fc.values, color=color, linewidth=1.5, label="Naive")

            elif model_name == "ARIMA":
                result = _forecast_arima(dataset_name, train_pct, horizon)
                mean_fc = result["mean"]
                ax.plot(mean_fc.index, mean_fc.values, color=color, linewidth=1.5, label="ARIMA")
                ax.fill_between(
                    mean_fc.index,
                    result["lower_95"].values,
                    result["upper_95"].values,
                    color=color, alpha=0.15, label="ARIMA 95% CI",
                )

            elif model_name == "Prophet":
                fc_df = _forecast_prophet(dataset_name, train_pct, horizon)
                idx = pd.DatetimeIndex(fc_df["ds"].values)
                ax.plot(idx, fc_df["yhat"].values, color=color, linewidth=1.5, label="Prophet")
                ax.fill_between(
                    idx,
                    fc_df["yhat_lower"].values,
                    fc_df["yhat_upper"].values,
                    color=color, alpha=0.15, label="Prophet 95% CI",
                )

            elif model_name == "LSTM":
                fc = _forecast_lstm(dataset_name, train_pct, horizon)
                ax.plot(fc.index, fc.values, color=color, linewidth=1.5, label="LSTM")

            elif model_name == "Ensemble":
                fc = _forecast_ensemble(dataset_name, train_pct, horizon)
                ax.plot(fc.index, fc.values, color=color, linewidth=1.5, label="Ensemble")

        except Exception as exc:
            st.warning(f"{model_name} forecast failed: {exc}")

    ax.set_title(f"{dataset_name} — {horizon}-day Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _build_stl_fig(df: pd.DataFrame, dataset_name: str) -> plt.Figure | None:
    from analysis.decomposition import stl_decompose
    series = df.set_index("ds")["y"]
    period = _PERIOD_MAP.get(dataset_name, 365)
    # Use a shorter period if series is too short
    if len(series) < 2 * period:
        period = max(7, len(series) // 3)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl_result = stl_decompose(series, period=period)
    except Exception as exc:
        st.warning(f"STL decomposition failed: {exc}")
        return None

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"STL Decomposition — {dataset_name} (period={period})", fontsize=13)
    panels = [
        (np.array(stl_result.trend), "Trend"),
        (np.array(stl_result.seasonal), "Seasonal"),
        (np.array(stl_result.resid), "Residual"),
    ]
    for ax, (data, label) in zip(axes, panels):
        ax.plot(series.index, data, linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metrics table builder
# ---------------------------------------------------------------------------

def _build_metrics_table(
    df: pd.DataFrame,
    train_pct: int,
    selected_models: list[str],
    horizon: int,
    dataset_name: str,
) -> pd.DataFrame:
    series = df.set_index("ds")["y"]
    n = len(series)
    cut = int(n * train_pct / 100)
    test_actual = series.iloc[cut: cut + horizon].values

    if len(test_actual) == 0:
        return pd.DataFrame()

    rows = []
    for model_name in selected_models:
        try:
            predicted = _get_forecast_values(model_name, dataset_name, train_pct, horizon)
            min_len = min(len(test_actual), len(predicted))
            row = _compute_metrics(test_actual[:min_len], predicted[:min_len])
            row["Model"] = model_name
            rows.append(row)
        except Exception as exc:
            st.warning(f"Metrics for {model_name} failed: {exc}")

    if not rows:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows).set_index("Model")
    return metrics_df[["MAPE", "sMAPE", "RMSE", "MAE", "MASE"]]


def _get_forecast_values(
    model_name: str, dataset_name: str, train_pct: int, horizon: int
) -> np.ndarray:
    """Return forecast as 1-D numpy array."""
    if model_name == "Naive":
        return _forecast_naive(dataset_name, train_pct, horizon).values
    if model_name == "ARIMA":
        return _forecast_arima(dataset_name, train_pct, horizon)["mean"].values
    if model_name == "Prophet":
        fc_df = _forecast_prophet(dataset_name, train_pct, horizon)
        return fc_df["yhat"].values
    if model_name == "LSTM":
        return _forecast_lstm(dataset_name, train_pct, horizon).values
    if model_name == "Ensemble":
        return _forecast_ensemble(dataset_name, train_pct, horizon).values
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Walk-forward results builder
# ---------------------------------------------------------------------------

def _build_wf_table(
    selected_models: list[str], dataset_name: str
) -> pd.DataFrame:
    rows = []
    for model_name in selected_models:
        if model_name == "Naive":
            continue
        with st.spinner(f"Running walk-forward for {model_name}…"):
            try:
                wf_df = _run_walkforward(dataset_name, model_name)
                if wf_df.empty:
                    continue
                mean_mape = wf_df["mape"].mean()
                std_mape = wf_df["mape"].std()
                rows.append({
                    "Model": model_name,
                    "MAPE mean": round(mean_mape, 3),
                    "MAPE std": round(std_mape, 3),
                    "RMSE mean": round(wf_df["rmse"].mean(), 3),
                    "MAE mean": round(wf_df["mae"].mean(), 3),
                    "Folds": len(wf_df),
                })
            except Exception as exc:
                st.warning(f"Walk-forward for {model_name} failed: {exc}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="TS Forecasting Dashboard", layout="wide")
    st.title("Time Series Forecasting Dashboard")

    # ------------------------------------------------------------------ Sidebar
    st.sidebar.header("Configuration")

    dataset = st.sidebar.selectbox("Dataset", DATASET_OPTIONS, index=0)

    train_pct = st.sidebar.slider(
        "Training window (%)", min_value=50, max_value=90, value=80, step=5
    )

    selected_models = st.sidebar.multiselect(
        "Models",
        options=ALL_MODELS,
        default=["Naive", "ARIMA"],
    )

    horizon = st.sidebar.selectbox(
        "Forecast horizon (days)",
        options=HORIZON_OPTIONS,
        index=0,
    )

    if not selected_models:
        st.warning("Select at least one model from the sidebar to continue.")
        return

    # ------------------------------------------------------------------ Load data
    try:
        df = _load_data(dataset)
    except FileNotFoundError as exc:
        st.error(
            f"Dataset not found: {exc}\n\n"
            "Run `python -m data.download` from the project root to download data."
        )
        return

    # ------------------------------------------------------------------ Tabs
    tab_forecast, tab_wf = st.tabs(["Forecast", "Walk-Forward"])

    with tab_forecast:
        st.subheader(f"{dataset} — {horizon}-day Forecast")

        # Forecast chart
        with st.spinner("Generating forecasts…"):
            fig = _build_forecast_fig(df, train_pct, selected_models, horizon, dataset)
        st.pyplot(fig)
        plt.close(fig)

        # Metrics table
        st.subheader("Metrics on held-out data")
        with st.spinner("Computing metrics…"):
            metrics_df = _build_metrics_table(df, train_pct, selected_models, horizon, dataset)

        if metrics_df.empty:
            st.info("No metrics available — held-out window may be empty.")
        else:
            st.dataframe(metrics_df.style.format("{:.3f}"))

        # STL decomposition expander
        with st.expander("STL Decomposition"):
            with st.spinner("Decomposing series…"):
                stl_fig = _build_stl_fig(df, dataset)
            if stl_fig is not None:
                st.pyplot(stl_fig)
                plt.close(stl_fig)

    with tab_wf:
        st.subheader("Walk-Forward Backtesting Results")
        st.caption(
            "Mean ± std MAPE across 3 expanding-window folds (Naive excluded)."
        )
        wf_table = _build_wf_table(selected_models, dataset)
        if wf_table.empty:
            st.info("No walk-forward results available (select ARIMA, Prophet, LSTM, or Ensemble).")
        else:
            st.dataframe(wf_table)


if __name__ == "__main__":
    main()
