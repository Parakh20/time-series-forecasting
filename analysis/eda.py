"""
Exploratory data analysis for time series data.

Functions for initial data exploration, visualization, and summary statistics.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

RANDOM_SEED = 42

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _ensure_results_dir() -> None:
    """Create results directory if it does not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_time_series(df: pd.DataFrame, name: str) -> str:
    """Plot the time series with 30-day rolling mean and rolling std.

    Args:
        df: DataFrame with 'ds' (datetime) and 'y' (float) columns.
        name: Dataset label used in the title and filename.

    Returns:
        Absolute path to the saved PNG file.
    """
    _ensure_results_dir()
    output_path = RESULTS_DIR / f"eda_{name}_timeseries.png"

    series = df.set_index("ds")["y"]
    rolling_mean = series.rolling(window=30).mean()
    rolling_std = series.rolling(window=30).std()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Time Series — {name}", fontsize=14)

    ax1.plot(series.index, series.values, linewidth=0.7, alpha=0.8, label="Original")
    ax1.plot(rolling_mean.index, rolling_mean.values, linewidth=1.5, label="30-day rolling mean")
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(rolling_std.index, rolling_std.values, linewidth=1.0, color="orange")
    ax2.set_ylabel("30-day rolling std")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def plot_acf_pacf(df: pd.DataFrame, name: str, lags: int = 40) -> str:
    """Plot ACF and PACF side by side.

    Args:
        df: DataFrame with 'y' column.
        name: Dataset label.
        lags: Number of lags to show.

    Returns:
        Absolute path to the saved PNG file.
    """
    _ensure_results_dir()
    output_path = RESULTS_DIR / f"eda_{name}_acf_pacf.png"

    series = df["y"].dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"ACF / PACF — {name}", fontsize=14)

    plot_acf(series, lags=lags, ax=ax1, title="ACF")
    plot_pacf(series, lags=lags, ax=ax2, title="PACF", method="ywm")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def plot_seasonal_subseries(df: pd.DataFrame, name: str) -> str:
    """Seasonal subseries plot grouped by calendar month.

    Each subplot shows the values for one month across all years,
    with a horizontal line at that month's mean.

    Args:
        df: DataFrame with 'ds' and 'y' columns.
        name: Dataset label.

    Returns:
        Absolute path to the saved PNG file.
    """
    _ensure_results_dir()
    output_path = RESULTS_DIR / f"eda_{name}_seasonal_subseries.png"

    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp["ds"]).dt.month
    tmp["month_name"] = pd.to_datetime(tmp["ds"]).dt.strftime("%b")

    month_order = list(range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=True)
    fig.suptitle(f"Seasonal Subseries (by month) — {name}", fontsize=14)
    axes_flat = axes.flatten()

    for idx, (month_num, label) in enumerate(zip(month_order, month_labels)):
        ax = axes_flat[idx]
        subset = tmp[tmp["month"] == month_num]["y"]
        if len(subset) == 0:
            ax.set_visible(False)
            continue
        ax.plot(range(len(subset)), subset.values, linewidth=0.8, alpha=0.7)
        ax.axhline(subset.mean(), color="red", linewidth=1.5, linestyle="--")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def plot_distributions(df: pd.DataFrame, name: str) -> str:
    """Plot value distribution histogram and returns (pct_change) distribution.

    Args:
        df: DataFrame with 'y' column.
        name: Dataset label.

    Returns:
        Absolute path to the saved PNG file.
    """
    _ensure_results_dir()
    output_path = RESULTS_DIR / f"eda_{name}_distributions.png"

    values = df["y"].dropna()
    returns = values.pct_change().dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Distributions — {name}", fontsize=14)

    ax1.hist(values, bins=50, edgecolor="black", linewidth=0.4)
    ax1.set_title("Value distribution")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)

    ax2.hist(returns, bins=50, edgecolor="black", linewidth=0.4, color="steelblue")
    ax2.set_title("Returns distribution (pct_change)")
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def run_eda(df: pd.DataFrame, name: str) -> dict:
    """Run all EDA plots and return a dict of plot file paths.

    Args:
        df: DataFrame with 'ds' and 'y' columns.
        name: Dataset label.

    Returns:
        Dict mapping plot type to absolute path strings:
          - 'timeseries'
          - 'acf_pacf'
          - 'seasonal_subseries'
          - 'distributions'
    """
    paths = {
        "timeseries": plot_time_series(df, name),
        "acf_pacf": plot_acf_pacf(df, name),
        "seasonal_subseries": plot_seasonal_subseries(df, name),
        "distributions": plot_distributions(df, name),
    }
    return paths


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from data.download import load_dataset

    try:
        energy = load_dataset("energy")
        commodity = load_dataset("commodity")

        # --- Energy EDA ---
        print("Running EDA for energy dataset...")
        e_paths = run_eda(energy, "energy")

        expected_keys = {"timeseries", "acf_pacf", "seasonal_subseries", "distributions"}
        assert set(e_paths.keys()) == expected_keys, f"Missing keys: {expected_keys - set(e_paths.keys())}"

        for key, path in e_paths.items():
            assert Path(path).exists(), f"Plot not saved: {path}"
            print(f"  {key}: {path}")

        # --- Commodity EDA ---
        print("\nRunning EDA for commodity dataset...")
        c_paths = run_eda(commodity, "commodity")

        for key, path in c_paths.items():
            assert Path(path).exists(), f"Plot not saved: {path}"
            print(f"  {key}: {path}")

        print("\nAll EDA smoke tests passed.")
        sys.exit(0)
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
