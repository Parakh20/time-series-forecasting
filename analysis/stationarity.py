"""
Stationarity testing and analysis.

Functions for testing time series stationarity (ADF test, KPSS test, etc.).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

RANDOM_SEED = 42


def adf_test(series: pd.Series) -> dict:
    """Run the Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Time series to test. Must have no NaN values.

    Returns:
        Dict with keys: statistic, p_value, is_stationary (p < 0.05), lags.

    Raises:
        ValueError: If the series is empty or contains NaN values.
    """
    clean = series.dropna()
    if len(clean) == 0:
        raise ValueError("Series is empty after dropping NaN values.")

    result = adfuller(clean, autolag="AIC")
    statistic, p_value, lags = result[0], result[1], result[2]

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "is_stationary": bool(p_value < 0.05),
        "lags": int(lags),
    }


def kpss_test(series: pd.Series) -> dict:
    """Run the KPSS test for stationarity.

    The null hypothesis of KPSS is stationarity, so p > 0.05 means stationary.

    Args:
        series: Time series to test. Must have no NaN values.

    Returns:
        Dict with keys: statistic, p_value, is_stationary (p > 0.05), lags.

    Raises:
        ValueError: If the series is empty or contains NaN values.
    """
    clean = series.dropna()
    if len(clean) == 0:
        raise ValueError("Series is empty after dropping NaN values.")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(clean, regression="c", nlags="auto")

    statistic, p_value, lags = result[0], result[1], result[2]

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "is_stationary": bool(p_value > 0.05),
        "lags": int(lags),
    }


def make_stationary(series: pd.Series, max_diffs: int = 2) -> tuple:
    """Apply differencing until the series is stationary (ADF p < 0.05).

    Args:
        series: Input time series.
        max_diffs: Maximum number of differences to apply.

    Returns:
        Tuple of (differenced_series, n_diffs applied).
    """
    current = series.dropna().copy()
    n_diffs = 0

    for _ in range(max_diffs):
        result = adf_test(current)
        if result["is_stationary"]:
            break
        current = current.diff().dropna()
        n_diffs += 1

    return current, n_diffs


def log_transform_if_needed(series: pd.Series) -> tuple:
    """Apply log transform if coefficient of variation exceeds 0.5.

    Coefficient of variation = std / mean. A high CV suggests multiplicative
    variance structure where log-transform stabilises variance.

    Args:
        series: Input time series (must be strictly positive for log).

    Returns:
        Tuple of (transformed_series, was_applied: bool).

    Raises:
        ValueError: If series contains non-positive values when log is needed.
    """
    clean = series.dropna()
    mean_val = clean.mean()

    if mean_val == 0:
        return series, False

    cv = clean.std() / abs(mean_val)

    if cv <= 0.5:
        return series, False

    if (clean <= 0).any():
        raise ValueError(
            "Log transform required (CV > 0.5) but series contains non-positive values."
        )

    return np.log(series), True


def stationarity_report(df: pd.DataFrame, name: str) -> dict:
    """Run ADF and KPSS tests before and after differencing; print p-values.

    Args:
        df: DataFrame with 'y' column.
        name: Dataset label for print output.

    Returns:
        Dict with keys 'pre' and 'post', each containing 'adf' and 'kpss' dicts.
    """
    series = df["y"].dropna()

    print(f"\n=== Stationarity Report: {name} ===")
    print(f"Series length: {len(series)}")

    # Pre-differencing
    adf_pre = adf_test(series)
    kpss_pre = kpss_test(series)

    print("\n--- Original series ---")
    print(f"  ADF  p-value: {adf_pre['p_value']:.6f}  stationary={adf_pre['is_stationary']}")
    print(f"  KPSS p-value: {kpss_pre['p_value']:.6f}  stationary={kpss_pre['is_stationary']}")

    # Post-differencing
    stationary_series, n_diffs = make_stationary(series)
    adf_post = adf_test(stationary_series)
    kpss_post = kpss_test(stationary_series)

    print(f"\n--- After {n_diffs} difference(s) ---")
    print(f"  ADF  p-value: {adf_post['p_value']:.6f}  stationary={adf_post['is_stationary']}")
    print(f"  KPSS p-value: {kpss_post['p_value']:.6f}  stationary={kpss_post['is_stationary']}")

    return {
        "pre": {"adf": adf_pre, "kpss": kpss_pre},
        "post": {"adf": adf_post, "kpss": kpss_post, "n_diffs": n_diffs},
    }


if __name__ == "__main__":
    # Add project root to path so data.download is importable
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from data.download import load_dataset

    try:
        energy = load_dataset("energy")
        commodity = load_dataset("commodity")

        # --- Basic API checks ---
        e_adf = adf_test(energy["y"])
        assert "p_value" in e_adf, "ADF result missing p_value"
        assert "is_stationary" in e_adf, "ADF result missing is_stationary"
        assert isinstance(e_adf["p_value"], float), "ADF p_value not float"
        print(f"Energy ADF p-value: {e_adf['p_value']:.6f}")

        e_kpss = kpss_test(energy["y"])
        assert "p_value" in e_kpss, "KPSS result missing p_value"
        assert isinstance(e_kpss["p_value"], float), "KPSS p_value not float"
        print(f"Energy KPSS p-value: {e_kpss['p_value']:.6f}")

        # --- Differencing reduces ADF p-value on commodity (non-stationary) ---
        c_adf_pre = adf_test(commodity["y"])
        stationary_series, n_diffs = make_stationary(commodity["y"])
        c_adf_post = adf_test(stationary_series)

        print(f"\nCommodity ADF pre-diff  p-value: {c_adf_pre['p_value']:.6f}")
        print(f"Commodity ADF post-diff p-value: {c_adf_post['p_value']:.6f} ({n_diffs} diffs)")

        assert c_adf_post["p_value"] < c_adf_pre["p_value"], (
            f"Differencing should reduce ADF p-value: "
            f"pre={c_adf_pre['p_value']:.4f} post={c_adf_post['p_value']:.4f}"
        )
        assert c_adf_post["is_stationary"], "Series should be stationary after differencing"

        # --- log_transform_if_needed ---
        transformed, applied = log_transform_if_needed(commodity["y"])
        print(f"\nCommodity log transform applied: {applied}")

        # --- Full reports ---
        stationarity_report(energy, "energy")
        stationarity_report(commodity, "commodity")

        print("\nAll stationarity smoke tests passed.")
        sys.exit(0)
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
