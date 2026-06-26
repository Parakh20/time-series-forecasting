"""
Time series decomposition analysis.

Functions for decomposing time series into trend, seasonal, and residual components.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose

RANDOM_SEED = 42

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _ensure_results_dir() -> None:
    """Create results directory if it does not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def decompose_additive(series: pd.Series, period: int):
    """Additive seasonal decomposition using statsmodels.

    Args:
        series: Time series to decompose. Must have no leading/trailing NaN.
        period: Seasonal period (e.g. 365 for annual daily, 252 for trading-day annual).

    Returns:
        statsmodels DecomposeResult (additive model).

    Raises:
        ValueError: If series is too short for the given period.
    """
    clean = series.dropna()
    if len(clean) < 2 * period:
        raise ValueError(
            f"Series length {len(clean)} is too short for period {period} "
            "(need at least 2 * period)."
        )
    return seasonal_decompose(clean, model="additive", period=period, extrapolate_trend="freq")


def decompose_multiplicative(series: pd.Series, period: int):
    """Multiplicative classical seasonal decomposition.

    Args:
        series: Time series to decompose. Values must be strictly positive.
        period: Seasonal period.

    Returns:
        statsmodels DecomposeResult (multiplicative model).

    Raises:
        ValueError: If series contains non-positive values or is too short.
    """
    clean = series.dropna()
    if len(clean) < 2 * period:
        raise ValueError(
            f"Series length {len(clean)} is too short for period {period} "
            "(need at least 2 * period)."
        )
    if (clean <= 0).any():
        raise ValueError("Multiplicative decomposition requires strictly positive values.")
    return seasonal_decompose(clean, model="multiplicative", period=period, extrapolate_trend="freq")


def stl_decompose(series: pd.Series, period: int):
    """STL (Seasonal and Trend decomposition using Loess) decomposition.

    Args:
        series: Time series to decompose.
        period: Seasonal period.

    Returns:
        statsmodels STLForecast/STL result (DecomposeResult-like with .trend,
        .seasonal, .resid).

    Raises:
        ValueError: If series is too short for the given period.
    """
    clean = series.dropna()
    if len(clean) < 2 * period:
        raise ValueError(
            f"Series length {len(clean)} is too short for period {period} "
            "(need at least 2 * period)."
        )
    stl = STL(clean, period=period, robust=True)
    return stl.fit()


def trend_strength(decomp_result) -> float:
    """Compute trend strength from a decomposition result.

    Follows Wang, Smith & Hyndman (2006):
      Ft = max(0, 1 - Var(resid) / Var(trend + resid))

    Args:
        decomp_result: Result with .trend and .resid attributes.

    Returns:
        Float in [0, 1]. Values near 1 indicate strong trend.
    """
    resid = np.array(decomp_result.resid, dtype=float)
    trend = np.array(decomp_result.trend, dtype=float)

    # Drop NaN (classical decomposition may leave NaN at edges)
    mask = ~(np.isnan(resid) | np.isnan(trend))
    resid = resid[mask]
    trend = trend[mask]

    var_resid = np.var(resid, ddof=1)
    var_trend_resid = np.var(trend + resid, ddof=1)

    if var_trend_resid == 0:
        return 0.0

    return float(max(0.0, 1.0 - var_resid / var_trend_resid))


def seasonal_strength(decomp_result) -> float:
    """Compute seasonal strength from a decomposition result.

    Follows Wang, Smith & Hyndman (2006):
      Fs = max(0, 1 - Var(resid) / Var(seasonal + resid))

    Args:
        decomp_result: Result with .seasonal and .resid attributes.

    Returns:
        Float in [0, 1]. Values near 1 indicate strong seasonality.
    """
    resid = np.array(decomp_result.resid, dtype=float)
    seasonal = np.array(decomp_result.seasonal, dtype=float)

    mask = ~(np.isnan(resid) | np.isnan(seasonal))
    resid = resid[mask]
    seasonal = seasonal[mask]

    var_resid = np.var(resid, ddof=1)
    var_seasonal_resid = np.var(seasonal + resid, ddof=1)

    if var_seasonal_resid == 0:
        return 0.0

    return float(max(0.0, 1.0 - var_resid / var_seasonal_resid))


def _save_stl_plot(stl_result, name: str) -> str:
    """Save a 4-panel STL decomposition plot to results/.

    Args:
        stl_result: Fitted STL result.
        name: Dataset name used for the filename.

    Returns:
        Absolute path of the saved file.
    """
    _ensure_results_dir()
    output_path = RESULTS_DIR / f"decomp_{name}_stl.png"

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"STL Decomposition — {name}", fontsize=14)

    components = [
        (stl_result.observed, "Observed"),
        (stl_result.trend, "Trend"),
        (stl_result.seasonal, "Seasonal"),
        (stl_result.resid, "Residual"),
    ]

    for ax, (data, label) in zip(axes, components):
        ax.plot(np.array(data), linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def decomposition_report(df: pd.DataFrame, name: str, period: int = None) -> dict:
    """Run additive, multiplicative, and STL decompositions; compute strengths.

    Saves a 4-panel STL plot to results/decomp_{name}_stl.png.

    Args:
        df: DataFrame with 'y' column.
        name: Dataset label.
        period: Seasonal period. Defaults to 365 for 'energy', 252 for 'commodity'.

    Returns:
        Dict with keys 'additive', 'multiplicative', 'stl', each containing
        'trend_strength' and 'seasonal_strength'; plus 'plot_path'.
    """
    series = df["y"].dropna()

    if period is None:
        period = 365 if name == "energy" else 252

    print(f"\n=== Decomposition Report: {name} (period={period}) ===")

    additive = decompose_additive(series, period)

    # Multiplicative decomposition requires strictly positive values.
    # If the series has non-positive values, shift it by (|min| + 1).
    if (series <= 0).any():
        shift = abs(series.min()) + 1.0
        print(f"  [multiplicative] Shifting series by +{shift:.4f} to ensure positivity.")
        multiplicative = decompose_multiplicative(series + shift, period)
    else:
        multiplicative = decompose_multiplicative(series, period)

    stl_result = stl_decompose(series, period)

    add_ts = trend_strength(additive)
    add_ss = seasonal_strength(additive)
    mul_ts = trend_strength(multiplicative)
    mul_ss = seasonal_strength(multiplicative)
    stl_ts = trend_strength(stl_result)
    stl_ss = seasonal_strength(stl_result)

    plot_path = _save_stl_plot(stl_result, name)

    print(f"  Additive       — trend_strength={add_ts:.3f}  seasonal_strength={add_ss:.3f}")
    print(f"  Multiplicative — trend_strength={mul_ts:.3f}  seasonal_strength={mul_ss:.3f}")
    print(f"  STL            — trend_strength={stl_ts:.3f}  seasonal_strength={stl_ss:.3f}")
    print(f"  STL plot saved: {plot_path}")

    return {
        "additive": {"trend_strength": add_ts, "seasonal_strength": add_ss},
        "multiplicative": {"trend_strength": mul_ts, "seasonal_strength": mul_ss},
        "stl": {"trend_strength": stl_ts, "seasonal_strength": stl_ss},
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from data.download import load_dataset

    try:
        energy = load_dataset("energy")
        commodity = load_dataset("commodity")

        # --- STL yields trend/seasonal/resid arrays ---
        stl_result = stl_decompose(energy["y"], period=365)
        assert hasattr(stl_result, "trend"), "STL result missing .trend"
        assert hasattr(stl_result, "seasonal"), "STL result missing .seasonal"
        assert hasattr(stl_result, "resid"), "STL result missing .resid"

        trend_arr = np.array(stl_result.trend)
        seasonal_arr = np.array(stl_result.seasonal)
        resid_arr = np.array(stl_result.resid)

        assert len(trend_arr) == len(energy), "trend length mismatch"
        assert len(seasonal_arr) == len(energy), "seasonal length mismatch"
        assert len(resid_arr) == len(energy), "resid length mismatch"
        print(f"Energy STL arrays OK — length {len(trend_arr)}")

        # --- Strength metrics are in [0, 1] ---
        ts = trend_strength(stl_result)
        ss = seasonal_strength(stl_result)
        assert 0.0 <= ts <= 1.0, f"trend_strength out of range: {ts}"
        assert 0.0 <= ss <= 1.0, f"seasonal_strength out of range: {ss}"
        print(f"Energy STL trend_strength={ts:.3f} seasonal_strength={ss:.3f}")

        # --- Full reports ---
        e_report = decomposition_report(energy, "energy", period=365)
        assert "plot_path" in e_report, "report missing plot_path"
        assert Path(e_report["plot_path"]).exists(), "STL plot not saved"

        c_report = decomposition_report(commodity, "commodity", period=252)
        assert Path(c_report["plot_path"]).exists(), "STL plot not saved for commodity"

        print("\nAll decomposition smoke tests passed.")
        sys.exit(0)
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
