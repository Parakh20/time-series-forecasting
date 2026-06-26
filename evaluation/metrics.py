"""
Evaluation metrics for time series forecasting.

Metrics include MAE, RMSE, MAPE, sMAPE, and MASE.
"""

import numpy as np

RANDOM_SEED = 42


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Skips indices where actual == 0 to avoid division by zero.

    Args:
        actual: Ground-truth values.
        predicted: Forecasted values.

    Returns:
        MAPE as a percentage in [0, 100+].
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0.0
    if not np.any(mask):
        return float("nan")
    pct_errors = np.abs((actual[mask] - predicted[mask]) / actual[mask]) * 100.0
    return float(np.mean(pct_errors))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE: 200 * mean(|a - p| / (|a| + |p|)).

    Skips indices where |a| + |p| < 1e-8 to avoid division by near-zero.

    Args:
        actual: Ground-truth values.
        predicted: Forecasted values.

    Returns:
        sMAPE as a percentage in [0, 200].
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.abs(actual) + np.abs(predicted)
    mask = denom >= 1e-8
    if not np.any(mask):
        return float("nan")
    values = 200.0 * np.abs(actual[mask] - predicted[mask]) / denom[mask]
    return float(np.mean(values))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        actual: Ground-truth values.
        predicted: Forecasted values.

    Returns:
        RMSE in the same units as actual/predicted.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        actual: Ground-truth values.
        predicted: Forecasted values.

    Returns:
        MAE in the same units as actual/predicted.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    seasonal_period: int = 1,
    train: np.ndarray | None = None,
) -> float:
    """Mean Absolute Scaled Error.

    Scale is the mean absolute naive seasonal forecast error on the in-sample
    (training) series: mean |train[s:] - train[:-s]|.  When ``train`` is not
    provided the scale falls back to the same computation on ``actual``
    (backward-compatible behaviour for smoke tests that don't have a separate
    training series).

    Args:
        actual: Ground-truth test values.
        predicted: Forecasted values.
        seasonal_period: Seasonal lag s used for the naive benchmark (default 1).
        train: In-sample training series used to compute the naive-forecast
            scale denominator.  If None, ``actual`` is used as a fallback.

    Returns:
        MASE (dimensionless ratio).  Returns nan if scale == 0.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    s = seasonal_period

    scale_series = np.asarray(train, dtype=float) if train is not None else actual
    if len(scale_series) <= s:
        return float("nan")
    scale = np.mean(np.abs(scale_series[s:] - scale_series[:-s]))
    if scale < 1e-10:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / scale)


if __name__ == "__main__":
    import sys

    actual_arr = np.array([100.0, 200.0, 150.0, 0.0, 50.0])
    pred_arr = np.array([110.0, 190.0, 160.0, 5.0, 45.0])

    mape_val = mape(actual_arr, pred_arr)
    rmse_val = rmse(actual_arr, pred_arr)
    mae_val = mae(actual_arr, pred_arr)
    mase_val = mase(actual_arr, pred_arr, seasonal_period=1)
    smape_val = smape(actual_arr, pred_arr)

    errors: list[str] = []

    # MAPE: skip idx3 (actual=0). Pct errors: 10%, 5%, 6.667%, 10% -> mean=7.917%
    if not abs(mape_val - 7.917) < 0.01:
        errors.append(f"MAPE={mape_val:.4f}, expected ~7.917")

    # RMSE: errors=[10,10,10,5,5], MSE=70, sqrt=8.3666
    if not abs(rmse_val - 8.3666) < 0.01:
        errors.append(f"RMSE={rmse_val:.4f}, expected ~8.3666")

    # MAE: mean([10,10,10,5,5])=8.0
    if not abs(mae_val - 8.0) < 0.01:
        errors.append(f"MAE={mae_val:.4f}, expected ~8.0")

    # MASE (period=1): naive_errors=|diff(actual)|=[100,50,150,50], scale=87.5, mase=8/87.5=0.09143
    if not abs(mase_val - 0.09143) < 0.001:
        errors.append(f"MASE={mase_val:.5f}, expected ~0.09143")

    # sMAPE: just verify in (0, 200)
    if not (0 < smape_val < 200):
        errors.append(f"sMAPE={smape_val:.4f}, expected in (0, 200)")

    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)

    print("All metric smoke tests passed.")
    print(
        f"  MAPE={mape_val:.3f}%  RMSE={rmse_val:.4f}  MAE={mae_val:.4f}"
        f"  MASE={mase_val:.5f}  sMAPE={smape_val:.3f}%"
    )
