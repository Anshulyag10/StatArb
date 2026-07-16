"""
cointegration.py -- Cointegration tests for pairs trading.

Tests:
    1. Engle-Granger (two-step, original)
    2. Augmented Dickey-Fuller (spread stationarity)
    3. Johansen (new, multivariate)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def engle_granger_test(
    series_y: pd.Series,
    series_x: pd.Series,
    p_threshold: float = 0.01,
) -> dict:
    """
    Run the Engle-Granger two-step cointegration test.

    Returns
    -------
    dict with keys:
        t_stat, p_value, crit_values, is_cointegrated
    """
    t_stat, p_value, crit_values = coint(series_y, series_x)

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "crit_values": {"1%": crit_values[0],
                        "5%": crit_values[1],
                        "10%": crit_values[2]},
        "is_cointegrated": p_value < p_threshold,
    }


def adf_test(
    spread: pd.Series,
    p_threshold: float = 0.001,
) -> dict:
    """
    Augmented Dickey-Fuller test on the spread to confirm stationarity.

    Returns
    -------
    dict with keys:
        adf_stat, p_value, used_lag, n_obs, crit_values, is_stationary
    """
    result = adfuller(spread.dropna(), autolag="AIC")
    adf_stat, p_value, used_lag, n_obs, crit_vals, _ = result

    return {
        "adf_stat": adf_stat,
        "p_value": p_value,
        "used_lag": used_lag,
        "n_obs": n_obs,
        "crit_values": crit_vals,
        "is_stationary": p_value < p_threshold,
    }


def johansen_test(
    series_y: pd.Series,
    series_x: pd.Series,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """
    Run the Johansen cointegration test (trace statistic).

    Returns
    -------
    dict with keys:
        trace_stat, crit_95, is_cointegrated (at 95% level)
    """
    data = pd.DataFrame({"y": series_y, "x": series_x}).dropna()

    try:
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
        trace_stat = result.lr1[0]
        crit_95 = result.cvt[0, 1]

        return {
            "trace_stat": float(trace_stat),
            "crit_95": float(crit_95),
            "is_cointegrated": trace_stat > crit_95,
        }
    except Exception:
        return {
            "trace_stat": 0.0,
            "crit_95": 0.0,
            "is_cointegrated": False,
        }


def print_cointegration_report(
    eg_result: dict,
    adf_result: dict,
    joh_result: dict | None = None,
) -> None:
    """Pretty-print cointegration and stationarity results."""
    print("\n" + "=" * 60)
    print("  COINTEGRATION ANALYSIS")
    print("=" * 60)

    print(f"\n  Engle-Granger Test")
    print(f"  {'t-statistic:':<20} {eg_result['t_stat']:.4f}")
    print(f"  {'p-value:':<20} {eg_result['p_value']:.6f}")
    for level, val in eg_result["crit_values"].items():
        print(f"  {'Crit (' + level + '):':<20} {val:.4f}")
    tag = "YES - COINTEGRATED" if eg_result["is_cointegrated"] else "NO - NOT cointegrated"
    print(f"  Result: {tag}")

    print(f"\n  Augmented Dickey-Fuller Test (spread)")
    print(f"  {'ADF statistic:':<20} {adf_result['adf_stat']:.4f}")
    print(f"  {'p-value:':<20} {adf_result['p_value']:.6f}")
    for level, val in adf_result["crit_values"].items():
        print(f"  {'Crit (' + level + '):':<20} {val:.4f}")
    tag = "YES - STATIONARY" if adf_result["is_stationary"] else "NO - NOT stationary"
    print(f"  Result: {tag}")

    if joh_result is not None:
        print(f"\n  Johansen Test (trace statistic)")
        print(f"  {'Trace stat:':<20} {joh_result['trace_stat']:.4f}")
        print(f"  {'95% critical:':<20} {joh_result['crit_95']:.4f}")
        tag = "YES - COINTEGRATED" if joh_result["is_cointegrated"] else "NO - NOT cointegrated"
        print(f"  Result: {tag}")

    print("=" * 60 + "\n")
