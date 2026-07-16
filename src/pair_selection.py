"""
pair_selection.py -- Multi-pair screening using statistical tests.

Screening criteria:
    1. Engle-Granger cointegration test
    2. Johansen cointegration test (trace statistic)
    3. Hurst exponent (R/S method) -- < 0.5 = mean-reverting
    4. Half-life of mean reversion (AR(1) regression)
    5. Correlation stability (std of rolling correlation)

Reference:
    Gatev, Goetzmann & Rouwenhorst (2006) -- "Pairs Trading"
    Vidyamurthy (2004) -- "Pairs Trading: Quantitative Methods and Analysis"
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.hedge_ratio import estimate_hedge_ratio, compute_spread


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """
    Compute the Hurst exponent using the Rescaled Range (R/S) method.

    H < 0.5 : mean-reverting (good for pairs trading)
    H = 0.5 : random walk
    H > 0.5 : trending (bad for pairs trading)
    """
    ts = series.dropna().values
    n = len(ts)
    if n < 40:
        return 0.5  # Not enough data

    max_lag = min(max_lag, n // 4)
    lags = range(2, max_lag + 1)
    rs_values = []

    for lag in lags:
        # Split into sub-series of length 'lag'
        n_subseries = n // lag
        if n_subseries < 1:
            continue

        rs_list = []
        for i in range(n_subseries):
            subset = ts[i * lag : (i + 1) * lag]
            mean = np.mean(subset)
            deviations = np.cumsum(subset - mean)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(subset, ddof=1)
            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            rs_values.append((np.log(lag), np.log(np.mean(rs_list))))

    if len(rs_values) < 3:
        return 0.5

    log_lags, log_rs = zip(*rs_values)
    # Linear regression: log(R/S) = H * log(lag) + c
    coeffs = np.polyfit(log_lags, log_rs, 1)
    return float(coeffs[0])


def variance_ratio_test(series: pd.Series, lag: int = 5) -> float:
    """
    Compute the Lo-MacKinlay Variance Ratio (VR) test.
    
    VR(k) = Var(r_t(k)) / (k * Var(r_t))
    
    VR < 1 : Mean-reverting (negative autocorrelation)
    VR = 1 : Random walk
    VR > 1 : Trending (positive autocorrelation)
    """
    ts = series.dropna()
    if len(ts) < lag * 2:
        return 1.0
    
    # 1-period returns (diffs)
    r1 = ts.diff().dropna()
    var_1 = r1.var(ddof=1)
    
    # k-period returns (diffs)
    rk = ts.diff(periods=lag).dropna()
    var_k = rk.var(ddof=1)
    
    if var_1 == 0:
        return 1.0
        
    vr = var_k / (lag * var_1)
    return float(vr)


def half_life_mean_reversion(spread: pd.Series) -> float:
    """
    Compute the half-life of mean reversion from AR(1) regression.

    Model: delta_spread_t = a + b * spread_{t-1} + eps
    Half-life = -ln(2) / ln(1 + b)

    Returns half-life in days. Returns inf if not mean-reverting.
    """
    s = spread.dropna()
    if len(s) < 30:
        return float("inf")

    y = s.diff().iloc[1:].values
    x = s.iloc[:-1].values

    x_const = np.column_stack([np.ones(len(x)), x])
    coeffs, _, _, _ = np.linalg.lstsq(x_const, y, rcond=None)
    b = coeffs[1]

    if b >= 0:
        return float("inf")  # Not mean-reverting

    half_life = -np.log(2) / np.log(1 + b)
    return float(half_life)


def correlation_stability(
    series_y: pd.Series,
    series_x: pd.Series,
    window: int = 60,
) -> float:
    """
    Compute the standard deviation of rolling correlation.
    Lower = more stable relationship (better for pairs trading).
    """
    rolling_corr = series_y.rolling(window).corr(series_x)
    return float(rolling_corr.std())


def johansen_test(
    series_y: pd.Series,
    series_x: pd.Series,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """
    Run the Johansen cointegration test.

    Returns trace statistic, critical values, and whether cointegrated
    at 95% confidence level.
    """
    data = pd.DataFrame({"y": series_y, "x": series_x}).dropna()

    try:
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

        # Trace statistic for r=0 (no cointegration) hypothesis
        trace_stat = result.lr1[0]   # First eigenvalue trace stat
        crit_95 = result.cvt[0, 1]   # 95% critical value for r=0

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


def screen_pairs(
    prices: pd.DataFrame,
    candidate_pairs: list[tuple[str, str]],
    eg_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Screen all candidate pairs and compute selection metrics.

    Parameters
    ----------
    prices         : DataFrame with columns for each ticker
    candidate_pairs: list of (ticker_y, ticker_x) tuples
    eg_threshold   : Engle-Granger p-value threshold

    Returns
    -------
    DataFrame with one row per pair and columns:
        ticker_y, ticker_x, correlation, eg_pvalue, johansen_trace, johansen_crit95,
        hurst, var_ratio, half_life, corr_stability, is_eg_coint, is_joh_coint
    """
    results = []

    # Config options
    from config import MIN_CORRELATION

    for ticker_y, ticker_x in candidate_pairs:
        if ticker_y not in prices.columns or ticker_x not in prices.columns:
            print(f"  [screen] Skipping {ticker_y}/{ticker_x} -- missing data")
            continue

        sy = prices[ticker_y].dropna()
        sx = prices[ticker_x].dropna()

        # Align
        common = sy.index.intersection(sx.index)
        sy = sy.loc[common]
        sx = sx.loc[common]

        if len(sy) < 100:
            continue

        # Correlation Pre-Filter
        corr = sy.corr(sx)
        if corr < MIN_CORRELATION:
            print(f"  [screen] Skipping {ticker_y}/{ticker_x} -- correlation too low ({corr:.2f} < {MIN_CORRELATION})")
            continue

        # Engle-Granger
        _, eg_pvalue, _ = coint(sy, sx)

        # Johansen
        joh = johansen_test(sy, sx)

        # Hedge ratio & spread
        hr = estimate_hedge_ratio(sy, sx)
        spread = compute_spread(sy, sx, hr["beta"], hr["alpha"])

        # Hurst
        h = hurst_exponent(spread)

        # Variance Ratio (Lag 5)
        vr = variance_ratio_test(spread, lag=5)

        # Half-life
        hl = half_life_mean_reversion(spread)

        # Correlation stability
        cs = correlation_stability(sy, sx)

        results.append({
            "ticker_y": ticker_y,
            "ticker_x": ticker_x,
            "correlation": float(corr),
            "eg_pvalue": float(eg_pvalue),
            "johansen_trace": joh["trace_stat"],
            "johansen_crit95": joh["crit_95"],
            "hurst": h,
            "var_ratio": vr,
            "half_life": hl,
            "corr_stability": cs,
            "is_eg_coint": eg_pvalue < eg_threshold,
            "is_joh_coint": joh["is_cointegrated"],
            "hedge_ratio": float(hr["beta"]),
            "r_squared": float(hr["r_squared"]),
        })

    df = pd.DataFrame(results)
    return df


def print_screening_report(screening_df: pd.DataFrame) -> None:
    """Pretty-print the pair screening results."""
    print("\n" + "=" * 85)
    print("  PAIR SCREENING RESULTS")
    print("=" * 85)
    print(f"  {'Pair':<10} {'Corr':>6} {'Joh Trace':>10} {'VR(5)':>7} {'Hurst':>7} "
          f"{'Half-life':>10} {'Corr Stab':>10} {'Joh?':>5}")
    print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*7} {'-'*7} "
          f"{'-'*10} {'-'*10} {'-'*5}")

    for _, row in screening_df.iterrows():
        pair = f"{row['ticker_y']}/{row['ticker_x']}"
        hl = f"{row['half_life']:.1f}d" if row['half_life'] < 9999 else "inf"
        print(
            f"  {pair:<10} "
            f"{row['correlation']:>6.2f} "
            f"{row['johansen_trace']:>10.2f} "
            f"{row['var_ratio']:>7.3f} "
            f"{row['hurst']:>7.3f} "
            f"{hl:>10} "
            f"{row['corr_stability']:>10.4f} "
            f"{'YES' if row['is_joh_coint'] else 'NO':>5}"
        )

    print("=" * 85 + "\n")
