"""
regime_filter.py -- Market regime detection to avoid trading in adverse conditions.

Filters:
    1. ADX (Average Directional Index) -- avoid trending markets
    2. VIX -- avoid high-volatility regimes
    3. Rolling Hurst exponent -- avoid non-mean-reverting regimes

Only trade when ALL conditions are favorable (conjunction).
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm


def compute_adx(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute the Average Directional Index (ADX) from price data.

    Since we only have close prices (not OHLC), we use a proxy based on
    price volatility directional movement.

    ADX < 20-25 : range-bound (favorable for mean-reversion)
    ADX > 25    : trending (unfavorable)
    """
    # Approximate +DM and -DM from close-to-close moves
    diff = prices.diff()
    plus_dm  = diff.clip(lower=0)
    minus_dm = (-diff).clip(lower=0)

    # True range proxy (close-to-close range)
    tr = prices.diff().abs()

    # Smoothed averages
    atr = tr.rolling(period).mean()
    plus_di  = 100 * (plus_dm.rolling(period).mean() / atr.clip(lower=1e-8))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.clip(lower=1e-8))

    # DX = |+DI - -DI| / (+DI + -DI)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-8) * 100

    # ADX = smoothed DX
    adx = dx.rolling(period).mean()

    return adx


def rolling_vr(
    spread: pd.Series,
    window: int = 60,
    vr_lag: int = 5,
) -> pd.Series:
    """
    Compute rolling Variance Ratio on the spread.

    VR < 1.0 : mean-reverting (trade)
    VR > 1.0 : trending (don't trade)
    """
    from src.pair_selection import variance_ratio_test

    n = len(spread)
    vr_values = pd.Series(np.nan, index=spread.index)

    for i in range(window, n):
        subset = spread.iloc[i - window : i]
        vr = variance_ratio_test(subset, lag=vr_lag)
        vr_values.iloc[i] = vr

    # Backfill early NaNs
    vr_values = vr_values.bfill()
    return vr_values


def compute_hmm_regime(
    spread: pd.Series,
    returns: pd.Series | None = None,
    n_components: int = 2
) -> pd.Series:
    """
    Detect latent market regimes using a Gaussian Hidden Markov Model.
    
    Parameters
    ----------
    spread : pd.Series of the spread.
    returns : pd.Series of market returns (e.g. SPY) to help HMM identify broad regimes.
    n_components : Number of hidden states (e.g., 2 = Bull/Bear or Low/High Vol).
    
    Returns
    -------
    pd.Series of the predicted hidden state (0 to n_components-1) for each day.
    """
    if len(spread) < 20:
        return pd.Series(0, index=spread.index)
        
    spread_vol = spread.pct_change().fillna(0).rolling(window=10, min_periods=1).std().fillna(0)
    
    if returns is not None:
        X = np.column_stack([spread_vol.values, returns.fillna(0).values])
    else:
        X = spread_vol.values.reshape(-1, 1)
        
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42)
    try:
        model.fit(X)
        states = model.predict(X)
    except Exception as e:
        print(f"  [warn] HMM fit failed: {e}")
        states = np.zeros(len(spread))
        
    return pd.Series(states, index=spread.index)


def compute_regime_mask(
    prices_y: pd.Series,
    spread: pd.Series,
    vix: pd.Series | None = None,
    adx_threshold: float = 25.0,
    adx_period: int = 14,
    vix_threshold: float = 30.0,
    vr_window: int = 60,
    vr_threshold: float = 1.0,
    use_adx: bool = True,
    use_vix: bool = True,
    use_vr: bool = True,
    method: str = "static",
    hmm_components: int = 2,
    market_returns: pd.Series | None = None,
) -> pd.Series:
    """
    Compute a boolean mask where True means the regime supports mean-reversion trading.
    Supports either static rules (ADX/VIX/VR) or dynamic HMM regime detection.
    """
    # 1) If HMM method
    if method == "hmm":
        states = compute_hmm_regime(spread, market_returns, n_components=hmm_components)
        
        # We need to determine which state is the "mean-reverting / tradable" state.
        # Usually, stat arb works best in low volatility, sideways markets.
        # We find the state with the lowest spread volatility.
        vol_by_state = spread.pct_change().groupby(states).std()
        best_state = vol_by_state.idxmin()
        
        mask = (states == best_state)
        
        pass_pct = mask.mean() * 100
        print(f"  [regime] HMM Filter (State {best_state}): {pass_pct:.1f}% of days pass")
        
        return mask

    # 2) Static Rule-based method
    idx = spread.index
    mask = pd.Series(True, index=idx)
    # ADX filter
    if use_adx:
        adx = compute_adx(prices_y.reindex(idx), period=adx_period)
        adx_ok = adx < adx_threshold
        mask = mask & adx_ok.fillna(True)
        pct = adx_ok.mean() * 100
        print(f"  [regime] ADX filter: {pct:.1f}% of days pass (threshold={adx_threshold})")

    # VIX filter
    if use_vix and vix is not None:
        vix_aligned = vix.reindex(idx, method="ffill")
        vix_ok = vix_aligned < vix_threshold
        mask = mask & vix_ok.fillna(True)
        pct = vix_ok.mean() * 100
        print(f"  [regime] VIX filter: {pct:.1f}% of days pass (threshold={vix_threshold})")

    # VR filter
    if use_vr:
        r_vr = rolling_vr(spread, window=vr_window)
        vr_ok = r_vr < vr_threshold
        mask = mask & vr_ok.fillna(True)
        pct = vr_ok.mean() * 100
        print(f"  [regime] VR filter: {pct:.1f}% of days pass (threshold={vr_threshold})")

    total_pct = mask.mean() * 100
    print(f"  [regime] Combined: {total_pct:.1f}% of days pass all filters")

    return mask
