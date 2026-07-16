"""
position_sizing.py -- Position sizing strategies for pairs trading.

Methods:
    1. Fixed:          signal in {-1, 0, +1}
    2. Kelly fraction: optimal fraction based on edge / odds
    3. Vol scaling:    target a fixed annualized volatility
"""

import numpy as np
import pandas as pd


def kelly_fraction(
    returns: pd.Series,
    fraction: float = 0.5,
) -> float:
    """
    Compute the (half-)Kelly fraction from historical returns.

    Full Kelly:  f* = (p * b - q) / b
        p = win probability
        q = 1 - p
        b = avg_win / avg_loss

    We use half-Kelly (fraction=0.5) by default for safety.
    """
    active = returns[returns != 0].dropna()
    if len(active) < 10:
        return 1.0  # Not enough data, use full position

    wins = active[active > 0]
    losses = active[active < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 1.0

    p = len(wins) / len(active)
    q = 1 - p
    b = abs(wins.mean() / losses.mean())

    kelly = (p * b - q) / b if b > 0 else 0

    # Apply fraction (half-Kelly) and clamp
    sized = kelly * fraction
    return float(np.clip(sized, 0.0, 2.0))


def kelly_ml(
    expected_returns: pd.Series,
    volatility: pd.Series,
    fraction: float = 0.5,
    max_leverage: float = 3.0,
) -> pd.Series:
    """
    Compute position sizes using continuous Kelly fraction from ML expected returns.
    Kelly = Expected Return / Variance
    
    If expected return is near zero or negative (relative to signal direction), size is 0.
    """
    variance = volatility ** 2
    kelly = expected_returns / variance.clip(lower=1e-6)
    
    # Kelly can be negative if expected return is opposite to signal (handled during sizing later)
    # We apply fraction and clamp
    sized = kelly * fraction
    return sized.clip(lower=-max_leverage, upper=max_leverage)


def volatility_scaling(
    spread_returns: pd.Series,
    target_vol: float = 0.10,
    lookback: int = 20,
    trading_days: int = 252,
    max_leverage: float = 3.0,
) -> pd.Series:
    """
    Compute position size multipliers to target a fixed annualized volatility.

        weight_t = target_vol / realized_vol_t

    Realized vol is estimated from a rolling window of spread returns.
    """
    # Annualized rolling volatility
    rolling_vol = spread_returns.rolling(window=lookback).std() * np.sqrt(trading_days)

    # Position multiplier
    weights = target_vol / rolling_vol.clip(lower=1e-6)

    # Clamp to prevent extreme leverage
    weights = weights.clip(lower=0.1, upper=max_leverage)

    # Fill NaN with 1.0 (full position during warm-up)
    weights = weights.fillna(1.0)

    return weights


def apply_position_sizing(
    signals: pd.Series,
    weights: pd.Series | float,
) -> pd.Series:
    """
    Scale signals by position size weights.

    Input signals are {-1, 0, +1}.
    Output signals are float (e.g., -1.5, 0, 0.8).
    """
    if isinstance(weights, (int, float)):
        return signals * weights
    return signals * weights.reindex(signals.index, method="ffill").fillna(1.0)
