"""
signals.py -- Signal generation for mean-reversion trading.

Supports:
    - Rolling z-score signals (original)
    - Regime-filtered signals (new)
    - Position-sized signals (new)
"""

import numpy as np
import pandas as pd


def compute_zscore(spread: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Compute the rolling z-score of the spread.

        z_t = (spread_t - mu_rolling) / sigma_rolling

    where mu and sigma are computed over the trailing *lookback* window.
    """
    rolling_mean = spread.rolling(window=lookback).mean()
    rolling_std  = spread.rolling(window=lookback).std(ddof=1)

    zscore = (spread - rolling_mean) / rolling_std
    return zscore


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss_threshold: float = 4.0,
    regime_mask: pd.Series | None = None,
) -> pd.Series:
    """
    Generate position signals from z-score with state tracking.

    Rules:
      - Enter SHORT spread when z > +entry_threshold  -> signal = -1
      - Enter LONG  spread when z < -entry_threshold  -> signal = +1
      - Exit (flatten) when |z| < exit_threshold      -> signal =  0
      - Stop-loss exit when |z| > stop_loss_threshold  -> signal =  0
      - Regime filter: force signal = 0 when regime_mask = False

    Parameters
    ----------
    regime_mask : optional boolean Series (True = OK to trade)

    Returns
    -------
    pd.Series of {-1, 0, +1} aligned with zscore index.
    """
    signals = pd.Series(0, index=zscore.index, dtype=int)
    position = 0
    stopped_out = False  # Track if we are currently stopped out

    for i in range(len(zscore)):
        z = zscore.iloc[i]

        if np.isnan(z):
            signals.iloc[i] = 0
            continue

        # Regime filter: force flat if regime is unfavorable
        if regime_mask is not None:
            date = zscore.index[i]
            if date in regime_mask.index and not regime_mask.loc[date]:
                position = 0
                signals.iloc[i] = 0
                continue

        # Check if we can reset the stopped_out flag
        if stopped_out:
            # Only allow re-entry if the z-score has returned inside the stop-loss bounds
            if abs(z) < stop_loss_threshold:
                stopped_out = False
            else:
                # Still stopped out, stay flat
                signals.iloc[i] = 0
                continue

        # Stop-loss
        if abs(z) > stop_loss_threshold and position != 0:
            position = 0
            stopped_out = True
            signals.iloc[i] = 0
            continue

        # Exit logic
        if position == -1 and z < exit_threshold:
            position = 0
        elif position == 1 and z > -exit_threshold:
            position = 0

        # Entry logic (only if we didn't just exit or stop out, and spread is not blown out)
        if position == 0 and not stopped_out and abs(z) < stop_loss_threshold:
            if z > entry_threshold:
                position = -1    # spread is rich -> short spread
            elif z < -entry_threshold:
                position = 1     # spread is cheap -> long spread

        signals.iloc[i] = position

    return signals


def signals_summary(signals: pd.Series) -> dict:
    """Return a quick summary of the generated signals."""
    trades = (signals.diff().abs() > 0).sum()
    long_days  = (signals == 1).sum()
    short_days = (signals == -1).sum()
    flat_days  = (signals == 0).sum()

    return {
        "total_signal_changes": int(trades),
        "long_days": int(long_days),
        "short_days": int(short_days),
        "flat_days": int(flat_days),
    }
