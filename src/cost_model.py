"""
cost_model.py -- Realistic multi-component transaction cost model.

Components:
    1. Commission:  fixed per-share cost
    2. Bid-ask spread:  estimated from volume (Hasbrouck model)
    3. Slippage:  fraction of daily volatility * sqrt(trade fraction)
    4. Market impact:  simplified Almgren-Chriss square-root model

Reference: Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions"
"""

import numpy as np
import pandas as pd


def estimate_bid_ask_spread(
    prices: pd.Series,
    volumes: pd.Series,
    model: str = "hasbrouck",
    fixed_bps: float = 2.0,
) -> pd.Series:
    """
    Estimate daily bid-ask spread in bps.

    Models:
        "fixed"     : constant spread
        "hasbrouck" : spread ~ c / sqrt(dollar_volume)
    """
    if model == "fixed":
        return pd.Series(fixed_bps, index=prices.index)

    # Hasbrouck model: spread inversely proportional to sqrt(dollar volume)
    dollar_volume = prices * volumes
    # Calibrated constant (empirical, typical for US large-cap)
    c = 50.0
    spread_bps = c / np.sqrt(dollar_volume.clip(lower=1))
    # Floor at 0.5 bps, cap at 20 bps
    spread_bps = spread_bps.clip(lower=0.5, upper=20.0)
    return spread_bps


def compute_slippage(
    prices: pd.Series,
    volumes: pd.Series,
    trade_size_dollars: float = 100_000,
    slippage_factor: float = 0.1,
    lookback: int = 20,
) -> pd.Series:
    """
    Estimate slippage in bps using square-root market impact model.

    slippage = factor * daily_vol * sqrt(trade_fraction)

    where trade_fraction = trade_size / daily_dollar_volume
    """
    daily_vol = prices.pct_change().rolling(lookback).std() * 100  # in %
    dollar_volume = prices * volumes
    trade_fraction = trade_size_dollars / dollar_volume.clip(lower=1)

    # Square-root impact
    slippage_pct = slippage_factor * daily_vol * np.sqrt(trade_fraction)
    slippage_bps = slippage_pct * 100  # convert % to bps
    slippage_bps = slippage_bps.clip(lower=0, upper=50)
    return slippage_bps


def compute_commission_bps(
    prices: pd.Series,
    commission_per_share: float = 0.005,
) -> pd.Series:
    """Convert per-share commission to bps relative to price."""
    return (commission_per_share / prices) * 10_000


def compute_total_cost(
    prices_y: pd.Series,
    prices_x: pd.Series,
    volumes_y: pd.Series | None = None,
    volumes_x: pd.Series | None = None,
    trade_flags: pd.Series | None = None,
    commission_per_share: float = 0.005,
    spread_model: str = "hasbrouck",
    fixed_spread_bps: float = 2.0,
    slippage_factor: float = 0.1,
    trade_size_dollars: float = 100_000,
    fallback_bps: float = 5.0,
) -> pd.Series:
    """
    Compute total transaction cost per trade in fractional terms (not bps).

    Returns a Series of cost fractions aligned with the price index.
    Cost is only applied on days where trade_flags == 1.

    If volume data is unavailable, falls back to fixed bps model.
    """
    idx = prices_y.index

    if volumes_y is None or volumes_x is None:
        # Fallback to simple fixed cost model
        cost_per_trade = 2 * (fallback_bps / 10_000)
        if trade_flags is not None:
            return trade_flags * cost_per_trade
        return pd.Series(cost_per_trade, index=idx)

    # -- Commission (both legs)
    comm_y = compute_commission_bps(prices_y, commission_per_share)
    comm_x = compute_commission_bps(prices_x, commission_per_share)
    total_commission = comm_y + comm_x  # bps

    # -- Bid-ask spread (both legs)
    spread_y = estimate_bid_ask_spread(prices_y, volumes_y, spread_model, fixed_spread_bps)
    spread_x = estimate_bid_ask_spread(prices_x, volumes_x, spread_model, fixed_spread_bps)
    total_spread = (spread_y + spread_x) / 2  # average half-spread per leg, 2 legs

    # -- Slippage (both legs)
    slip_y = compute_slippage(prices_y, volumes_y, trade_size_dollars, slippage_factor)
    slip_x = compute_slippage(prices_x, volumes_x, trade_size_dollars, slippage_factor)
    total_slippage = slip_y + slip_x  # bps

    # Total in bps
    total_bps = total_commission + total_spread + total_slippage

    # Convert to fraction
    total_fraction = total_bps / 10_000

    if trade_flags is not None:
        return trade_flags * total_fraction

    return total_fraction
