"""
backtest.py -- Vectorized backtesting engine for pairs-trading strategies.

Supports:
    - Fixed or time-varying hedge ratios
    - Fractional position sizing (not just {-1, 0, +1})
    - Simple or realistic transaction cost models
"""

import numpy as np
import pandas as pd


def backtest_pair(
    prices_y: pd.Series,
    prices_x: pd.Series,
    signals: pd.Series,
    hedge_ratio: float | pd.Series = 1.0,
    cost_bps: float = 5.0,
    volumes_y: pd.Series | None = None,
    volumes_x: pd.Series | None = None,
    use_realistic_costs: bool = False,
    cost_config: dict | None = None,
) -> pd.DataFrame:
    """
    Vectorized PnL computation for a pairs strategy.

    The strategy trades a dollar-neutral spread:
        Long spread  (signal > 0):  buy Y, sell beta*X
        Short spread (signal < 0):  sell Y, buy beta*X

    Parameters
    ----------
    prices_y     : daily prices of asset Y
    prices_x     : daily prices of asset X
    signals      : position signals (int or float if position-sized)
    hedge_ratio  : beta -- float (static) or Series (time-varying)
    cost_bps     : simple cost per leg in basis points (fallback)
    volumes_y/x  : daily volumes (needed for realistic costs)
    use_realistic_costs : use the multi-component cost model
    cost_config  : dict of cost model parameters

    Returns
    -------
    DataFrame with columns:
        signal, return_y, return_x, spread_return, gross_return,
        trade_flag, cost, net_return, cumulative_return, equity
    """
    idx = signals.index
    ret_y = prices_y.reindex(idx).pct_change()
    ret_x = prices_x.reindex(idx).pct_change()

    # Handle time-varying hedge ratio
    if isinstance(hedge_ratio, pd.Series):
        hr = hedge_ratio.reindex(idx).ffill().bfill()
    else:
        hr = pd.Series(hedge_ratio, index=idx)

    # Gross exposure for dollar-neutral weighting (1 part Y, beta parts X)
    gross_exposure = 1.0 + hr.abs()

    # Spread return: normalized to gross capital base
    spread_return = (ret_y - hr * ret_x) / gross_exposure

    # Gross return: position * spread_return (position from previous day)
    prev_signals = signals.shift(1).fillna(0)
    gross_return = prev_signals * spread_return

    # Trade detection
    trade_flag = (signals.diff().abs() > 0).astype(int)

    # Transaction costs
    if use_realistic_costs and volumes_y is not None and volumes_x is not None:
        from src.cost_model import compute_total_cost
        cfg = cost_config or {}
        cost = compute_total_cost(
            prices_y=prices_y.reindex(idx),
            prices_x=prices_x.reindex(idx),
            volumes_y=volumes_y.reindex(idx),
            volumes_x=volumes_x.reindex(idx),
            signals=signals,
            hedge_ratio=hr,
            cost_config=cfg
        )
    else:
        # Simple bps cost per leg, normalized to capital base
        turnover_y = signals.diff().abs()
        turnover_x = (signals * hr).diff().abs()
        total_turnover = (turnover_y + turnover_x) / gross_exposure
        cost = total_turnover * (cost_bps / 10000.0)
        cost = cost.fillna(0)
        
    net_return = gross_return - cost

    # Build result
    result = pd.DataFrame({
        "signal": signals,
        "return_y": ret_y,
        "return_x": ret_x,
        "spread_return": spread_return,
        "gross_return": gross_return,
        "trade_flag": trade_flag,
        "cost": cost,
        "net_return": net_return,
    }, index=idx)

    result["cumulative_return"] = (1 + result["net_return"].fillna(0)).cumprod() - 1
    result["equity"] = (1 + result["net_return"].fillna(0)).cumprod()

    return result


def buy_and_hold_benchmark(prices_y: pd.Series) -> pd.DataFrame:
    """Simple buy-and-hold benchmark on asset Y for comparison."""
    ret = prices_y.pct_change().fillna(0)
    equity = (1 + ret).cumprod()

    return pd.DataFrame({
        "return": ret,
        "cumulative_return": equity - 1,
        "equity": equity,
    }, index=prices_y.index)
