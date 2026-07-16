"""
portfolio_optimizer.py -- Allocates capital across multiple pairs in a portfolio.

Supports:
    - Equal Weighting
    - Risk Parity (Inverse Volatility) weighting
"""

import numpy as np
import pandas as pd


def equal_weight_portfolio(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Allocate equal capital to each pair.
    
    Parameters
    ----------
    returns_df : DataFrame where each column is the daily net return of a pair.
    
    Returns
    -------
    DataFrame with portfolio 'net_return' and 'equity'.
    """
    num_pairs = returns_df.shape[1]
    if num_pairs == 0:
        return pd.DataFrame({"net_return": [], "equity": []})
        
    weights = np.ones(num_pairs) / num_pairs
    
    # Portfolio return is the dot product of weights and pair returns
    port_return = returns_df.dot(weights)
    
    # Portfolio equity
    equity = (1 + port_return.fillna(0)).cumprod()
    
    return pd.DataFrame({
        "net_return": port_return,
        "equity": equity
    }, index=returns_df.index)


def risk_parity_portfolio(returns_df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Allocate capital based on inverse realized volatility (Risk Parity).
    Pairs with lower volatility get higher weight.
    
    Parameters
    ----------
    returns_df : DataFrame where each column is the daily net return of a pair.
    window     : Rolling window to estimate volatility.
    
    Returns
    -------
    DataFrame with portfolio 'net_return' and 'equity'.
    """
    if returns_df.empty:
        return pd.DataFrame({"net_return": [], "equity": []})
        
    # Calculate rolling standard deviation (volatility)
    rolling_vol = returns_df.rolling(window=window, min_periods=10).std()
    
    # Fallback to expanding vol if early in the series
    expanding_vol = returns_df.expanding(min_periods=2).std()
    rolling_vol = rolling_vol.fillna(expanding_vol)
    
    # Handle zeros to avoid division by zero
    rolling_vol = rolling_vol.replace(0, 1e-6)
    
    # Inverse volatility weights
    inv_vol = 1.0 / rolling_vol
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    
    # Shift weights by 1 day to avoid look-ahead bias (using yesterday's vol for today's allocation)
    weights = weights.shift(1).fillna(1.0 / returns_df.shape[1])
    
    # Portfolio return
    port_return = (returns_df * weights).sum(axis=1)
    
    # Portfolio equity
    equity = (1 + port_return.fillna(0)).cumprod()
    
    return pd.DataFrame({
        "net_return": port_return,
        "equity": equity
    }, index=returns_df.index)
