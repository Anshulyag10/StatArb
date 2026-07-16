"""
meta_labeling.py -- Implements Lopez de Prado's Triple Barrier Method and Meta-Labeling.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


def apply_triple_barrier(
    prices: pd.Series,
    signals: pd.Series,
    profit_take_mult: float = 2.0,
    stop_loss_mult: float = 2.0,
    max_holding_period: int = 20,
    volatility: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Apply the Triple Barrier Method to label trades.
    
    Parameters
    ----------
    prices : pd.Series of asset/spread prices.
    signals : pd.Series of primary signals (1 for long, -1 for short, 0 for flat).
              We only care about the entry signals (where signal changes from 0 to 1/-1).
    profit_take_mult : Multiplier for volatility to set upper barrier.
    stop_loss_mult : Multiplier for volatility to set lower barrier.
    max_holding_period : Maximum number of days to hold before hitting the time barrier.
    volatility : Daily rolling volatility of the asset/spread.
    
    Returns
    -------
    DataFrame containing 'entry_time', 'exit_time', 'return', and 'label'.
    Label is 1 if hit profit barrier, -1 if hit stop loss, 0 if hit time barrier (or expired).
    """
    if volatility is None:
        # Default to 20-day standard deviation of returns if not provided
        returns = prices.pct_change().fillna(0)
        volatility = returns.rolling(20, min_periods=5).std().fillna(returns.std())
    
    # Find entry points (where signal changes from 0)
    sig_diff = signals.diff()
    # Long entries: signal was 0, now 1
    long_entries = signals.index[(signals == 1) & (sig_diff == 1)]
    # Short entries: signal was 0, now -1
    short_entries = signals.index[(signals == -1) & (sig_diff == -1)]
    
    entries = sorted(list(long_entries) + list(short_entries))
    
    labels = []
    
    for entry in entries:
        side = signals.loc[entry]
        idx_pos = prices.index.get_loc(entry)
        
        # Determine barriers
        vol_t = volatility.loc[entry]
        price_t = prices.loc[entry]
        
        # Absolute barriers (assuming prices are a mean-reverting spread)
        # If it's a spread, the volatility might be absolute spread vol, not pct.
        # We assume volatility is in the same units as the price for a spread.
        ub = price_t + (profit_take_mult * vol_t) if side == 1 else price_t - (profit_take_mult * vol_t)
        lb = price_t - (stop_loss_mult * vol_t) if side == 1 else price_t + (stop_loss_mult * vol_t)
        
        # Time barrier index
        tb_idx = min(idx_pos + max_holding_period, len(prices) - 1)
        tb_time = prices.index[tb_idx]
        
        # Slice future path up to time barrier
        path = prices.iloc[idx_pos+1 : tb_idx+1]
        
        label = 0  # Default to time barrier
        exit_time = tb_time
        ret = 0.0
        
        for t, pt in path.items():
            if side == 1:
                if pt >= ub:
                    label = 1
                    exit_time = t
                    ret = pt - price_t
                    break
                elif pt <= lb:
                    label = -1
                    exit_time = t
                    ret = pt - price_t
                    break
            else: # side == -1
                if pt <= ub: # Actually for short, lower price is profit
                    label = 1
                    exit_time = t
                    ret = price_t - pt
                    break
                elif pt >= lb:
                    label = -1
                    exit_time = t
                    ret = price_t - pt
                    break
        
        # If no barrier hit, calculate return at time barrier
        if label == 0:
            pt = prices.loc[tb_time]
            ret = (pt - price_t) if side == 1 else (price_t - pt)
            # Optionally label 1 if return > 0, 0 otherwise. 
            # Standard Lopez de Prado labels based on sign of return at time barrier if we just want binary.
            label = 1 if ret > 0 else 0
            
        labels.append({
            "entry_time": entry,
            "exit_time": exit_time,
            "side": side,
            "return": ret,
            "label": label
        })
        
    return pd.DataFrame(labels)


def train_meta_model(
    features_df: pd.DataFrame, 
    labels_series: pd.Series
) -> xgb.XGBRegressor:
    """
    Train an XGBoost Meta-Labeling model for Expected Return.
    The model predicts the continuous expected return of a primary signal.
    
    Parameters
    ----------
    features_df : DataFrame of features at entry time (e.g. Volatility, VIX, Half-life).
    labels_series : Series of continuous return labels.
    
    Returns
    -------
    Trained XGBRegressor model.
    """
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    
    if len(features_df) > 10: # Only train if enough samples
        model.fit(features_df, labels_series)
        return model
    
    return None
