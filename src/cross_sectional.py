"""
cross_sectional.py -- Cross-Sectional Statistical Arbitrage Engine
Orchestrates the PCA Factor extraction, S-score generation, and Portfolio construction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

from src.walk_forward import walk_forward_split
from src.pca_factors import fit_pca, compute_loadings, compute_residuals
from src.signals import compute_zscore, generate_signals
from src.meta_labeling import apply_triple_barrier, train_meta_model
from src.position_sizing import volatility_scaling, kelly_ml
from src.regime_filter import compute_adx, rolling_vr, compute_regime_mask
from src.backtest import backtest_pair
from src.performance import compute_metrics

def run_cross_sectional_stat_arb(
    prices: pd.DataFrame,
    num_folds: int = 4,
    purge_days: int = 20,
    embargo_days: int = 5,
    num_pca_factors: int = 15,
    zscore_lookback: int = 20,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    stop_loss_zscore: float = 4.0,
    position_sizing: str = "kelly_ml",
    target_vol: float = 0.10,
    vol_lookback: int = 60,
    use_meta_labeling: bool = True,
    tb_profit: float = 2.0,
    tb_loss: float = 2.0,
    tb_time: int = 20,
    use_regime: bool = True,
    regime_method: str = "hmm",
    hmm_components: int = 2,
    cost_bps: float = 5.0,
    trading_days: int = 252,
) -> Dict[str, Any]:
    """
    Execute Cross-Sectional PCA Statistical Arbitrage over Walk-Forward folds.
    """
    # 1. Compute daily returns for the whole universe
    returns = prices.pct_change().dropna(how='all')
    
    # 2. Walk-Forward splits
    splits = walk_forward_split(returns.index, num_folds, purge_days, embargo_days)
    fold_results = []
    
    portfolio_daily_returns = pd.Series(dtype=float)
    
    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold_num}/{num_folds} (Cross-Sectional PCA)")
        print(f"  Train: {train_idx[0].date()} -> {train_idx[-1].date()} ({len(train_idx)} days)")
        print(f"  Test:  {test_idx[0].date()} -> {test_idx[-1].date()} ({len(test_idx)} days)")
        print(f"{'=' * 60}")
        
        train_rets = returns.loc[train_idx]
        test_rets = returns.loc[test_idx]
        
        # Ensure dense matrix for PCA
        train_rets = train_rets.dropna(axis=1, thresh=len(train_rets) * 0.9).fillna(0)
        valid_tickers = train_rets.columns
        test_rets = test_rets[valid_tickers].fillna(0)
        
        print(f"  [PCA] Fitting {num_pca_factors} factors on {len(valid_tickers)} valid stocks...")
        pca_model, train_factors, scaler_mean, scaler_std = fit_pca(train_rets, num_factors=num_pca_factors)
        betas = compute_loadings(train_rets, train_factors)
        
        # Compute Train and Test residuals
        train_res_rets, train_res_prices = compute_residuals(train_rets, pca_model, scaler_mean, scaler_std, betas)
        test_res_rets, test_res_prices = compute_residuals(test_rets, pca_model, scaler_mean, scaler_std, betas)
        
        # We will accumulate positions for all valid stocks
        fold_positions = pd.DataFrame(index=test_idx, columns=valid_tickers).fillna(0.0)
        
        # Iterate over each stock to generate signals
        for ticker in valid_tickers:
            # Generate Z-scores (S-scores)
            train_z = compute_zscore(train_res_prices[ticker], lookback=zscore_lookback)
            test_z = compute_zscore(test_res_prices[ticker], lookback=zscore_lookback)
            
            # Regime Filter (using market factor or individual residual vol)
            train_regime, test_regime = None, None
            if use_regime:
                train_rm = compute_regime_mask(
                    train_res_prices[ticker], train_res_prices[ticker], 
                    method=regime_method, hmm_components=hmm_components
                )
                test_rm = compute_regime_mask(
                    test_res_prices[ticker], test_res_prices[ticker], 
                    method=regime_method, hmm_components=hmm_components
                )
                train_regime = train_rm
                test_regime = test_rm
                
            # Primary Signals
            train_sigs = generate_signals(train_z, entry_zscore, exit_zscore, stop_loss_zscore, regime_mask=train_regime)
            test_sigs = generate_signals(test_z, entry_zscore, exit_zscore, stop_loss_zscore, regime_mask=test_regime)
            
            # Expected Return Meta-Labeling
            expected_returns = pd.Series(0.0, index=test_idx)
            if use_meta_labeling:
                train_vol = train_res_rets[ticker].rolling(20, min_periods=5).std().fillna(0)
                train_adx = compute_adx(prices.loc[train_idx, ticker], period=14).fillna(0)
                train_vr = rolling_vr(train_res_prices[ticker], window=5).fillna(1.0)
                train_z_slope = train_z.diff(3).fillna(0)
                
                X_train = pd.DataFrame({
                    "vol": train_vol,
                    "adx": train_adx,
                    "vr": train_vr,
                    "zscore": train_z.fillna(0),
                    "z_slope": train_z_slope,
                })
                
                labels_df = apply_triple_barrier(
                    train_res_prices[ticker], train_sigs, 
                    profit_take_mult=tb_profit, stop_loss_mult=tb_loss, 
                    max_holding_period=tb_time, volatility=train_vol
                )
                
                if len(labels_df) > 5:
                    y_train_series = pd.Series(index=labels_df['entry_time'], data=labels_df['return'].values)
                    X_train_entries = X_train.loc[y_train_series.index]
                    meta_model = train_meta_model(X_train_entries, y_train_series)
                else:
                    meta_model = None
                    
                if meta_model is not None:
                    test_vol = test_res_rets[ticker].rolling(20, min_periods=5).std().fillna(0)
                    test_adx = compute_adx(prices.loc[test_idx, ticker], period=14).fillna(0)
                    test_vr = rolling_vr(test_res_prices[ticker], window=5).fillna(1.0)
                    test_z_slope = test_z.diff(3).fillna(0)
                    
                    X_test = pd.DataFrame({
                        "vol": test_vol,
                        "adx": test_adx,
                        "vr": test_vr,
                        "zscore": test_z.fillna(0),
                        "z_slope": test_z_slope,
                    })
                    
                    try:
                        preds = meta_model.predict(X_test)
                        expected_returns = pd.Series(preds, index=X_test.index)
                        # Filter negative EV
                        test_sigs = test_sigs * (expected_returns > 0)
                    except Exception:
                        pass
                        
            # Position Sizing
            if position_sizing == "kelly_ml" and use_meta_labeling:
                test_vol = test_res_rets[ticker].rolling(20, min_periods=5).std().fillna(0)
                weights = kelly_ml(expected_returns, test_vol, fraction=0.5)
                # Ensure weight logic applies properly (long if sig=1, short if sig=-1)
                test_pos = test_sigs * weights
            elif position_sizing == "vol_scaling":
                weights = volatility_scaling(test_res_rets[ticker], target_vol, vol_lookback, trading_days)
                # apply_position_sizing handles filling
                test_pos = test_sigs * weights.reindex(test_sigs.index, method="ffill").fillna(1.0)
            else:
                test_pos = test_sigs.astype(float)
                
            fold_positions[ticker] = test_pos

        # Compute Portfolio Returns for the fold
        # Portfolio Return = Sum of (Position * Residual Return) / Total Absolute Allocation
        # This assumes we are trading the idiosyncratic residuals (which means we are implicitly 
        # hedging out the PCA factors for each stock, so the actual trade is Stock - Beta*Factors).
        # We can just multiply position * residual_return.
        
        # Calculate daily gross return per ticker
        daily_pnl = fold_positions.shift(1) * test_res_rets
        
        # Subtract transaction costs (assuming cost_bps applied to change in position)
        pos_change = fold_positions.diff().fillna(0).abs()
        tx_costs = pos_change * (cost_bps / 10000.0)
        
        daily_net = daily_pnl - tx_costs
        
        # Equal capital allocation across the N active positions
        # If we have 30 stocks, we allocate 1/30th of capital to each stock's max leverage
        fold_portfolio_ret = daily_net.sum(axis=1) / len(valid_tickers)
        
        portfolio_daily_returns = pd.concat([portfolio_daily_returns, fold_portfolio_ret])
        
        print(f"  [Fold {fold_num}] Return: {fold_portfolio_ret.sum():.2%}  Max Active Positions: {(fold_positions != 0).sum(axis=1).max()}")

    # Compute overall metrics
    port_df = portfolio_daily_returns.to_frame(name="net_return")
    port_df["equity"] = (1 + port_df["net_return"]).cumprod()
    metrics = compute_metrics(port_df, trading_days=trading_days)
    print(f"\n{'=' * 60}")
    print(f"  CROSS-SECTIONAL STATARB PORTFOLIO METRICS")
    print(f"{'=' * 60}")
    print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio: {metrics['sortino_ratio']:.4f}")
    print(f"  Max Drawdown:  {metrics['max_drawdown']:.2%}")
    print(f"  Total Return:  {metrics['total_return']:.2%}")
    print(f"  Win Rate:      {metrics['win_rate']:.2%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.4f}")

    return {
        "portfolio_returns": portfolio_daily_returns,
        "metrics": metrics
    }
