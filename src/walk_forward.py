"""
walk_forward.py -- Walk-forward validation with all v2 features.

Each fold:
    1. Estimate hedge ratio (Kalman/Rolling OLS/OLS) on training data
    2. Fit OU process on training spread
    3. Optionally optimize thresholds on training data
    4. Compute regime mask on test data
    5. Generate signals with regime filter
    6. Apply position sizing
    7. Backtest with cost model
    8. Compute full metrics suite
"""

import numpy as np
import pandas as pd

from src.cointegration import engle_granger_test, adf_test, johansen_test
from src.hedge_ratio import (
    compute_hedge_ratio, compute_spread_dynamic, estimate_hedge_ratio, compute_spread,
)
from src.ou_process import fit_ou, should_trade_ou
from src.signals import compute_zscore, generate_signals
from src.backtest import backtest_pair
from src.performance import compute_metrics
from src.position_sizing import volatility_scaling, apply_position_sizing, kelly_ml
from src.meta_labeling import apply_triple_barrier, train_meta_model
from src.regime_filter import compute_adx, rolling_vr, compute_regime_mask
from src.threshold_optimizer import optimize_thresholds
from src.ou_process import fit_ou, should_trade_ou, ou_zscore


def walk_forward_split(
    index: pd.DatetimeIndex,
    num_folds: int = 4,
    purge_days: int = 20,
    embargo_days: int = 5,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Create expanding-window walk-forward splits with Purging and Embargo.
    
    Purging: Drops the first `purge_days` of the test set to prevent rolling 
             features (like z-score) from leaking training data.
    Embargo: Adds an extra `embargo_days` gap between train and test.
    """
    n = len(index)
    block_size = n // (num_folds + 1)
    splits = []

    for k in range(num_folds):
        train_end = (k + 1) * block_size
        
        # Test start is shifted by embargo + purge to prevent leakage
        test_start = train_end + embargo_days + purge_days
        
        # If the gap pushes us past the end of the data, break
        if test_start >= n:
            break
            
        test_end = min(test_start + block_size, n)

        train_idx = index[:train_end]
        test_idx  = index[test_start:test_end]
        
        # Only append if we have enough test data
        if len(test_idx) > 20:
            splits.append((train_idx, test_idx))

    return splits


def run_walk_forward(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    num_folds: int = 4,
    hedge_method: str = "kalman",
    hedge_kwargs: dict = None,
    zscore_lookback: int = 20,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    stop_loss_zscore: float = 4.0,
    cost_bps: float = 5.0,
    eg_pvalue: float = 0.05,
    adf_pvalue: float = 0.05,
    trading_days: int = 252,
    # v2 features
    use_ou_filter: bool = False,
    ou_hl_min: float = 5,
    ou_hl_max: float = 120,
    regime_mask: pd.Series | None = None,
    position_sizing: str = "fixed",
    target_vol: float = 0.10,
    vol_lookback: int = 20,
    volumes_y: pd.Series | None = None,
    volumes_x: pd.Series | None = None,
    use_realistic_costs: bool = False,
    cost_config: dict | None = None,
    benchmark_returns: pd.Series | None = None,
    # Purged CV parameters
    purge_days: int = 20,
    embargo_days: int = 5,
    # Meta Labeling
    use_meta_labeling: bool = False,
    tb_profit: float = 2.0,
    tb_loss: float = 2.0,
    tb_time: int = 20,
    # Optuna / HMM params for internal fold optimization
    optimize_thresholds_flag: bool = False,
    opt_method: str = "bayesian",
    opt_trials: int = 15,
    optimize_lookback: bool = False,
    optimize_stoploss: bool = False,
    use_regime: bool = False,
    regime_method: str = "static",
    hmm_components: int = 2,
    market_returns: pd.Series | None = None,
    vix: pd.Series | None = None,
) -> list[dict]:
    """
    Execute the full walk-forward validation pipeline with v2 features.

    Returns list of fold result dicts.
    """
    if hedge_kwargs is None:
        hedge_kwargs = {}

    splits = walk_forward_split(prices.index, num_folds, purge_days, embargo_days)
    fold_results = []

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold_num}/{num_folds}")
        print(f"  Train: {train_idx[0].date()} -> {train_idx[-1].date()}  "
              f"({len(train_idx)} days)")
        print(f"  Test:  {test_idx[0].date()} -> {test_idx[-1].date()}  "
              f"({len(test_idx)} days)")
        print(f"{'=' * 60}")

        # Training data
        train_y = prices.loc[train_idx, ticker_y]
        train_x = prices.loc[train_idx, ticker_x]

        # Test data
        test_y = prices.loc[test_idx, ticker_y]
        test_x = prices.loc[test_idx, ticker_x]

        # -- 1. Hedge ratio (train) --
        hedge_df = compute_hedge_ratio(train_y, train_x, method=hedge_method, **hedge_kwargs)
        avg_beta = hedge_df["beta"].iloc[-1]
        avg_alpha = hedge_df["alpha"].iloc[-1]
        print(f"  Hedge ratio ({hedge_method}): beta={avg_beta:.6f}, alpha={avg_alpha:.6f}")

        # -- 2. Cointegration check (train) --
        eg = engle_granger_test(train_y, train_x, p_threshold=eg_pvalue)
        train_spread_static = compute_spread(train_y, train_x, avg_beta, avg_alpha)
        adf = adf_test(train_spread_static, p_threshold=adf_pvalue)

        print(f"  EG p-value: {eg['p_value']:.6f}  "
              f"({'YES' if eg['is_cointegrated'] else 'NO'})")
        print(f"  ADF p-value: {adf['p_value']:.6f}  "
              f"({'YES' if adf['is_stationary'] else 'NO'})")

        # -- 2.2 OU process (train spread) --
        ou_ok = True
        ou_params = None
        if use_ou_filter:
            ou_params = fit_ou(train_spread_static)
            tradable, reason = should_trade_ou(ou_params, ou_hl_min, ou_hl_max)
            print(f"  OU: {reason}")
            ou_ok = tradable

        # -- 2.5 Compute test spread --
        test_hedge_df = compute_hedge_ratio(
            pd.concat([train_y, test_y]),
            pd.concat([train_x, test_x]),
            method=hedge_method,
            **hedge_kwargs,
        )
        test_hedge_df = test_hedge_df.loc[test_idx]
        test_spread = compute_spread_dynamic(test_y, test_x, test_hedge_df)
        test_beta = test_hedge_df["beta"].mean()

        # -- 3. HMM / Regime Mask (Train & Test) --
        test_regime = None
        if use_regime:
            train_mr = market_returns.reindex(train_idx) if market_returns is not None else None
            train_v = vix.reindex(train_idx) if vix is not None else None
            # Fit/Predict on Train
            train_rm = compute_regime_mask(
                train_y, train_spread_static, vix=train_v,
                method=regime_method, hmm_components=hmm_components,
                market_returns=train_mr
            )
            # Predict on Test
            test_mr = market_returns.reindex(test_idx) if market_returns is not None else None
            test_v = vix.reindex(test_idx) if vix is not None else None
            test_regime = compute_regime_mask(
                test_y, test_spread, vix=test_v,
                method=regime_method, hmm_components=hmm_components,
                market_returns=test_mr
            )

        # -- 4. Threshold Optimization (Train) --
        fold_entry_z = entry_zscore
        fold_exit_z = exit_zscore
        fold_lookback = zscore_lookback
        fold_sl = stop_loss_zscore

        if optimize_thresholds_flag:
            print(f"  [Threshold Opt] Optimizing parameters for fold {fold_num}...")
            opt_result = optimize_thresholds(
                train_y, train_x, hedge_df.loc[train_idx],
                method=opt_method, n_trials=opt_trials,
                zscore_lookback=zscore_lookback, stop_loss=stop_loss_zscore,
                cost_bps=cost_bps, num_cv_folds=2,
                optimize_lookback=optimize_lookback, optimize_stoploss=optimize_stoploss,
            )
            fold_entry_z = opt_result["best_entry"]
            fold_exit_z = opt_result["best_exit"]
            if optimize_lookback:
                fold_lookback = opt_result.get("best_lookback", zscore_lookback)
            if optimize_stoploss:
                fold_sl = opt_result.get("best_stop_loss", stop_loss_zscore)

        # -- 4.5 Cointegration Gate --
        # Skip folds where the pair is NOT cointegrated (no edge to trade)
        if not eg['is_cointegrated'] and not adf['is_stationary']:
            print(f"  [SKIP] Pair not cointegrated in this fold (EG p={eg['p_value']:.4f}, ADF p={adf['p_value']:.4f})")
            continue

        # -- 5. Primary Signals (Test) --
        # Use rolling z-score — OU z-score is unstable because mu drifts between train/test
        zscore = compute_zscore(test_spread, lookback=fold_lookback)
        
        sigs = generate_signals(
            zscore,
            entry_threshold=fold_entry_z,
            exit_threshold=fold_exit_z,
            stop_loss_threshold=fold_sl,
            regime_mask=test_regime,
        )

        # -- 6. Expected Return Regression (Meta-Labeling) --
        expected_returns = pd.Series(0.0, index=test_idx)
        
        if use_meta_labeling:
            train_spread_dyn = compute_spread_dynamic(train_y, train_x, hedge_df.loc[train_idx])
            train_z = compute_zscore(train_spread_dyn, lookback=fold_lookback)
            train_sigs = generate_signals(train_z, fold_entry_z, fold_exit_z, fold_sl, regime_mask=train_rm if use_regime else None)
            
            # Features: Spread Vol, ADX, VR, Z-score, Z-slope
            train_vol = train_spread_dyn.pct_change().fillna(0).rolling(20, min_periods=5).std().fillna(0)
            train_adx = compute_adx(train_y, period=14).fillna(0)
            train_vr = rolling_vr(train_spread_dyn, window=5).fillna(1.0)
            train_z_slope = train_z.diff(3).fillna(0)
            
            X_train = pd.DataFrame({
                "vol": train_vol,
                "adx": train_adx,
                "vr": train_vr,
                "zscore": train_z.fillna(0),
                "z_slope": train_z_slope,
            })
            
            labels_df = apply_triple_barrier(
                train_spread_dyn, train_sigs, 
                profit_take_mult=tb_profit, stop_loss_mult=tb_loss, 
                max_holding_period=tb_time, volatility=train_vol
            )
            
            if len(labels_df) > 5:
                # Regress on ACTUAL continuous return, not binary label
                y_train_series = pd.Series(index=labels_df['entry_time'], data=labels_df['return'].values)
                X_train_entries = X_train.loc[y_train_series.index]
                
                meta_model = train_meta_model(X_train_entries, y_train_series)
            else:
                meta_model = None
                
            if meta_model is not None:
                test_vol = test_spread.pct_change().fillna(0).rolling(20, min_periods=5).std().fillna(0)
                test_adx = compute_adx(test_y, period=14).fillna(0)
                test_vr = rolling_vr(test_spread, window=5).fillna(1.0)
                test_z_slope = zscore.diff(3).fillna(0)
                
                X_test = pd.DataFrame({
                    "vol": test_vol,
                    "adx": test_adx,
                    "vr": test_vr,
                    "zscore": zscore.fillna(0),
                    "z_slope": test_z_slope,
                })
                
                try:
                    preds = meta_model.predict(X_test)
                    expected_returns = pd.Series(preds, index=X_test.index)
                    
                    # Filter out negative expectation trades
                    ml_mask = expected_returns > 0
                    sigs = sigs * ml_mask
                    print(f"  [meta-labeling] XGBoost Filter: {ml_mask.mean()*100:.1f}% of test days positive EV")
                except Exception as e:
                    print(f"  [warn] Meta-labeling prediction failed: {e}")
            else:
                print("  [meta-labeling] Not enough train labels. Skipping ML filter.")

        # Soft-scale signals based on OU quality instead of binary kill
        if use_ou_filter and not ou_ok:
            # Instead of zeroing all signals, scale down by a confidence factor
            hl = ou_params.get('half_life', float('inf')) if ou_params else float('inf')
            theta = ou_params.get('theta', 0.0) if ou_params else 0.0
            if theta <= 0:
                ou_confidence = 0.1  # Nearly zero but not totally flat
            elif hl > ou_hl_max:
                # Scale inversely with how far outside the range we are
                ou_confidence = max(0.1, ou_hl_max / hl)
            elif hl < ou_hl_min:
                ou_confidence = max(0.1, hl / ou_hl_min)
            else:
                ou_confidence = 1.0
            sigs = (sigs * ou_confidence).round().astype(int)
            print(f"  [OU] Soft-scaling signals by {ou_confidence:.2f} (half-life={hl:.1f}d)")

        # -- 7. Position sizing --
        if position_sizing == "vol_scaling":
            spread_ret = test_spread.pct_change().fillna(0)
            weights = volatility_scaling(spread_ret, target_vol, vol_lookback, trading_days)
            sized_sigs = apply_position_sizing(sigs, weights)
        elif position_sizing == "kelly_ml" and use_meta_labeling:
            test_vol = test_spread.pct_change().fillna(0).rolling(20, min_periods=5).std().fillna(0)
            weights = kelly_ml(expected_returns, test_vol, fraction=0.5)
            # Kelly weights can be negative (wrong direction), apply_position_sizing will multiply signal * weight
            # So if signal is 1 and weight is -0.5, we get -0.5, which is backwards. 
            # We already filtered sigs to be 0 if expected_return is negative, so weights should be positive.
            sized_sigs = apply_position_sizing(sigs, weights)
        else:
            sized_sigs = sigs.astype(float)

        # -- 9. Backtest -- (use TIME-VARYING beta, not static average)
        bt = backtest_pair(
            test_y, test_x, sized_sigs, test_hedge_df["beta"],
            cost_bps=cost_bps,
            volumes_y=volumes_y.reindex(test_idx) if volumes_y is not None else None,
            volumes_x=volumes_x.reindex(test_idx) if volumes_x is not None else None,
            use_realistic_costs=use_realistic_costs,
            cost_config=cost_config,
        )

        # -- 10. Metrics --
        bench = benchmark_returns.reindex(test_idx) if benchmark_returns is not None else None
        metrics = compute_metrics(bt, benchmark_returns=bench, trading_days=trading_days)

        print(f"  Sharpe: {metrics['sharpe_ratio']:.4f}  "
              f"Sortino: {metrics['sortino_ratio']:.4f}  "
              f"MaxDD: {metrics['max_drawdown']:.2%}  "
              f"Return: {metrics['total_return']:.2%}  "
              f"Trades: {metrics['num_trades']}")

        fold_results.append({
            "fold": fold_num,
            "train_start": str(train_idx[0].date()),
            "train_end": str(train_idx[-1].date()),
            "test_start": str(test_idx[0].date()),
            "test_end": str(test_idx[-1].date()),
            "hedge_ratio": avg_beta,
            "intercept": avg_alpha,
            "eg_pvalue": eg["p_value"],
            "adf_pvalue": adf["p_value"],
            "is_cointegrated": eg["is_cointegrated"],
            "is_stationary": adf["is_stationary"],
            "ou_params": ou_params,
            "ou_tradable": ou_ok,
            "metrics": metrics,
            "backtest": bt,
        })

    return fold_results


def print_walk_forward_summary(fold_results: list[dict]) -> None:
    """Print an aggregate summary table of walk-forward results."""
    print("\n" + "=" * 90)
    print("  WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 90)
    print(f"  {'Fold':<5} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Return':>8} "
          f"{'Trades':>7} {'Coint?':>7} {'OU?':>5}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*7} {'-'*7} {'-'*5}")

    sharpes = []
    for fr in fold_results:
        m = fr["metrics"]
        sharpes.append(m["sharpe_ratio"])
        print(
            f"  {fr['fold']:<5d} "
            f"{m['sharpe_ratio']:>8.4f} "
            f"{m['sortino_ratio']:>8.4f} "
            f"{m['max_drawdown']:>7.2%} "
            f"{m['total_return']:>7.2%} "
            f"{m['num_trades']:>7d} "
            f"{'YES' if fr['is_cointegrated'] else 'NO':>7} "
            f"{'YES' if fr.get('ou_tradable', True) else 'NO':>5}"
        )

    profitable_folds = sum(
        1 for fr in fold_results if fr["metrics"]["total_return"] > 0
    )
    avg_sharpe = np.mean(sharpes)

    print(f"\n  Profitable folds: {profitable_folds}/{len(fold_results)}")
    print(f"  Average Sharpe:   {avg_sharpe:.4f}")
    print("=" * 90 + "\n")
