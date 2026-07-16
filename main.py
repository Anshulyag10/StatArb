"""
main.py -- End-to-end pipeline for the StatArb Backtester v7 (Graph AI Pair Discovery).
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project imports
from config import (
    UNIVERSE, CANDIDATE_PAIRS, PORTFOLIO_MAX_PAIRS,
    START_DATE, END_DATE,
    HEDGE_RATIO_METHOD, KALMAN_DELTA, KALMAN_OBS_COV, ROLLING_OLS_WINDOW,
    ZSCORE_LOOKBACK, ENTRY_ZSCORE, EXIT_ZSCORE, STOP_LOSS_ZSCORE,
    OPTIMIZE_THRESHOLDS, OPTIMIZATION_METHOD, OPTUNA_TRIALS,
    OPTIMIZE_LOOKBACK, OPTIMIZE_STOPLOSS,
    USE_OU_FILTER, OU_HALF_LIFE_MIN, OU_HALF_LIFE_MAX,
    USE_REGIME_FILTER, REGIME_METHOD, HMM_COMPONENTS,
    USE_META_LABELING, TRIPLE_BARRIER_PROFIT, TRIPLE_BARRIER_LOSS, TRIPLE_BARRIER_TIME,
    POSITION_SIZING_METHOD, TARGET_VOLATILITY, VOL_LOOKBACK,
    USE_REALISTIC_COSTS, COMMISSION_PER_SHARE, SPREAD_MODEL,
    FIXED_SPREAD_BPS, SLIPPAGE_FACTOR, TRADE_SIZE_DOLLARS,
    TRANSACTION_COST_BPS, NUM_FOLDS, PURGE_DAYS, EMBARGO_DAYS,
    EG_PVALUE_THRESHOLD, ADF_PVALUE_THRESHOLD,
    TRADING_DAYS_PER_YEAR,
    GRAPH_EMBEDDING_DIM, GRAPH_TRAIN_WINDOW, GRAPH_CORR_THRESHOLD
)
from data.fetch_data import fetch_multi_prices, fetch_prices, fetch_volumes
from src.walk_forward import run_walk_forward
from src.performance import compute_metrics
from src.portfolio_optimizer import equal_weight_portfolio, risk_parity_portfolio

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    print("\n" + "=" * 80)
    print("  StatArb Backtester v8 - Statistical Pair Screening Engine")
    print(f"  Period: {START_DATE} -> {END_DATE}")
    print(f"  Universe: {len(UNIVERSE)} stocks")
    print("=" * 80)

    os.makedirs("output", exist_ok=True)

    # ====================================================================
    # 1. STATISTICAL PAIR SCREENING & RANKING
    # ====================================================================
    print("\n[1] STATISTICAL PAIR SCREENING")
    print("-" * 50)
    
    multi_prices = fetch_multi_prices(UNIVERSE, START_DATE, END_DATE)
    if multi_prices.empty:
        print("[ERROR] Failed to fetch price data. Exiting.")
        return
    
    from src.pair_selection import screen_pairs
    from src.pair_ranker import rank_pairs
    
    screening_df = screen_pairs(multi_prices, CANDIDATE_PAIRS, eg_threshold=EG_PVALUE_THRESHOLD)
    
    if screening_df.empty:
        print("[ERROR] No pairs passed screening. Exiting.")
        return
    
    # Filter to only pairs that pass at least one cointegration test
    coint_df = screening_df[screening_df['is_eg_coint'] | screening_df['is_joh_coint']]
    if coint_df.empty:
        print("[WARN] No pairs pass strict cointegration. Using top-ranked pairs by composite score.")
        coint_df = screening_df
    
    ranked_df = rank_pairs(coint_df)
    
    # Select top N pairs
    top_pairs = ranked_df.head(PORTFOLIO_MAX_PAIRS)
    selected_pairs = [
        {"ticker_y": row["ticker_y"], "ticker_x": row["ticker_x"]}
        for _, row in top_pairs.iterrows()
    ]
    
    print(f"\n  Selected {len(selected_pairs)} pairs:")
    for p in selected_pairs:
        print(f"    {p['ticker_y']} / {p['ticker_x']}")
    
    if not selected_pairs:
        print("[ERROR] No valid pairs found. Exiting.")
        return

    # ====================================================================
    # 2. INDIVIDUAL PAIR EXECUTION (WALK-FORWARD)
    # ====================================================================
    print("\n[2] INDIVIDUAL PAIR EXECUTION")
    print("-" * 50)
    
    portfolio_returns = {}
    pair_metrics = {}

    for i, pair in enumerate(selected_pairs, start=1):
        sel_y, sel_x = pair["ticker_y"], pair["ticker_x"]
        
        print(f"\n============================================================")
        print(f"  PROCESSING PAIR {i}/{len(selected_pairs)}: {sel_y} / {sel_x}")
        print("============================================================")

        prices = fetch_prices(sel_y, sel_x, START_DATE, END_DATE)

        volumes_y, volumes_x = None, None
        if USE_REALISTIC_COSTS:
            try:
                volumes = fetch_volumes([sel_y, sel_x], START_DATE, END_DATE)
                volumes_y, volumes_x = volumes[sel_y], volumes[sel_x]
            except Exception:
                pass

        # -- Walk-Forward (Purged CV) --
        cost_cfg = {
            "commission_per_share": COMMISSION_PER_SHARE,
            "spread_model": SPREAD_MODEL,
            "fixed_spread_bps": FIXED_SPREAD_BPS,
            "slippage_factor": SLIPPAGE_FACTOR,
            "trade_size_dollars": TRADE_SIZE_DOLLARS,
        }
        
        hedge_kwargs = {}
        if HEDGE_RATIO_METHOD == "kalman":
            hedge_kwargs = {"delta": KALMAN_DELTA, "obs_cov": KALMAN_OBS_COV}
        elif HEDGE_RATIO_METHOD == "rolling_ols":
            hedge_kwargs = {"window": ROLLING_OLS_WINDOW}
            
        fold_results = run_walk_forward(
            prices, ticker_y=sel_y, ticker_x=sel_x, num_folds=NUM_FOLDS,
            hedge_method=HEDGE_RATIO_METHOD, hedge_kwargs=hedge_kwargs,
            zscore_lookback=ZSCORE_LOOKBACK, entry_zscore=ENTRY_ZSCORE, exit_zscore=EXIT_ZSCORE,
            stop_loss_zscore=STOP_LOSS_ZSCORE, cost_bps=TRANSACTION_COST_BPS,
            eg_pvalue=EG_PVALUE_THRESHOLD, adf_pvalue=ADF_PVALUE_THRESHOLD,
            trading_days=TRADING_DAYS_PER_YEAR, use_ou_filter=USE_OU_FILTER,
            ou_hl_min=OU_HALF_LIFE_MIN, ou_hl_max=OU_HALF_LIFE_MAX,
            position_sizing=POSITION_SIZING_METHOD,
            target_vol=TARGET_VOLATILITY, vol_lookback=VOL_LOOKBACK,
            volumes_y=volumes_y, volumes_x=volumes_x,
            use_realistic_costs=USE_REALISTIC_COSTS, cost_config=cost_cfg,
            purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS,
            use_meta_labeling=USE_META_LABELING, tb_profit=TRIPLE_BARRIER_PROFIT,
            tb_loss=TRIPLE_BARRIER_LOSS, tb_time=TRIPLE_BARRIER_TIME,
            optimize_thresholds_flag=OPTIMIZE_THRESHOLDS, opt_method=OPTIMIZATION_METHOD,
            opt_trials=OPTUNA_TRIALS, optimize_lookback=OPTIMIZE_LOOKBACK, optimize_stoploss=OPTIMIZE_STOPLOSS,
            use_regime=USE_REGIME_FILTER, regime_method=REGIME_METHOD, hmm_components=HMM_COMPONENTS,
        )

        if not fold_results:
            print("  No valid out-of-sample folds generated.")
            continue
            
        bt_full = pd.concat([res['backtest'] for res in fold_results])
        portfolio_returns[f"{sel_y}/{sel_x}"] = bt_full['net_return'].fillna(0)
        pair_metrics[f"{sel_y}/{sel_x}"] = compute_metrics(bt_full, trading_days=TRADING_DAYS_PER_YEAR)

    # ====================================================================
    # 3. PORTFOLIO AGGREGATION & CHARTS
    # ====================================================================
    print("\n[3] GENERATING METRICS & CHARTS")
    print("-" * 50)
    
    if not portfolio_returns:
        print("No valid portfolio returns to aggregate.")
        return

    port_ret_df = pd.DataFrame(portfolio_returns)
    
    # 1. Equal Weight
    eq_port_ret = equal_weight_portfolio(port_ret_df)
    
    eq_metrics = compute_metrics(eq_port_ret, trading_days=TRADING_DAYS_PER_YEAR)
    
    metrics_dict = {"Graph_Equal_Weight": eq_metrics}
    portfolio_df = pd.DataFrame({"Graph_Equal_Weight": eq_port_ret['net_return']})

    # 2. Risk Parity
    try:
        rp_port_ret = risk_parity_portfolio(port_ret_df)
        rp_metrics = compute_metrics(rp_port_ret, trading_days=TRADING_DAYS_PER_YEAR)
        
        metrics_dict["Graph_Risk_Parity"] = rp_metrics
        portfolio_df["Graph_Risk_Parity"] = rp_port_ret['net_return']
    except Exception:
        pass

    plot_portfolio(portfolio_df, metrics_dict, save_path="output/portfolio_results.png")
    print("\n[main] Done. Portfolio chart saved to output/portfolio_results.png")

def plot_portfolio(returns_df: pd.DataFrame, metrics_dict: dict, save_path: str) -> None:
    """Generate portfolio-level charts."""
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f"StatArb v8 Portfolio (Stat-Screened Pairs)", fontsize=16, fontweight="bold")

    # Aggregated Portfolios
    for col in returns_df.columns:
        equity = (1 + returns_df[col]).cumprod()
        axes.plot(equity.index, equity, label=col.replace("_", " "), linewidth=2.0)
        
    axes.set_ylabel("Portfolio Equity")
    axes.set_title("Aggregated Portfolio Performance")
    axes.legend(loc="upper left")
    axes.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
