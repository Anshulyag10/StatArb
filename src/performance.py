"""
performance.py -- Comprehensive risk/return metrics and reporting.

Metrics:
    Core:       Sharpe, Sortino, Calmar, Max Drawdown, Total Return
    Risk:       Max DD Duration, Tail Ratio, Strategy Beta/Alpha
    Trading:    Win Rate, Profit Factor, Expectancy, Avg Holding Period, Turnover
    Cost:       Total Tx Cost, Information Ratio
"""

import numpy as np
import pandas as pd


# ============================================================================
# INDIVIDUAL METRIC FUNCTIONS
# ============================================================================

def sharpe_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    """Annualized Sharpe ratio (assumes risk-free rate ~ 0)."""
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(trading_days))


def sortino_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    """Annualized Sortino ratio (penalizes only downside volatility)."""
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(trading_days))


def calmar_ratio(returns: pd.Series, equity: pd.Series, trading_days: int = 252) -> float:
    """Calmar ratio = annualized return / |max drawdown|."""
    ann_ret = annualized_return(equity, trading_days)
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(ann_ret / mdd)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g., -0.0216 = -2.16%)."""
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    return float(drawdown.min())


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Maximum drawdown duration in trading days (peak to recovery)."""
    cummax = equity_curve.cummax()
    is_underwater = equity_curve < cummax

    if not is_underwater.any():
        return 0

    # Find streaks of underwater periods
    groups = (~is_underwater).cumsum()
    underwater_groups = groups[is_underwater]
    if len(underwater_groups) == 0:
        return 0

    durations = underwater_groups.groupby(underwater_groups).count()
    return int(durations.max())


def total_return(equity_curve: pd.Series) -> float:
    """Total cumulative return."""
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def annualized_return(equity_curve: pd.Series, trading_days: int = 252) -> float:
    """CAGR-style annualized return."""
    n_days = len(equity_curve)
    total = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if total <= 0:
        return -1.0
    return float(total ** (trading_days / n_days) - 1)


def win_rate(returns: pd.Series) -> float:
    """Fraction of positive-return days (excluding flat days)."""
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).sum() / len(active))


def profit_factor(returns: pd.Series) -> float:
    """Gross profits / gross losses. > 1 is profitable."""
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def expectancy(returns: pd.Series) -> float:
    """Expected return per trade = win_rate * avg_win - loss_rate * avg_loss."""
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    wins = active[active > 0]
    losses = active[active < 0]
    p_win = len(wins) / len(active) if len(active) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    return float(p_win * avg_win - (1 - p_win) * avg_loss)


def avg_holding_period(signals: pd.Series) -> float:
    """Average duration of non-zero signal streaks (in days)."""
    is_active = signals != 0
    if not is_active.any():
        return 0.0

    # Detect streak boundaries
    streak_id = (is_active != is_active.shift()).cumsum()
    active_streaks = streak_id[is_active]

    if len(active_streaks) == 0:
        return 0.0

    streak_lengths = active_streaks.groupby(active_streaks).count()
    return float(streak_lengths.mean())


def turnover(signals: pd.Series) -> float:
    """Daily turnover = sum of absolute signal changes / total days."""
    changes = signals.diff().abs().sum()
    return float(changes / len(signals))


def tail_ratio(returns: pd.Series) -> float:
    """Tail ratio = |95th percentile| / |5th percentile|. > 1 means fatter right tail."""
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    p95 = np.percentile(clean, 95)
    p5  = np.percentile(clean, 5)
    if abs(p5) == 0:
        return float("inf") if p95 > 0 else 0.0
    return float(abs(p95) / abs(p5))


def strategy_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Beta = cov(strategy, benchmark) / var(benchmark)."""
    aligned = pd.DataFrame({
        "strat": strategy_returns,
        "bench": benchmark_returns,
    }).dropna()
    if len(aligned) < 10 or aligned["bench"].var() == 0:
        return 0.0
    return float(aligned["strat"].cov(aligned["bench"]) / aligned["bench"].var())


def strategy_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Annualized Jensen's alpha = annualized(strat - beta * bench)."""
    beta = strategy_beta(strategy_returns, benchmark_returns)
    excess = strategy_returns - beta * benchmark_returns
    return float(excess.mean() * trading_days)


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Information ratio = mean(active return) / std(active return) * sqrt(252)."""
    active = (strategy_returns - benchmark_returns).dropna()
    if len(active) < 10 or active.std() == 0:
        return 0.0
    return float(active.mean() / active.std() * np.sqrt(trading_days))


# ============================================================================
# AGGREGATE METRICS COMPUTATION
# ============================================================================

def compute_metrics(
    result_df: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    trading_days: int = 252,
) -> dict:
    """
    Compute the full suite of performance metrics from a backtest DataFrame.

    Expects columns: 'net_return', 'equity', 'signal', 'trade_flag', 'cost'.
    """
    net = result_df["net_return"].dropna()
    eq  = result_df["equity"].dropna()
    sig = result_df.get("signal", pd.Series(dtype=float))

    # Guard against empty data
    if len(net) < 2 or len(eq) < 2:
        return {
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "max_drawdown": 0.0,
            "max_dd_duration": 0, "total_return": 0.0, "annualized_return": 0.0,
            "tail_ratio": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "avg_holding_period": 0.0, "turnover": 0.0,
            "num_trades": 0, "total_cost": 0.0, "calmar_ratio": 0.0,
            "information_ratio": 0.0, "strategy_beta": 0.0, "strategy_alpha": 0.0,
        }

    metrics = {
        # Core
        "sharpe_ratio":         sharpe_ratio(net, trading_days),
        "sortino_ratio":        sortino_ratio(net, trading_days),
        "max_drawdown":         max_drawdown(eq),
        "max_dd_duration":      max_drawdown_duration(eq),
        "total_return":         total_return(eq),
        "annualized_return":    annualized_return(eq, trading_days),

        # Risk
        "tail_ratio":           tail_ratio(net),

        # Trading
        "win_rate":             win_rate(net),
        "profit_factor":        profit_factor(net),
        "expectancy":           expectancy(net),
        "avg_holding_period":   avg_holding_period(sig) if len(sig) > 0 else 0,
        "turnover":             turnover(sig) if len(sig) > 0 else 0,
        "num_trades":           int(result_df.get("trade_flag", pd.Series(0, index=result_df.index)).sum()),
        "total_cost":           float(result_df.get("cost", pd.Series(0.0, index=result_df.index)).sum()),
    }

    # Benchmark-relative metrics (if benchmark provided)
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(net.index).fillna(0)
        metrics["calmar_ratio"]       = calmar_ratio(net, eq, trading_days)
        metrics["information_ratio"]  = information_ratio(net, bench, trading_days)
        metrics["strategy_beta"]      = strategy_beta(net, bench)
        metrics["strategy_alpha"]     = strategy_alpha(net, bench, trading_days)
    else:
        metrics["calmar_ratio"]       = calmar_ratio(net, eq, trading_days)
        metrics["information_ratio"]  = 0.0
        metrics["strategy_beta"]      = 0.0
        metrics["strategy_alpha"]     = 0.0

    return metrics


# ============================================================================
# PRETTY-PRINTING
# ============================================================================

def print_metrics(metrics: dict, label: str = "Strategy") -> None:
    """Pretty-print the full metrics dictionary."""
    print(f"\n{'-' * 60}")
    print(f"  {label} - Performance Metrics")
    print(f"{'-' * 60}")

    # Core
    print(f"  {'Sharpe Ratio:':<28} {metrics['sharpe_ratio']:>10.4f}")
    print(f"  {'Sortino Ratio:':<28} {metrics['sortino_ratio']:>10.4f}")
    print(f"  {'Calmar Ratio:':<28} {metrics['calmar_ratio']:>10.4f}")
    print(f"  {'Max Drawdown:':<28} {metrics['max_drawdown']:>10.2%}")
    print(f"  {'Max DD Duration (days):':<28} {metrics['max_dd_duration']:>10d}")
    print(f"  {'Total Return:':<28} {metrics['total_return']:>10.2%}")
    print(f"  {'Annualized Return:':<28} {metrics['annualized_return']:>10.2%}")

    # Risk
    print(f"  {'Tail Ratio (95/5):':<28} {metrics['tail_ratio']:>10.4f}")
    if metrics.get('strategy_beta', 0) != 0 or metrics.get('strategy_alpha', 0) != 0:
        print(f"  {'Strategy Beta:':<28} {metrics['strategy_beta']:>10.4f}")
        print(f"  {'Strategy Alpha (ann.):':<28} {metrics['strategy_alpha']:>10.4f}")
        print(f"  {'Information Ratio:':<28} {metrics['information_ratio']:>10.4f}")

    # Trading
    print(f"  {'Win Rate:':<28} {metrics['win_rate']:>10.2%}")
    print(f"  {'Profit Factor:':<28} {metrics['profit_factor']:>10.4f}")
    print(f"  {'Expectancy (per trade):':<28} {metrics['expectancy']:>10.6f}")
    print(f"  {'Avg Holding Period (days):':<28} {metrics['avg_holding_period']:>10.1f}")
    print(f"  {'Daily Turnover:':<28} {metrics['turnover']:>10.4f}")
    print(f"  {'Number of Trades:':<28} {metrics['num_trades']:>10d}")
    print(f"  {'Total Tx Cost:':<28} {metrics['total_cost']:>10.4f}")
    print(f"{'-' * 60}\n")


def print_benchmark_metrics(bh_df: pd.DataFrame, trading_days: int = 252) -> dict:
    """Compute and print buy-and-hold benchmark metrics."""
    ret = bh_df["return"].dropna()
    eq  = bh_df["equity"].dropna()

    metrics = {
        "sharpe_ratio":      sharpe_ratio(ret, trading_days),
        "sortino_ratio":     sortino_ratio(ret, trading_days),
        "max_drawdown":      max_drawdown(eq),
        "total_return":      total_return(eq),
        "annualized_return": annualized_return(eq, trading_days),
    }

    print(f"\n{'-' * 60}")
    print(f"  Buy-and-Hold Benchmark - Performance Metrics")
    print(f"{'-' * 60}")
    print(f"  {'Sharpe Ratio:':<28} {metrics['sharpe_ratio']:>10.4f}")
    print(f"  {'Sortino Ratio:':<28} {metrics['sortino_ratio']:>10.4f}")
    print(f"  {'Max Drawdown:':<28} {metrics['max_drawdown']:>10.2%}")
    print(f"  {'Total Return:':<28} {metrics['total_return']:>10.2%}")
    print(f"  {'Annualized Return:':<28} {metrics['annualized_return']:>10.2%}")
    print(f"{'-' * 60}\n")

    return metrics
