"""
threshold_optimizer.py -- Grid search optimization for entry/exit thresholds.

Evaluates all combinations of entry and exit z-score thresholds using
cross-validated walk-forward splits. Selects the combination that maximizes
out-of-fold Sharpe ratio.
"""

import itertools
import numpy as np
import pandas as pd

from src.signals import compute_zscore, generate_signals
from src.backtest import backtest_pair
from src.performance import sharpe_ratio


def evaluate_thresholds(
    series_y: pd.Series,
    series_x: pd.Series,
    hedge_df: pd.DataFrame,
    entry_z: float,
    exit_z: float,
    zscore_lookback: int,
    stop_loss: float,
    cost_bps: float,
    num_cv_folds: int,
) -> float:
    """Evaluate a specific (entry, exit) pair using walk-forward CV."""
    from src.walk_forward import walk_forward_split
    from src.signals import compute_zscore, generate_signals
    from src.backtest import backtest_pair
    from src.performance import compute_metrics
    
    splits = walk_forward_split(series_y.index, num_cv_folds)
    fold_sharpes = []

    for train_idx, _ in splits:
        train_y = series_y.loc[train_idx]
        train_x = series_x.loc[train_idx]
        train_hedge = hedge_df.loc[train_idx]

        # Spread
        spread = train_y - train_hedge["beta"] * train_x - train_hedge["alpha"]
        zscore = compute_zscore(spread, lookback=zscore_lookback)

        sigs = generate_signals(
            zscore,
            entry_threshold=entry_z,
            exit_threshold=exit_z,
            stop_loss_threshold=stop_loss,
        )

        bt = backtest_pair(
            train_y, train_x, sigs,
            hedge_ratio=train_hedge["beta"],
            cost_bps=cost_bps,
        )

        m = compute_metrics(bt, trading_days=252)
        fold_sharpes.append(m["sharpe_ratio"])

    return float(np.mean(fold_sharpes))


def optimize_thresholds_grid(
    prices_y: pd.Series,
    prices_x: pd.Series,
    hedge_df: pd.DataFrame,
    entry_grid: list[float] = None,
    exit_grid: list[float] = None,
    zscore_lookback: int = 20,
    stop_loss: float = 4.0,
    cost_bps: float = 5.0,
    num_cv_folds: int = 3,
    trading_days: int = 252,
) -> dict:
    """
    Grid search over entry/exit thresholds using walk-forward CV.

    Parameters
    ----------
    prices_y, prices_x : price series
    hedge_df           : DataFrame with 'alpha', 'beta' columns (time-varying)
    entry_grid         : list of entry z-score thresholds to test
    exit_grid          : list of exit z-score thresholds to test
    num_cv_folds       : number of inner CV folds

    Returns
    -------
    dict with:
        best_entry, best_exit, best_sharpe, results_grid (DataFrame)
    """
    if entry_grid is None:
        entry_grid = [1.5, 1.75, 2.0, 2.25, 2.5]
    if exit_grid is None:
        exit_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Compute spread using time-varying hedge ratio
    from src.hedge_ratio import compute_spread_dynamic
    spread = compute_spread_dynamic(prices_y, prices_x, hedge_df)

    # Walk-forward CV splits
    n = len(spread)
    block_size = n // (num_cv_folds + 1)

    results = []

    for entry_z, exit_z in itertools.product(entry_grid, exit_grid):
        # Skip invalid combinations (exit must be < entry)
        if exit_z >= entry_z:
            continue

        fold_sharpes = []

        for k in range(num_cv_folds):
            test_start = (k + 1) * block_size
            test_end = min(test_start + block_size, n)

            test_idx = spread.index[test_start:test_end]
            test_spread = spread.loc[test_idx]
            test_y = prices_y.reindex(test_idx)
            test_x = prices_x.reindex(test_idx)

            # Use average beta for this fold's backtest
            fold_beta = hedge_df["beta"].reindex(test_idx).mean()

            zscore = compute_zscore(test_spread, lookback=zscore_lookback)
            sigs = generate_signals(
                zscore,
                entry_threshold=entry_z,
                exit_threshold=exit_z,
                stop_loss_threshold=stop_loss,
            )

            bt = backtest_pair(test_y, test_x, sigs, fold_beta, cost_bps=cost_bps)
            net_ret = bt["net_return"].dropna()
            sr = sharpe_ratio(net_ret, trading_days)
            fold_sharpes.append(sr)

        avg_sharpe = np.mean(fold_sharpes)
        results.append({
            "entry_zscore": entry_z,
            "exit_zscore": exit_z,
            "avg_sharpe": avg_sharpe,
            "std_sharpe": np.std(fold_sharpes),
            "fold_sharpes": fold_sharpes,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("avg_sharpe", ascending=False).reset_index(drop=True)

    best = results_df.iloc[0]
    best_entry = best['entry_zscore']
    best_exit = best['exit_zscore']
    best_sharpe = best['avg_sharpe']
    best_std = best['std_sharpe']

    print(f"  [optimize] Grid search: {len(results)} combinations tested")
    print(f"  [optimize] Best: entry={best_entry}, exit={best_exit}, "
          f"Sharpe={best_sharpe:.4f} (+/- {best_std:.4f})")

    return {
        "best_entry": best_entry,
        "best_exit": best_exit,
        "best_sharpe": best_sharpe,
        "results_grid": results_df,
    }


def optimize_thresholds_bayesian(
    series_y: pd.Series,
    series_x: pd.Series,
    hedge_df: pd.DataFrame,
    n_trials: int = 30,
    zscore_lookback: int = 20,
    stop_loss: float = 4.0,
    cost_bps: float = 5.0,
    num_cv_folds: int = 3,
    optimize_lookback: bool = False,
    optimize_stoploss: bool = False,
) -> dict:
    """
    Optimize entry/exit thresholds (and optionally lookback/stoploss) using Bayesian Optimization (Optuna).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        entry_z = trial.suggest_float("entry_z", 1.0, 3.0, step=0.1)
        # Ensure exit is strictly less than entry
        exit_z = trial.suggest_float("exit_z", 0.0, entry_z - 0.1, step=0.1)
        
        # Optionally optimize lookback and stop loss
        lookback = trial.suggest_int("lookback", 10, 60, step=5) if optimize_lookback else zscore_lookback
        sl = trial.suggest_float("stop_loss", entry_z + 0.5, 6.0, step=0.5) if optimize_stoploss else stop_loss
        
        return evaluate_thresholds(
            series_y, series_x, hedge_df,
            entry_z, exit_z, lookback, sl, cost_bps, num_cv_folds
        )

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_entry = study.best_params["entry_z"]
    best_exit = study.best_params["exit_z"]
    best_lookback = study.best_params.get("lookback", zscore_lookback)
    best_sl = study.best_params.get("stop_loss", stop_loss)
    best_sharpe = study.best_value

    print(f"  [optimize] Bayesian (Optuna): {n_trials} trials completed")
    print(f"  [optimize] Best: entry={best_entry:.2f}, exit={best_exit:.2f}, "
          f"lookback={best_lookback}, stop_loss={best_sl:.1f}, Sharpe={best_sharpe:.4f}")

    # Build a dummy results grid for heatmap compatibility
    trials_df = study.trials_dataframe()
    results = []
    for _, row in trials_df.iterrows():
        if row["state"] == "COMPLETE":
            results.append({
                "entry": row["params_entry_z"],
                "exit": row["params_exit_z"],
                "mean_sharpe": row["value"],
            })
    
    return {
        "best_entry": best_entry,
        "best_exit": best_exit,
        "best_lookback": best_lookback,
        "best_stop_loss": best_sl,
        "best_sharpe": best_sharpe,
        "results_grid": pd.DataFrame(results),
    }


def optimize_thresholds(
    series_y: pd.Series,
    series_x: pd.Series,
    hedge_df: pd.DataFrame,
    method: str = "grid",
    entry_grid: list[float] | None = None,
    exit_grid: list[float] | None = None,
    n_trials: int = 30,
    zscore_lookback: int = 20,
    stop_loss: float = 4.0,
    cost_bps: float = 5.0,
    num_cv_folds: int = 3,
    optimize_lookback: bool = False,
    optimize_stoploss: bool = False,
) -> dict:
    """Dispatcher for threshold optimization."""
    if method == "bayesian":
        return optimize_thresholds_bayesian(
            series_y, series_x, hedge_df, n_trials,
            zscore_lookback, stop_loss, cost_bps, num_cv_folds,
            optimize_lookback, optimize_stoploss
        )
    else:
        if entry_grid is None or exit_grid is None:
            raise ValueError("entry_grid and exit_grid required for grid search")
        return optimize_thresholds_grid(
            series_y, series_x, hedge_df, entry_grid, exit_grid,
            zscore_lookback, stop_loss, cost_bps, num_cv_folds
        )


def plot_threshold_heatmap(results_df: pd.DataFrame, out_path: str = "output/threshold_heatmap.png") -> None:
    """Generate and save a heatmap of Sharpe ratio across the entry/exit grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = results_df.pivot(
        index="exit_zscore",
        columns="entry_zscore",
        values="avg_sharpe",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])

    ax.set_xlabel("Entry Z-Score Threshold")
    ax.set_ylabel("Exit Z-Score Threshold")
    ax.set_title("Threshold Optimization: Avg Sharpe Ratio (CV)")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="black", fontsize=9)

    plt.colorbar(im, label="Sharpe Ratio")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved: {output_path}")
