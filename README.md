# StatArb-Backtester v2

**Statistical Arbitrage & Alpha Backtesting Engine**

A production-grade backtesting engine in Python for pairs-trading / mean-reversion strategies, featuring Kalman filter hedge ratios, OU process modeling, multi-pair screening, regime filtering, and walk-forward validation.

## Features

### Core Engine
- **Multi-Pair Screening** -- Automatically screens candidate pairs using Engle-Granger, Johansen cointegration, Hurst exponent, half-life of mean reversion, and correlation stability. Ranks pairs by composite score.
- **Dynamic Hedge Ratio** -- Kalman Filter (default), Rolling OLS, or static OLS for time-varying beta estimation.
- **Ornstein-Uhlenbeck Process** -- Models the spread as an OU process. Estimates theta (mean-reversion speed), mu (equilibrium), and sigma. Filters out non-mean-reverting regimes.

### Signal Generation
- **Z-Score Signals** -- Rolling z-score with configurable entry/exit/stop-loss thresholds.
- **Threshold Optimization** -- Grid search over entry/exit thresholds with walk-forward cross-validation. Includes heatmap visualization.
- **Regime Filter** -- ADX (trending detection), VIX (volatility), and rolling Hurst exponent. Only trades when all conditions favor mean-reversion.

### Risk Management
- **Position Sizing** -- Kelly fraction, volatility scaling (target vol), or fixed sizing.
- **Realistic Cost Model** -- Commission + bid-ask spread (Hasbrouck model) + slippage (square-root impact).
- **Walk-Forward Validation** -- 4-fold expanding-window out-of-sample testing with all features integrated per fold.

### Performance Analytics
- **18 Metrics** -- Sharpe, Sortino, Calmar, Max Drawdown, Max DD Duration, Tail Ratio, Win Rate, Profit Factor, Expectancy, Avg Holding Period, Turnover, Strategy Beta/Alpha, Information Ratio, and more.

## Project Structure

```
statarb/
├── config.py                    # Central configuration (all parameters)
├── data/
│   └── fetch_data.py            # yfinance downloader (multi-ticker, VIX, volumes)
├── src/
│   ├── cointegration.py         # Engle-Granger, ADF, Johansen tests
│   ├── hedge_ratio.py           # OLS, Kalman Filter, Rolling OLS
│   ├── ou_process.py            # Ornstein-Uhlenbeck spread modeling
│   ├── pair_selection.py        # Multi-pair screening (Hurst, half-life, etc.)
│   ├── pair_ranker.py           # Composite scoring & ranking
│   ├── signals.py               # Z-score signal generation + regime mask
│   ├── regime_filter.py         # ADX, VIX, Hurst regime detection
│   ├── threshold_optimizer.py   # Grid search threshold optimization
│   ├── position_sizing.py       # Kelly, vol-scaling, risk parity
│   ├── cost_model.py            # Realistic multi-component cost model
│   ├── backtest.py              # Vectorized backtesting engine
│   ├── performance.py           # 18 risk/return metrics
│   └── walk_forward.py          # Walk-forward validation
├── main.py                      # End-to-end pipeline
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Results and charts are saved to `output/`.

## Configuration

All parameters are in `config.py`. Key sections:

| Section | Key Parameters |
|---------|---------------|
| Pair Selection | `CANDIDATE_PAIRS`, `AUTO_SELECT_PAIR` |
| Hedge Ratio | `HEDGE_RATIO_METHOD` (kalman/rolling_ols/ols), `KALMAN_DELTA` |
| OU Process | `USE_OU_FILTER`, `OU_HALF_LIFE_MIN/MAX` |
| Thresholds | `OPTIMIZE_THRESHOLDS`, `ENTRY_GRID`, `EXIT_GRID` |
| Regime Filter | `USE_REGIME_FILTER`, `ADX_THRESHOLD`, `VIX_THRESHOLD` |
| Position Sizing | `POSITION_SIZING_METHOD` (vol_scaling/kelly/fixed) |
| Cost Model | `USE_REALISTIC_COSTS`, `COMMISSION_PER_SHARE`, `SLIPPAGE_FACTOR` |

## Methodology

1. **Pair Selection**: Screen 5 candidate pairs across sectors (Energy, Banking, Payments, Beverages). Rank by composite score (Hurst, half-life, Johansen, correlation stability).
2. **Cointegration**: Engle-Granger + Johansen + ADF stationarity tests.
3. **OU Process**: Fit spread to dX = theta(mu - X)dt + sigma*dW. Filter if half-life outside [5, 120] days.
4. **Dynamic Hedge Ratio**: Kalman Filter tracks time-varying beta for adaptive spread computation.
5. **Threshold Optimization**: Grid search over entry/exit z-score thresholds with CV.
6. **Regime Filter**: Trade only when ADX < 25 (range-bound), VIX < 30, and Hurst < 0.5 (mean-reverting).
7. **Position Sizing**: Volatility-target sizing to maintain 10% annualized vol.
8. **Backtesting**: Vectorized PnL with realistic transaction costs.
9. **Validation**: 4-fold walk-forward with expanding training windows.

## References

- Avellaneda & Lee (2010) -- *Statistical Arbitrage in the US Equities Market*
- Gatev, Goetzmann & Rouwenhorst (2006) -- *Pairs Trading: Performance of a Relative Value Arbitrage Rule*
- Vidyamurthy (2004) -- *Pairs Trading: Quantitative Methods and Analysis*
- Lopez de Prado (2018) -- *Advances in Financial Machine Learning*
- Almgren & Chriss (2000) -- *Optimal Execution of Portfolio Transactions*
