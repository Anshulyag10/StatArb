"""
config.py -- Central configuration for the StatArb Backtester v2.
"""

import itertools

# ============================================================================
# UNIVERSE SELECTION & GRAPH AI SETTINGS
# ============================================================================
UNIVERSE = [
    # Tech / Comm
    "AAPL", "MSFT", "GOOG", "META", "NVDA", "AMD", "INTC", "CSCO", 
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST",
    # Energy / Industrials
    "XOM", "CVX", "GE", "BA", "CAT",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK"
]

CANDIDATE_PAIRS = list(itertools.combinations(UNIVERSE, 2))
PORTFOLIO_MAX_PAIRS = 10

# Graph AI Settings
GRAPH_EMBEDDING_DIM = 4
GRAPH_TRAIN_WINDOW = 504  # 2 years of history to train Link Prediction
GRAPH_CORR_THRESHOLD = 0.7

# ============================================================================
# DATE RANGE
# ============================================================================
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"

# ============================================================================
# HEDGE RATIO METHOD
# ============================================================================
# Options: "ols", "kalman", "rolling_ols"
HEDGE_RATIO_METHOD = "kalman"

# Kalman filter parameters
KALMAN_DELTA       = 1e-4   # State transition covariance scaling
KALMAN_OBS_COV     = 1.0    # Observation noise variance

# Rolling OLS parameters
ROLLING_OLS_WINDOW = 60     # Lookback window for rolling OLS

# ============================================================================
# Z-SCORE SIGNAL PARAMETERS
# ============================================================================
ZSCORE_LOOKBACK   = 20      # Rolling window for z-score (days)
ENTRY_ZSCORE      = 1.25    # Avellaneda & Lee (2010) optimal entry threshold
EXIT_ZSCORE       = 0.50    # Exit when spread near equilibrium
STOP_LOSS_ZSCORE  = 3.0     # Tighter risk control

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================
OPTIMIZE_THRESHOLDS   = False  # Disabled: per-fold Optuna overfits on tiny inner CV
OPTIMIZATION_METHOD   = "bayesian"  # "grid" or "bayesian"

# For Grid Search
ENTRY_GRID = [1.5, 1.75, 2.0, 2.25, 2.5]
EXIT_GRID  = [0.0, 0.25, 0.5, 0.75, 1.0]

# For Bayesian Optimization (Optuna)
OPTUNA_TRIALS = 15          # Number of trials for Bayesian opt
OPTIMIZE_LOOKBACK = True    # Let Optuna optimize ZSCORE_LOOKBACK
OPTIMIZE_STOPLOSS = True    # Let Optuna optimize STOP_LOSS_ZSCORE

OPTIM_CV_FOLDS = 3          # Inner CV folds for threshold optimization

# ============================================================================
# ORNSTEIN-UHLENBECK FILTER
# ============================================================================
USE_OU_FILTER    = True
OU_HALF_LIFE_MIN = 3.0      # Minimum half-life (days) to avoid microstructure noise
OU_HALF_LIFE_MAX = 120.0    # Relaxed from 60 -- less aggressive pair rejection

# ============================================================================
# REGIME FILTER (HMM / ADX / VIX)
# ============================================================================
USE_REGIME_FILTER = False   # Disabled: HMM re-fits on test data (look-ahead bias) & over-filters
REGIME_METHOD     = "hmm"   # Options: "static", "hmm"

# Static Thresholds (if REGIME_METHOD == "static")
ADX_PERIOD        = 14      # Lookback for ADX calculation
ADX_THRESHOLD     = 25      # ADX > threshold indicates trending market (avoid)
VIX_THRESHOLD     = 30      # VIX > threshold indicates panic (avoid)
REGIME_VR_WINDOW  = 5       # Window for Variance Ratio
VR_THRESHOLD      = 1.0     # VR < 1.0 indicates mean-reversion

# Hidden Markov Model (if REGIME_METHOD == "hmm")
HMM_COMPONENTS    = 2       # Number of latent regimes (e.g. 2 = Low Vol, High Vol)

# ============================================================================
# META-LABELING & TRIPLE BARRIER (v5)
# ============================================================================
USE_META_LABELING = False   # Disabled: XGBoost trains on <30 samples per fold = pure overfitting
TRIPLE_BARRIER_PROFIT = 2.0  # Take-profit multiple of spread vol
TRIPLE_BARRIER_LOSS   = 2.0  # Stop-loss multiple of spread vol
TRIPLE_BARRIER_TIME   = 20   # Maximum holding period (days)g)

# ============================================================================
# POSITION SIZING
# ============================================================================
POSITION_SIZING_METHOD = "constant" # Simplest: no amplification of noise
KELLY_FRACTION = 0.5            # Half-Kelly for safety (unused with constant)
TARGET_VOLATILITY = 0.15        # Target annualized portfolio volatility (unused with constant)
VOL_LOOKBACK = 20               # Shorter lookback for more responsive vol estimate

# ============================================================================
# TRANSACTION COST MODEL
# ============================================================================
# Simple model (backward compatible)
TRANSACTION_COST_BPS = 2  # Reduced: 2 bps per leg for liquid large-caps

# Realistic model
USE_REALISTIC_COSTS    = False  # Disabled until baseline Sharpe established; may double-count
COMMISSION_PER_SHARE   = 0.005   # $/share
SPREAD_MODEL           = "hasbrouck"  # "fixed" or "hasbrouck"
FIXED_SPREAD_BPS       = 2       # Used if SPREAD_MODEL = "fixed"
SLIPPAGE_FACTOR        = 0.1     # Fraction of daily vol
TRADE_SIZE_DOLLARS     = 100_000  # Notional per leg

# ============================================================================
# PAIR SELECTION THRESHOLDS
# ============================================================================
MIN_CORRELATION          = 0.70   # Min Pearson correlation to proceed
VR_PAIR_THRESHOLD        = 1.0    # VR < this for pair to be considered
MIN_HALF_LIFE            = 5      # Minimum half-life (days)
MAX_HALF_LIFE            = 60     # Maximum half-life (days)
CORRELATION_STABILITY_MAX = 0.15  # Max std of rolling correlation

# ============================================================================
# WALK-FORWARD VALIDATION (Purged CV)
# ============================================================================
NUM_FOLDS = 4
PURGE_DAYS = ZSCORE_LOOKBACK  # Drop data points leaking into test
EMBARGO_DAYS = 5              # Extra gap between train and test

# ============================================================================
# COINTEGRATION THRESHOLDS
# ============================================================================
EG_PVALUE_THRESHOLD  = 0.05   # Relaxed from 0.01 for multi-pair screening
ADF_PVALUE_THRESHOLD = 0.05   # Relaxed from 0.001 for multi-pair screening

# ============================================================================
# ANNUALIZATION
# ============================================================================
TRADING_DAYS_PER_YEAR = 252
