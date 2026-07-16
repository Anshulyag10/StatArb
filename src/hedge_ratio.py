"""
hedge_ratio.py -- Hedge ratio estimation for pairs trading.

Methods:
    1. Static OLS (original)
    2. Kalman Filter (time-varying, best)
    3. Rolling OLS (time-varying, simpler)

Reference:
    "Pairs Trading with Kalman Filter" -- QuantStart
    Durbin & Koopman -- Time Series Analysis by State Space Methods
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================================
# 1. STATIC OLS (Original)
# ============================================================================

def estimate_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
) -> dict:
    """
    Estimate the hedge ratio beta via OLS:  Y = alpha + beta*X + eps

    Returns
    -------
    dict with keys:
        beta      : float -- hedge ratio (coefficient on X)
        alpha     : float -- intercept
        r_squared : float
        residuals : pd.Series -- OLS residuals (i.e., the spread)
    """
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()

    return {
        "beta": model.params.iloc[1],
        "alpha": model.params.iloc[0],
        "r_squared": model.rsquared,
        "residuals": model.resid,
    }


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    beta: float,
    alpha: float = 0.0,
) -> pd.Series:
    """Compute the spread:  spread_t = Y_t - beta*X_t - alpha"""
    return y - beta * x - alpha


# ============================================================================
# 2. KALMAN FILTER (Time-Varying)
# ============================================================================

def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    obs_cov: float = 1.0,
) -> pd.DataFrame:
    """
    Estimate time-varying hedge ratio using a linear Kalman Filter.

    State model:
        state = [alpha_t, beta_t]  (intercept and hedge ratio)
        state_t = state_{t-1} + w_t,   w_t ~ N(0, Q)

    Observation model:
        y_t = alpha_t + beta_t * x_t + v_t,   v_t ~ N(0, R)

    Parameters
    ----------
    y     : dependent asset prices
    x     : independent asset prices
    delta : state transition covariance scaling (smaller = smoother)
    obs_cov : observation noise variance

    Returns
    -------
    DataFrame with columns: 'alpha', 'beta', indexed by date
    """
    n = len(y)
    y_vals = y.values
    x_vals = x.values

    # State: [alpha, beta]
    state = np.array([0.0, 1.0])  # Initial guess: alpha=0, beta=1

    # State covariance
    P = np.eye(2) * 1.0

    # State transition covariance (random walk with small noise)
    Q = np.eye(2) * delta

    # Observation noise
    R = obs_cov

    # Storage
    alphas = np.zeros(n)
    betas  = np.zeros(n)

    for t in range(n):
        # Observation vector: y_t = [1, x_t] @ [alpha, beta]
        H = np.array([1.0, x_vals[t]])

        # -- Predict --
        # state_{t|t-1} = state_{t-1}  (random walk)
        # P_{t|t-1} = P_{t-1} + Q
        P = P + Q

        # -- Update --
        # Innovation
        y_hat = H @ state
        innovation = y_vals[t] - y_hat

        # Innovation covariance
        S = H @ P @ H + R

        # Kalman gain
        K = P @ H / S

        # Update state
        state = state + K * innovation

        # Update covariance
        P = P - np.outer(K, H) @ P

        # Store
        alphas[t] = state[0]
        betas[t]  = state[1]

    return pd.DataFrame({
        "alpha": alphas,
        "beta": betas,
    }, index=y.index)


# ============================================================================
# 3. ROLLING OLS (Time-Varying)
# ============================================================================

def rolling_ols_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """
    Estimate time-varying hedge ratio using rolling OLS.

    For each day t, run OLS on [t-window+1 : t] to get beta_t and alpha_t.

    Returns
    -------
    DataFrame with columns: 'alpha', 'beta', indexed by date.
    NaN for the first (window-1) rows.
    """
    n = len(y)
    alphas = np.full(n, np.nan)
    betas  = np.full(n, np.nan)

    y_vals = y.values
    x_vals = x.values

    for t in range(window - 1, n):
        y_win = y_vals[t - window + 1 : t + 1]
        x_win = x_vals[t - window + 1 : t + 1]

        x_const = np.column_stack([np.ones(window), x_win])
        coeffs, _, _, _ = np.linalg.lstsq(x_const, y_win, rcond=None)
        alphas[t] = coeffs[0]
        betas[t]  = coeffs[1]

    df = pd.DataFrame({"alpha": alphas, "beta": betas}, index=y.index)
    # Forward-fill early NaNs with first valid value
    df = df.bfill()
    return df


# ============================================================================
# DYNAMIC SPREAD (Time-Varying Hedge Ratio)
# ============================================================================

def compute_spread_dynamic(
    y: pd.Series,
    x: pd.Series,
    hedge_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute the spread using time-varying hedge ratio:
        spread_t = y_t - beta_t * x_t - alpha_t
    """
    betas  = hedge_df["beta"].reindex(y.index).ffill().bfill()
    alphas = hedge_df["alpha"].reindex(y.index).ffill().bfill()
    return y - betas * x - alphas


# ============================================================================
# DISPATCHER
# ============================================================================

def compute_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    method: str = "kalman",
    **kwargs,
) -> pd.DataFrame:
    """
    Dispatch to the appropriate hedge ratio method.

    Parameters
    ----------
    method : "ols", "kalman", or "rolling_ols"

    Returns
    -------
    DataFrame with 'alpha' and 'beta' columns (constant for OLS).
    """
    if method == "kalman":
        delta = kwargs.get("delta", 1e-4)
        obs_cov = kwargs.get("obs_cov", 1.0)
        return kalman_hedge_ratio(y, x, delta=delta, obs_cov=obs_cov)

    elif method == "rolling_ols":
        window = kwargs.get("window", 60)
        return rolling_ols_hedge_ratio(y, x, window=window)

    elif method == "ols":
        hr = estimate_hedge_ratio(y, x)
        # Return as constant time series for uniform API
        df = pd.DataFrame({
            "alpha": hr["alpha"],
            "beta": hr["beta"],
        }, index=y.index)
        return df

    else:
        raise ValueError(f"Unknown hedge ratio method: {method}")
