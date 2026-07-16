"""
ou_process.py -- Ornstein-Uhlenbeck process for spread modeling.

Model:  dX = theta * (mu - X) * dt + sigma * dW

Where:
    theta : mean-reversion speed
    mu    : long-run equilibrium level
    sigma : volatility of the process
    half_life = ln(2) / theta

Reference: Avellaneda & Lee (2010), "Statistical Arbitrage in the US Equities Market"
"""

import numpy as np
import pandas as pd


def fit_ou(spread: pd.Series) -> dict:
    """
    Estimate OU process parameters via OLS on the discretized model.

    Discrete version (Euler):
        spread[t] - spread[t-1] = a + b * spread[t-1] + eps

    Then:
        theta = -b / dt   (where dt = 1 day)
        mu    = -a / b
        sigma = std(eps) * sqrt(2 * theta)

    Returns
    -------
    dict with keys:
        theta      : float -- mean-reversion speed (per day)
        mu         : float -- long-run mean
        sigma      : float -- OU volatility
        half_life  : float -- half-life in days = ln(2) / theta
        b_coeff    : float -- AR(1) coefficient (should be negative for MR)
        residual_std : float -- std of regression residuals
    """
    s = spread.dropna()
    if len(s) < 30:
        return {
            "theta": 0.0, "mu": 0.0, "sigma": 0.0,
            "half_life": float("inf"), "b_coeff": 0.0, "residual_std": 0.0,
        }

    y = s.diff().iloc[1:].values      # spread[t] - spread[t-1]
    x = s.iloc[:-1].values            # spread[t-1]

    # OLS: y = a + b * x
    x_with_const = np.column_stack([np.ones(len(x)), x])
    coeffs, residuals, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
    a, b = coeffs

    # Predict and compute residual std
    y_pred = x_with_const @ coeffs
    eps = y - y_pred
    residual_std = float(np.std(eps, ddof=2))

    # Extract OU parameters (dt = 1 day)
    if b >= 0:
        # Not mean-reverting
        return {
            "theta": 0.0, "mu": float(np.mean(s)),
            "sigma": residual_std, "half_life": float("inf"),
            "b_coeff": float(b), "residual_std": residual_std,
        }

    theta = -b  # mean-reversion speed
    mu = -a / b  # long-run mean
    sigma = residual_std * np.sqrt(2 * theta) if theta > 0 else residual_std

    half_life = np.log(2) / theta if theta > 0 else float("inf")

    return {
        "theta": float(theta),
        "mu": float(mu),
        "sigma": float(sigma),
        "half_life": float(half_life),
        "b_coeff": float(b),
        "residual_std": residual_std,
    }


def ou_zscore(spread: pd.Series, mu: float, sigma: float) -> pd.Series:
    """
    Compute z-score using OU-estimated equilibrium parameters.

        z_t = (spread_t - mu) / sigma

    More theoretically grounded than rolling z-score since mu and sigma
    are estimated from the OU model rather than a rolling window.
    """
    if sigma <= 0:
        return pd.Series(0.0, index=spread.index)
    return (spread - mu) / sigma


def should_trade_ou(
    ou_params: dict,
    half_life_min: float = 5,
    half_life_max: float = 120,
) -> tuple[bool, str]:
    """
    Determine if the spread is suitable for trading based on OU parameters.

    Conditions:
        1. theta > 0 (mean-reverting)
        2. half_life within [min, max] range

    Returns (should_trade: bool, reason: str)
    """
    theta = ou_params["theta"]
    hl = ou_params["half_life"]

    if theta <= 0:
        return False, f"Not mean-reverting (theta={theta:.4f})"
    if hl < half_life_min:
        return False, f"Half-life too short ({hl:.1f}d < {half_life_min}d) -- likely noise"
    if hl > half_life_max:
        return False, f"Half-life too long ({hl:.1f}d > {half_life_max}d) -- too slow"

    return True, f"OK (theta={theta:.4f}, half-life={hl:.1f}d)"


def print_ou_report(ou_params: dict) -> None:
    """Pretty-print OU process parameters."""
    print("\n" + "=" * 50)
    print("  ORNSTEIN-UHLENBECK PROCESS")
    print("=" * 50)
    print(f"  {'theta (MR speed):':<25} {ou_params['theta']:.6f}")
    print(f"  {'mu (equilibrium):':<25} {ou_params['mu']:.6f}")
    print(f"  {'sigma (OU vol):':<25} {ou_params['sigma']:.6f}")
    print(f"  {'half-life (days):':<25} {ou_params['half_life']:.2f}")
    print(f"  {'AR(1) coeff (b):':<25} {ou_params['b_coeff']:.6f}")

    tradable, reason = should_trade_ou(ou_params)
    status = "YES" if tradable else "NO"
    print(f"  {'Tradable:':<25} {status} -- {reason}")
    print("=" * 50 + "\n")
