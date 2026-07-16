"""
pca_factors.py -- PCA Factor Extraction and Residual Computation 
Implementation of Avellaneda & Lee (2008) Cross-Sectional Statistical Arbitrage.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def fit_pca(train_returns: pd.DataFrame, num_factors: int = 15) -> tuple[PCA, pd.DataFrame, pd.Series, pd.Series]:
    """
    Fit PCA on the training return matrix and extract factor returns.
    
    Returns:
        pca_model: Fitted sklearn PCA model
        factor_returns: DataFrame of shape (T, num_factors)
        scaler_mean: Mean of training returns
        scaler_std: Std of training returns
    """
    # Standardize returns before PCA
    scaler_mean = train_returns.mean()
    scaler_std = train_returns.std()
    norm_returns = (train_returns - scaler_mean) / scaler_std.clip(lower=1e-8)
    
    # Fill any remaining NaNs to prevent PCA failure
    norm_returns = norm_returns.fillna(0)
    
    pca = PCA(n_components=num_factors)
    pca.fit(norm_returns)
    
    # Project returns onto principal components to get factor returns
    # F = R * W
    factor_returns_np = pca.transform(norm_returns)
    factor_returns = pd.DataFrame(
        factor_returns_np, 
        index=train_returns.index,
        columns=[f"Factor_{i+1}" for i in range(num_factors)]
    )
    
    return pca, factor_returns, scaler_mean, scaler_std

def compute_loadings(train_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Regress each stock against the PCA factors to find Beta loadings.
    R_i = alpha_i + sum(beta_ij * F_j) + epsilon_i
    """
    betas = {}
    train_returns = train_returns.fillna(0)
    X = factor_returns.values
    
    for ticker in train_returns.columns:
        y = train_returns[ticker].values
        
        lr = LinearRegression()
        lr.fit(X, y)
        betas[ticker] = lr.coef_
        
    return pd.DataFrame(betas, index=factor_returns.columns).T


def compute_residuals(
    returns: pd.DataFrame, 
    pca_model: PCA, 
    scaler_mean: pd.Series, 
    scaler_std: pd.Series, 
    betas: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute out-of-sample residual returns and synthetic residual prices.
    
    1. Project out-of-sample returns onto the fixed PCA components to get F_t.
    2. Multiply F_t by fixed Betas to get systematic returns.
    3. Residual = Actual - Systematic.
    4. Synthetic Price = Cumsum(Residual).
    """
    # Standardize using TRAINING mean/std to prevent leakage
    norm_returns = (returns - scaler_mean) / scaler_std.clip(lower=1e-8)
    norm_returns = norm_returns.fillna(0)
    
    # Get factor returns
    factor_returns_np = pca_model.transform(norm_returns)
    factor_returns = pd.DataFrame(
        factor_returns_np, 
        index=returns.index,
        columns=[f"Factor_{i+1}" for i in range(pca_model.n_components)]
    )
    
    # Compute systematic returns (Betas x Factors)
    # betas is shape (N, K), factor_returns is shape (T, K)
    # Systematic = factor_returns @ betas.T
    systematic_returns = factor_returns.dot(betas.T)
    
    # Idiosyncratic residuals
    returns_clean = returns.fillna(0)
    residual_returns = returns_clean - systematic_returns
    
    # Synthetic residual price series
    residual_prices = residual_returns.cumsum()
    
    return residual_returns, residual_prices
