"""
graph_discovery.py -- Graph AI (Spectral Link Prediction) for Pair Selection
Replaces brute-force cointegration with a ML-driven dynamic financial graph.
"""

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from xgboost import XGBRegressor
import itertools

from src.cointegration import engle_granger_test

def build_adjacency_matrix(returns: pd.DataFrame, threshold: float = 0.7) -> np.ndarray:
    """
    Builds an adjacency matrix from the correlation matrix.
    Edges exist if absolute correlation > threshold.
    """
    corr = returns.corr().fillna(0).values
    adj = (np.abs(corr) > threshold).astype(float)
    np.fill_diagonal(adj, 0.0)
    return adj

def compute_spectral_embeddings(adj: np.ndarray, dim: int = 4) -> np.ndarray:
    """
    Computes Laplacian Eigenmaps (Spectral Embeddings) for nodes in the graph.
    """
    # Compute normalized Laplacian
    lap = laplacian(adj, normed=True)
    
    # Eigendecomposition
    try:
        eigenvals, eigenvecs = eigh(lap)
        # Skip the first eigenvector (which is constant for connected components)
        embeddings = eigenvecs[:, 1:dim+1]
    except Exception:
        # Fallback to random embeddings if matrix is singular
        embeddings = np.random.randn(adj.shape[0], dim)
        
    return embeddings

def build_edge_features(embeddings: np.ndarray, node_indices: list) -> pd.DataFrame:
    """
    Creates edge features by concatenating node embeddings.
    """
    features = []
    pairs = []
    
    for i, j in itertools.combinations(range(len(node_indices)), 2):
        emb_i = embeddings[i]
        emb_j = embeddings[j]
        
        # Edge features: Concatenate, Element-wise product, Absolute difference
        edge_feat = np.concatenate([
            emb_i, 
            emb_j, 
            emb_i * emb_j,
            np.abs(emb_i - emb_j)
        ])
        features.append(edge_feat)
        pairs.append((node_indices[i], node_indices[j]))
        
    cols = [f"emb_i_{k}" for k in range(embeddings.shape[1])] + \
           [f"emb_j_{k}" for k in range(embeddings.shape[1])] + \
           [f"prod_{k}" for k in range(embeddings.shape[1])] + \
           [f"diff_{k}" for k in range(embeddings.shape[1])]
           
    return pd.DataFrame(features, columns=cols), pairs

def get_forward_labels(prices: pd.DataFrame, pairs: list) -> pd.Series:
    """
    Computes the forward Engle-Granger p-value for each pair to act as the ground truth label.
    Lower p-value = higher probability of mean reversion.
    We convert p-value to a score: 1 - p_value. Higher is better.
    """
    scores = []
    for y_idx, x_idx in pairs:
        y_prices = prices[y_idx]
        x_prices = prices[x_idx]
        eg_res = engle_granger_test(y_prices, x_prices)
        
        # Invert p-value so higher = better target
        score = 1.0 - eg_res["p_value"]
        scores.append(score)
        
    return pd.Series(scores)

def gnn_pair_discovery(
    prices: pd.DataFrame, 
    candidate_pairs: list, 
    train_window: int = 504, 
    emb_dim: int = 4, 
    corr_threshold: float = 0.7,
    top_n: int = 10
) -> list:
    """
    Executes the Graph AI link prediction pipeline to select the best pairs.
    """
    print(f"  [GNN] Training Link Predictor on {train_window} days of historical data...")
    
    # We need Train and Test periods
    # Train Period: Start to (End - 252 days) -> Generates Embeddings
    # Label Period: (End - 252) to End -> Generates True Cointegration scores
    # Test Period (Current): (End - 252) to End -> Generates Embeddings
    # Inference: Predict future cointegration using Test Embeddings
    
    total_days = len(prices)
    if total_days < train_window + 252:
        print("  [warn] Not enough data for GNN training. Falling back to simple correlation.")
        return candidate_pairs[:top_n]
        
    # --- TRAINING PHASE ---
    t_start = total_days - train_window - 252
    t_mid = total_days - 252
    
    train_prices = prices.iloc[t_start:t_mid]
    train_rets = train_prices.pct_change().fillna(0)
    
    label_prices = prices.iloc[t_mid:]
    
    # 1. Build Graph & Embeddings (T-1)
    train_adj = build_adjacency_matrix(train_rets, threshold=corr_threshold)
    train_emb = compute_spectral_embeddings(train_adj, dim=emb_dim)
    
    # 2. Build Edge Features (T-1)
    tickers = list(prices.columns)
    X_train, _ = build_edge_features(train_emb, tickers)
    
    # 3. Get True Labels (T)
    # We only compute labels for all possible edges
    all_edges = list(itertools.combinations(tickers, 2))
    print(f"  [GNN] Computing forward labels for {len(all_edges)} edges...")
    y_train = get_forward_labels(label_prices, all_edges)
    
    # 4. Train Link Predictor
    print(f"  [GNN] Training XGBoost Link Predictor...")
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # --- INFERENCE PHASE ---
    print(f"  [GNN] Generating current Graph Embeddings...")
    current_prices = prices.iloc[t_mid:]
    current_rets = current_prices.pct_change().fillna(0)
    
    current_adj = build_adjacency_matrix(current_rets, threshold=corr_threshold)
    current_emb = compute_spectral_embeddings(current_adj, dim=emb_dim)
    
    X_test, test_pairs = build_edge_features(current_emb, tickers)
    
    print(f"  [GNN] Predicting future pair profitability...")
    predictions = model.predict(X_test)
    
    # Rank pairs by predicted score
    results = pd.DataFrame({
        "ticker_y": [p[0] for p in test_pairs],
        "ticker_x": [p[1] for p in test_pairs],
        "pred_score": predictions
    })
    
    results = results.sort_values(by="pred_score", ascending=False)
    
    top_pairs = []
    for _, row in results.head(top_n).iterrows():
        top_pairs.append({"ticker_y": row["ticker_y"], "ticker_x": row["ticker_x"]})
        
    print(f"  [GNN] Top predicted pairs:")
    for p in top_pairs:
        print(f"      {p['ticker_y']} / {p['ticker_x']}")
        
    return top_pairs
