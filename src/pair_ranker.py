"""
pair_ranker.py -- Composite scoring and ranking for pair selection.

Ranks candidate pairs using a weighted composite score across multiple
screening criteria. The best pair has the highest composite score.
"""

import numpy as np
import pandas as pd


def rank_pairs(
    screening_df: pd.DataFrame,
    var_ratio_weight: float = 0.30,
    half_life_weight: float = 0.25,
    johansen_weight: float = 0.25,
    corr_stability_weight: float = 0.20,
    vr_threshold: float = 1.0,
    min_half_life: float = 5,
    max_half_life: float = 60,
) -> pd.DataFrame:
    """
    Rank pairs by composite score.

    Scoring (higher = better):
        VR:          (1.0 - VR), clamped to [0, 1] (lower VR = better MR)
        Half-life:   1.0 if in [min, max], penalized outside
        Johansen:    trace_stat / crit_95 (ratio > 1 = cointegrated)
        Corr stab:   1 - normalized_std (lower std = better)
    """
    df = screening_df.copy()

    if len(df) == 0:
        return df

    # -- Variance Ratio score: lower VR = better (< 1.0 is mean-reverting)
    df["vr_score"] = ((vr_threshold - df["var_ratio"]) / vr_threshold).clip(0, 1)

    # -- Half-life score: prefer within [min, max], penalize outside
    def hl_score(hl):
        if hl == float("inf") or np.isnan(hl):
            return 0.0
        if hl < min_half_life:
            return max(0, hl / min_half_life)  # Too fast
        if hl > max_half_life:
            return max(0, 1 - (hl - max_half_life) / max_half_life)  # Too slow
        ideal = np.sqrt(min_half_life * max_half_life)
        distance = abs(np.log(hl / ideal))
        return float(np.exp(-distance))

    df["hl_score"] = df["half_life"].apply(hl_score)

    # -- Johansen score: trace_stat / crit_95 (> 1 = reject null)
    df["joh_score"] = (df["johansen_trace"] / df["johansen_crit95"].clip(lower=1e-6)).clip(0, 3) / 3

    # -- Correlation stability score: lower std = better
    max_cs = df["corr_stability"].max()
    if max_cs > 0:
        df["corr_score"] = 1 - (df["corr_stability"] / max_cs)
    else:
        df["corr_score"] = 1.0

    # -- Composite score
    df["composite_score"] = (
        var_ratio_weight * df["vr_score"]
        + half_life_weight * df["hl_score"]
        + johansen_weight * df["joh_score"]
        + corr_stability_weight * df["corr_score"]
    )

    # Sort descending
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    return df


def print_ranking_report(ranked_df: pd.DataFrame) -> None:
    """Pretty-print the pair ranking results."""
    print("\n" + "=" * 85)
    print("  PAIR RANKING (by composite score)")
    print("=" * 85)
    print(f"  {'Rank':<5} {'Pair':<10} {'VR':>7} {'HL':>7} {'Joh':>7} "
          f"{'Corr':>7} {'COMPOSITE':>10}")
    print(f"  {'-'*5} {'-'*10} {'-'*7} {'-'*7} {'-'*7} "
          f"{'-'*7} {'-'*10}")

    for i, row in ranked_df.iterrows():
        pair = f"{row['ticker_y']}/{row['ticker_x']}"
        marker = " <-- BEST" if i == 0 else ""
        print(
            f"  {i+1:<5d} {pair:<10} "
            f"{row['vr_score']:>7.3f} "
            f"{row['hl_score']:>7.3f} "
            f"{row['joh_score']:>7.3f} "
            f"{row['corr_score']:>7.3f} "
            f"{row['composite_score']:>10.4f}{marker}"
        )

    print("=" * 85 + "\n")

    if len(ranked_df) > 0:
        best = ranked_df.iloc[0]
        print(f"  Selected pair: {best['ticker_y']} / {best['ticker_x']}  "
              f"(score: {best['composite_score']:.4f})")
        print(f"  VR(5): {best['var_ratio']:.3f}, "
              f"Half-life: {best['half_life']:.1f}d, "
              f"Hedge ratio: {best['hedge_ratio']:.4f}\n")
