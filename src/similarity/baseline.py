# src/similarity/baseline.py
#
# Example usage:
#   python src/similarity/baseline.py

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ============================================================
# Core NaN-aware engine
# ============================================================

def _nan_aware_pearson(X: np.ndarray, min_periods: int) -> np.ndarray:
    """
    Compute pairwise Pearson correlation using only jointly observed patients per pair.

    For each protein pair (i, j), the correlation uses only patients where
    both proteins have a non-NaN value. Pairs with fewer than min_periods
    jointly observed patients are set to NaN.
    """
    M   = (~np.isnan(X)).astype(np.float32)
    X_f = np.where(np.isnan(X), 0.0, X).astype(np.float32)

    # All matrices are (P, P); computed via outer products over the patient axis.
    # SumX[i,j]  = sum of X[i] over patients jointly observed by protein i and j
    # SumX2[i,j] = sum of X[i]^2 over jointly observed patients
    # SumXY[i,j] = sum of X[i]*X[j] over jointly observed patients
    N     = M @ M.T
    SumX  = X_f @ M.T
    SumX2 = (X_f ** 2) @ M.T
    SumXY = X_f @ X_f.T

    num = N * SumXY - SumX * SumX.T
    var_x = N * SumX2 - SumX ** 2
    den = np.sqrt(np.clip(var_x * var_x.T, 0.0, None))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = np.where(den > 0, num / den, np.nan)

    r[N < min_periods] = np.nan
    np.fill_diagonal(r, 1.0)
    return r


def _nan_rankdata_rows(X: np.ndarray) -> np.ndarray:
    """
    Rank each protein's observed values across patients.

    Ranks are assigned globally over each protein's own observed patients,
    not restricted to the jointly observed set per pair. This is an
    approximation: for proteins observed in all patients the result is exact;
    for proteins with missing values, the ranks can differ slightly from
    true per-pair ranks. Exact per-pair ranking is O(P^2) and impractical
    at 12K proteins.
    """
    out = np.full_like(X, np.nan, dtype=np.float32)
    for i in range(X.shape[0]):
        row  = X[i]
        mask = ~np.isnan(row)
        out[i, mask] = rankdata(row[mask]).astype(np.float32)
    return out


# ============================================================
# Public similarity functions
# ============================================================

def pearson_similarity(df: pd.DataFrame, min_periods: int = 1) -> np.ndarray:
    """
    NaN-aware Pearson correlation between all protein pairs.

    Each pair uses only patients where both proteins are observed.
    min_periods=1 means all pairs with at least one shared observation
    are computed; pairs with zero shared patients remain NaN.
    """
    return _nan_aware_pearson(df.values.astype(np.float32), min_periods)


def spearman_similarity(df: pd.DataFrame, min_periods: int = 1) -> np.ndarray:
    """
    NaN-aware Spearman correlation between all protein pairs.

    Ranks are computed per protein over its own observed patients (see
    _nan_rankdata_rows), then Pearson is applied to those ranks over
    jointly observed patients. Exact for fully observed proteins;
    approximate for proteins with missing values.
    """
    X_ranked = _nan_rankdata_rows(df.values)
    return _nan_aware_pearson(X_ranked, min_periods)


def overlap_matrix(df: pd.DataFrame) -> np.ndarray:
    M = (~df.isna()).astype(np.float32).values
    return (M @ M.T).astype(np.int32)


# ============================================================
# Diagnostics
# ============================================================

def summarise_overlap(df: pd.DataFrame) -> None:
    overlap = overlap_matrix(df)
    vals    = overlap[np.triu_indices(len(overlap), k=1)]
    print(f"  Min shared patients:     {vals.min()}")
    print(f"  Mean shared patients:    {vals.mean():.1f}")
    print(f"  Median shared patients:  {np.median(vals):.1f}")
    print(f"  Max shared patients:     {vals.max()}")
    print(f"  q10/q90 shared patients: {np.quantile(vals, 0.10):.1f} / {np.quantile(vals, 0.90):.1f}")


def summarise_similarity(name: str, mat: np.ndarray) -> None:
    vals  = mat[np.triu_indices(len(mat), k=1)]
    valid = vals[~np.isnan(vals)]

    if len(valid) == 0:
        print(f"  {name} | all pairwise values are NaN")
        return

    print(
        f"  {name} | "
        f"NaNs: {np.isnan(vals).sum()} | "
        f"mean: {valid.mean():.3f} | "
        f"std: {valid.std():.3f} | "
        f"range: [{valid.min():.3f}, {valid.max():.3f}] | "
        f"q95: {np.quantile(valid, 0.95):.3f} | "
        f"q99: {np.quantile(valid, 0.99):.3f}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    os.makedirs('outputs/similarity', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)

    print("── Load ──")
    df = pd.read_csv('data/processed/proteomics_data_processed.csv', index_col=0)
    print(f"  Loaded: {df.shape[0]} proteins × {df.shape[1]} patients")

    print("── Overlap diagnostics ──")
    summarise_overlap(df)

    # min_periods=1: for each protein pair, use all jointly observed patients.
    # Pairs with zero shared patients (no overlap at all) remain NaN.
    # Output is always the full (n_proteins x n_proteins) matrix.
    min_p = 10

    print("── Pearson ──")
    pearson_sim = pearson_similarity(df, min_periods=min_p)
    summarise_similarity('Pearson', pearson_sim)

    print("── Spearman ──")
    spearman_sim = spearman_similarity(df, min_periods=min_p)
    summarise_similarity('Spearman', spearman_sim)

    print("── Save ──")
    np.save('outputs/similarity/pearson.npy', pearson_sim)   # raw, NaN preserved
    np.save('outputs/similarity/spearman.npy', spearman_sim)
    np.save('data/processed/protein_overlap.npy', overlap_matrix(df))  # confidence matrix
    # protein_index is not saved here — use data/processed/protein_index.csv as the canonical source
    print("  Saved pearson.npy, spearman.npy, overlap.npy")
    