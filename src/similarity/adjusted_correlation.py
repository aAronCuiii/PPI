# src/similarity/adjusted_correlation.py
#
# Example usage:
#   python src/similarity/adjusted_correlation.py

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from src.similarity.baseline import (
    pearson_similarity,
    spearman_similarity,
    overlap_matrix,
    )


# ============================================================
# Missingness / detection-pattern similarities
# ============================================================

def detection_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Binary detection matrix.

    M[i, p] = 1 if protein i is quantified in patient p.
    M[i, p] = 0 if protein i is NaN in patient p.
    """
    return (~df.isna()).astype(np.float32).values


def union_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Number of patients where at least one of the two proteins is observed.
    """
    M = detection_matrix(df)
    observed_per_protein = M.sum(axis=1, keepdims=True)
    overlap = M @ M.T
    union = observed_per_protein + observed_per_protein.T - overlap
    return union.astype(np.int32)


def jaccard_detection_similarity(df: pd.DataFrame) -> np.ndarray:
    """
    Jaccard similarity between protein detection patterns.

    Jaccard(i, j) =
        number of patients where both proteins are observed
        /
        number of patients where at least one protein is observed

    This treats NaN/detection pattern as informative.
    """
    M = detection_matrix(df)

    intersection = M @ M.T
    observed_per_protein = M.sum(axis=1, keepdims=True)
    union = observed_per_protein + observed_per_protein.T - intersection

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jaccard = np.where(union > 0, intersection / union, np.nan)

    np.fill_diagonal(jaccard, 1.0)
    return jaccard.astype(np.float32)


def detection_cosine_similarity(df: pd.DataFrame) -> np.ndarray:
    """
    Cosine similarity between binary detection patterns.

    This is another way to measure whether two proteins are detected in
    similar patient subsets.
    """
    M = detection_matrix(df)
    dot = M @ M.T
    norm = np.sqrt(np.clip(np.diag(dot), 0.0, None))

    den = np.outer(norm, norm)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = np.where(den > 0, dot / den, np.nan)

    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


# ============================================================
# Reliability-adjusted and combined scores
# ============================================================

def overlap_reliability_weight(
    overlap: np.ndarray,
    n_patients: int,
    min_periods: int,
) -> np.ndarray:
    """
    Convert number of shared observed patients into a reliability weight.

    weight = sqrt(shared_patients / total_patients)

    Pairs with shared patients below min_periods are set to NaN.
    """
    weight = np.sqrt(overlap.astype(np.float32) / float(n_patients))
    weight[overlap < min_periods] = np.nan
    np.fill_diagonal(weight, 1.0)
    return weight.astype(np.float32)


def adjusted_correlation(
    corr: np.ndarray,
    overlap: np.ndarray,
    n_patients: int,
    min_periods: int,
) -> np.ndarray:
    """
    Downweight correlations estimated from fewer shared patients.

    adjusted_corr(i, j) =
        corr(i, j) * sqrt(shared_patients(i, j) / total_patients)

    This keeps strong correlations but penalizes scores estimated from
    very small patient overlap.
    """
    weight = overlap_reliability_weight(
        overlap=overlap,
        n_patients=n_patients,
        min_periods=min_periods,
    )

    adjusted = corr * weight   # NaN propagates: NaN * weight = NaN, corr * NaN = NaN
    np.fill_diagonal(adjusted, 1.0)

    return adjusted.astype(np.float32)


def combined_abundance_missingness_similarity(
    corr_adjusted: np.ndarray,
    detection_sim: np.ndarray,
    lambda_missingness: float = 0.2,
) -> np.ndarray:
    """
    Combine abundance correlation and missingness/detection similarity.

    combined_score =
        adjusted abundance correlation + lambda * detection similarity

    Since adjusted correlation can be negative while Jaccard is non-negative,
    this score is most useful for ranking positive association candidates.

    Use CORUM enrichment later to compare lambda values.
    """
    # NaN propagates: NaN + anything = NaN, so pairs invalid in corr_adjusted
    # are not rescued solely by detection similarity.
    combined = corr_adjusted + lambda_missingness * detection_sim

    np.fill_diagonal(combined, 1.0)
    return combined.astype(np.float32)


# ============================================================
# Summaries
# ============================================================

def summarise_overlap(df: pd.DataFrame) -> None:
    overlap = overlap_matrix(df)
    vals = overlap[np.triu_indices(len(overlap), k=1)]

    print(f"  Min shared patients:    {vals.min()}")
    print(f"  Mean shared patients:   {vals.mean():.1f}")
    print(f"  Median shared patients: {np.median(vals):.1f}")
    print(f"  Max shared patients:    {vals.max()}")
    print(f"  q10/q90 shared patients: {np.quantile(vals, 0.10):.1f} / {np.quantile(vals, 0.90):.1f}")


def summarise_similarity(name: str, mat: np.ndarray) -> None:
    vals = mat[np.triu_indices(len(mat), k=1)]
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

if __name__ == "__main__":
    output_dir = "outputs/similarity_reversed"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    print("── Load ──")
    df = pd.read_csv("data/processed/proteomics_data_reversed.csv", index_col=0)
    print(f"  Loaded: {df.shape[0]} proteins × {df.shape[1]} patients")

    # Keep permissive because missingness may be meaningful.
    # This only removes nearly uninformative proteins with very few observations.
    min_observed_protein = 10

    proteins = df.index.tolist()
    n_patients = df.shape[1]

    print("── Overlap diagnostics ──")
    overlap = overlap_matrix(df)
    summarise_overlap(df)

    # Pairwise minimum overlap.
    # Start with 10 because you want to keep sparse but potentially informative proteins.
    # Use 20 or 30 as sensitivity checks.
    min_pair_overlap = 10

    # Missingness contribution to combined score.
    # Tune later using CORUM enrichment.
    lambda_missingness = 0.2

    print("── Pearson ──")
    pearson_sim = pearson_similarity(df, min_periods=min_pair_overlap)
    summarise_similarity("Pearson raw", pearson_sim)

    print("── Spearman ──")
    spearman_sim = spearman_similarity(df, min_periods=min_pair_overlap)
    summarise_similarity("Spearman raw", spearman_sim)

    print("── Detection-pattern similarities ──")
    jaccard_sim = jaccard_detection_similarity(df)
    summarise_similarity("Jaccard detection", jaccard_sim)

    detection_cosine_sim = detection_cosine_similarity(df)
    summarise_similarity("Detection cosine", detection_cosine_sim)

    print("── Overlap-adjusted correlations ──")
    pearson_adjusted = adjusted_correlation(
        corr=pearson_sim,
        overlap=overlap,
        n_patients=n_patients,
        min_periods=min_pair_overlap,
    )
    summarise_similarity("Pearson adjusted", pearson_adjusted)

    spearman_adjusted = adjusted_correlation(
        corr=spearman_sim,
        overlap=overlap,
        n_patients=n_patients,
        min_periods=min_pair_overlap,
    )
    summarise_similarity("Spearman adjusted", spearman_adjusted)

    print("── Combined abundance + missingness similarities ──")
    combined_pearson_jaccard = combined_abundance_missingness_similarity(
        corr_adjusted=pearson_adjusted,
        detection_sim=jaccard_sim,
        lambda_missingness=lambda_missingness,
    )
    summarise_similarity("Combined Pearson + Jaccard", combined_pearson_jaccard)

    combined_spearman_jaccard = combined_abundance_missingness_similarity(
        corr_adjusted=spearman_adjusted,
        detection_sim=jaccard_sim,
        lambda_missingness=lambda_missingness,
    )
    summarise_similarity("Combined Spearman + Jaccard", combined_spearman_jaccard)

    print("── Save matrices ──")

    np.save(os.path.join(output_dir, "pearson_adjusted.npy"), pearson_adjusted)
    np.save(os.path.join(output_dir, "spearman_adjusted.npy"), spearman_adjusted)
    np.save(os.path.join(output_dir, "combined_pearson_jaccard.npy"), combined_pearson_jaccard)
    np.save(os.path.join(output_dir, "combined_spearman_jaccard.npy"), combined_spearman_jaccard)
    # protein_index is not saved here — use data/processed/protein_index.csv as the canonical source

    print("── Done ──")
    print(
        "  Saved raw correlations, detection-pattern similarities, "
        "overlap-adjusted correlations, combined scores, overlap matrix, "
        "and protein index."
    )