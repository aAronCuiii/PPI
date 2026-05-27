# src/similarity/kernel.py

# Example usage:
#   python src/similarity/kernel.py \
#     --input data/processed/proteomics_data_reversed.csv \
#     --output_dir outputs/similarity2 \
#     --min_observed 10 \
#     --min_pair_overlap 10 \
#     --gamma 0.1 \
#     --alpha 0.8 \
#     --adjust_by_overlap

import os
import argparse
import warnings
import numpy as np
import pandas as pd


# ============================================================
# Loading and preprocessing
# ============================================================

def load_proteomics(path: str) -> pd.DataFrame:
    """
    Load protein x patient matrix.

    Supports CSV and tab-separated TXT.
    Rows are proteins.
    Columns are patients.
    """
    try:
        df = pd.read_csv(path, index_col=0)

        # Fallback for tab-separated text parsed as one column
        if df.shape[1] <= 1:
            df = pd.read_csv(path, index_col=0, sep="\t")

    except Exception:
        df = pd.read_csv(path, index_col=0, sep="\t")

    return df



def standardize_observed_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each patient column using observed values only.

    NaNs remain NaNs.
    """
    means = df.mean(axis=0, skipna=True)
    stds = df.std(axis=0, skipna=True)

    stds = stds.replace(0.0, np.nan)

    return (df - means) / stds


# ============================================================
# Detection and overlap
# ============================================================

def detection_matrix(X: np.ndarray) -> np.ndarray:
    """
    Binary matrix:
    1 if observed, 0 if NaN.
    """
    return (~np.isnan(X)).astype(np.float32)


def overlap_matrix_from_array(X: np.ndarray) -> np.ndarray:
    """
    Number of jointly observed patients for each protein pair.
    """
    M = detection_matrix(X)
    return (M @ M.T).astype(np.int32)


def mask_sparse_proteins(
    sim: np.ndarray,
    X: np.ndarray,
    min_observed: int,
) -> np.ndarray:
    """
    Set sim[i, j] = NaN if protein i or protein j has fewer than
    min_observed total observed patients across all patients.

    This is a per-protein filter, distinct from min_periods (which is a
    pair-level joint-overlap filter). Even if two sparse proteins happen
    to share enough jointly observed patients, their pair is still masked
    if either protein is too sparse on its own.

    The diagonal is restored to 1.0 after masking.
    """
    n_obs  = (~np.isnan(X)).sum(axis=1)  # (P,) observed patient count per protein
    sparse = n_obs < min_observed         # (P,) True where protein is too sparse

    out = sim.copy()
    out[sparse, :] = np.nan
    out[:, sparse] = np.nan
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


# ============================================================
# Kernels
# ============================================================

def intensity_rbf_kernel(
    X: np.ndarray,
    gamma: float = 0.5,
    min_periods: int = 10,
    adjust_by_overlap: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NaN-aware RBF kernel over observed dimensions only.

    For each pair (i, j):

        dist2(i, j) =
            mean over shared observed patients of (x_i - x_j)^2

        K(i, j) =
            exp(-gamma * dist2(i, j))

    Pairs with fewer than min_periods shared patients are set to NaN.

    If adjust_by_overlap=True:

        K(i, j) = K(i, j) * sqrt(shared_patients / total_patients)

    Returns
    -------
    K:
        RBF similarity matrix.
    dist2:
        Pairwise mean squared distance matrix.
    N:
        Pairwise shared observed patient counts.
    """
    X = X.astype(np.float32)

    M = (~np.isnan(X)).astype(np.float32)
    X_f = np.where(np.isnan(X), 0.0, X).astype(np.float32)

    N = M @ M.T

    # sum over jointly observed patients:
    # sum (x_i - x_j)^2
    # = sum x_i^2 + sum x_j^2 - 2 sum x_i x_j
    XX = (X_f ** 2) @ M.T
    YY = XX.T
    XY = X_f @ X_f.T

    ssd = XX + YY - 2.0 * XY
    ssd = np.clip(ssd, 0.0, None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist2 = np.where(N > 0, ssd / N, np.nan)

    dist2[N < min_periods] = np.nan
    np.fill_diagonal(dist2, 0.0)

    K = np.exp(-gamma * dist2)
    K[np.isnan(dist2)] = np.nan

    if adjust_by_overlap:
        weight = np.sqrt(N.astype(np.float32) / float(X.shape[1]))
        weight[N < min_periods] = np.nan
        np.fill_diagonal(weight, 1.0)

        K = K * weight
        K[np.isnan(weight)] = np.nan

    np.fill_diagonal(K, 1.0)

    return K.astype(np.float32), dist2.astype(np.float32), N.astype(np.int32)


def codetection_jaccard_kernel(
    X: np.ndarray,
) -> np.ndarray:
    """
    Jaccard kernel on observed/missing detection patterns.

    K(i, j) =
        number of patients where both proteins are observed
        /
        number of patients where at least one protein is observed
    """
    M = (~np.isnan(X)).astype(np.float32)

    intersect = M @ M.T
    observed = M.sum(axis=1, keepdims=True)
    union = observed + observed.T - intersect

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        K = np.where(union > 0, intersect / union, np.nan)

    np.fill_diagonal(K, 1.0)

    return K.astype(np.float32)


def codetection_cosine_kernel(
    X: np.ndarray,
) -> np.ndarray:
    """
    Cosine similarity on binary detection patterns.
    """
    M = (~np.isnan(X)).astype(np.float32)

    dot = M @ M.T
    norm = np.sqrt(np.clip(np.diag(dot), 0.0, None))
    den = np.outer(norm, norm)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        K = np.where(den > 0, dot / den, np.nan)

    np.fill_diagonal(K, 1.0)

    return K.astype(np.float32)


def composite_kernel(
    intensity_K: np.ndarray,
    codetection_K: np.ndarray,
    alpha: float = 0.8,
    preserve_intensity_nan: bool = True,
) -> np.ndarray:
    """
    Composite kernel:

        K = alpha * K_intensity + (1 - alpha) * K_codetection

    Recommended:
        alpha = 0.8

    This puts more weight on abundance closeness but still uses missingness signal.

    If preserve_intensity_nan=True, pairs invalid under intensity kernel remain NaN.
    This prevents pairs with too few shared quantitative observations from being
    ranked solely because of similar missingness patterns.
    """
    K = alpha * intensity_K + (1.0 - alpha) * codetection_K

    if preserve_intensity_nan:
        K[np.isnan(intensity_K)] = np.nan

    np.fill_diagonal(K, 1.0)

    return K.astype(np.float32)


# ============================================================
# Exports and summaries
# ============================================================

def summarise_similarity(name: str, K: np.ndarray) -> None:
    vals = K[np.triu_indices_from(K, k=1)]
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


def similarity_matrix_to_edge_table(
    sim: np.ndarray,
    proteins: list[str],
    overlap: np.ndarray | None = None,
    score_name: str = "score",
    top_k: int | None = None,
    min_score: float | None = None,
) -> pd.DataFrame:
    rows, cols = np.triu_indices_from(sim, k=1)
    scores = sim[rows, cols]

    mask = ~np.isnan(scores)

    if min_score is not None:
        mask &= scores >= min_score

    data = {
        "protein_1": np.array(proteins)[rows[mask]],
        "protein_2": np.array(proteins)[cols[mask]],
        score_name: scores[mask],
    }

    if overlap is not None:
        data["shared_patients"] = overlap[rows[mask], cols[mask]]

    edge_df = pd.DataFrame(data)
    edge_df = edge_df.sort_values(score_name, ascending=False).reset_index(drop=True)
    edge_df["rank"] = np.arange(1, len(edge_df) + 1)

    if top_k is not None:
        edge_df = edge_df.head(top_k).copy()

    return edge_df


# ============================================================
# Main
# ============================================================

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("── Load ──")
    df = load_proteomics(args.input)
    print(f"  Loaded: {df.shape[0]} proteins × {df.shape[1]} patients")

    proteins = df.index.tolist()

    print("── Standardize observed values ──")
    df_std = standardize_observed_values(df)
    X = df_std.values.astype(np.float32)

    print("── Intensity RBF kernel ──")
    Ki, dist2, overlap = intensity_rbf_kernel(
        X,
        gamma=args.gamma,
        min_periods=args.min_pair_overlap,
        adjust_by_overlap=args.adjust_by_overlap,
    )

    summarise_similarity("Intensity RBF", Ki)

    print("── Co-detection kernels ──")
    Kj = codetection_jaccard_kernel(X)
    Kc = codetection_cosine_kernel(X)

    summarise_similarity("Jaccard co-detection", Kj)
    summarise_similarity("Cosine co-detection", Kc)

    print("── Composite kernel ──")
    K_comp = composite_kernel(
        intensity_K=Ki,
        codetection_K=Kj,
        alpha=args.alpha,
        preserve_intensity_nan=True,
    )

    summarise_similarity("Composite RBF + Jaccard", K_comp)

    print("── Mask sparse proteins ──")
    n_obs   = (~np.isnan(X)).sum(axis=1)
    n_sparse = int((n_obs < args.min_observed).sum())
    print(f"  Proteins with < {args.min_observed} observed patients: {n_sparse} "
          f"→ all their pairwise similarities set to NaN")

    Ki     = mask_sparse_proteins(Ki,     X, args.min_observed)
    Kj     = mask_sparse_proteins(Kj,     X, args.min_observed)
    Kc     = mask_sparse_proteins(Kc,     X, args.min_observed)
    K_comp = mask_sparse_proteins(K_comp, X, args.min_observed)

    summarise_similarity("Intensity RBF (masked)",          Ki)
    summarise_similarity("Composite RBF + Jaccard (masked)", K_comp)

    print("── Save matrices ──")
    np.save(os.path.join(args.output_dir, "intensity_rbf.npy"), Ki)
    np.save(os.path.join(args.output_dir, "composite_rbf_jaccard.npy"), K_comp)
    # intensity_distance.npy (dist2) is not saved — it is a distance matrix, not a similarity
    # matrix, and would be misread by evaluate_corum_similarity.py as a similarity score.
    # protein_index is not saved here — use data/processed/protein_index.csv as the canonical source

    print("── Save ranked edge tables ──")
    print("── Done ──")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/proteomics_data_reversed.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/similarity_reversed",
    )

    parser.add_argument(
        "--min_observed",
        type=int,
        default=10,
        help=(
            "Per-protein threshold. If protein i or j has fewer than this many "
            "observed patients in total, sim(i, j) is set to NaN. "
            "Applied after all kernels are computed; proteins are never dropped from the matrix."
        ),
    )

    parser.add_argument(
        "--min_pair_overlap",
        type=int,
        default=10,
        help=(
            "Pair-level threshold. If proteins i and j share fewer than this many "
            "jointly observed patients, sim(i, j) is set to NaN."
        ),
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Composite kernel weight. alpha=1 uses only intensity RBF; alpha=0 uses only co-detection.",
    )

    parser.add_argument(
        "--adjust_by_overlap",
        action="store_true",
        help="If set, multiply intensity RBF by sqrt(shared patients / total patients).",
    )

    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--n_components",
        type=int,
        default=50,
    )

    args = parser.parse_args()
    main(args)