# src/evaluation/evaluate_corum.py

"""
CORUM-based evaluation for protein-protein similarity methods.

This script evaluates whether predicted protein-pair similarity scores recover
known CORUM co-complex relationships.

It saves only concise summary outputs:

outputs/evaluation_CORUM/
├── method_comparison_metrics.csv
├── precision_enrichment_at_k.csv
├── complex_level_validation.csv
└── README_evaluation.txt

Example:

python -m src.evaluation.evaluate_corum \
  --overlap_genes outputs/dataset_overlap/overlap_proteomics_CORUM_genes.csv \
  --positive_pairs outputs/dataset_overlap/CORUM_positive_pairs_in_proteomics.csv \
  --complex_coverage outputs/dataset_overlap/CORUM_complex_coverage_in_proteomics.csv \
  --output_dir outputs/evaluation_CORUM \
  --k_values 500,1000,5000,10000 \
  --min_complex_size 3 \
  --n_random 1000
"""

import os
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    """
    Store pairs in sorted order so A-B and B-A are identical.
    """
    return tuple(sorted((str(a).strip(), str(b).strip())))


def read_gene_list(path: str) -> set[str]:
    """
    Load the proteomics-CORUM overlap gene/protein list.

    Expected input examples:
        overlap_proteomics_CORUM_genes.csv

    Flexible column handling:
        gene, protein, or first column.
    """
    df = pd.read_csv(path)

    if "gene" in df.columns:
        col = "gene"
    elif "protein" in df.columns:
        col = "protein"
    else:
        col = df.columns[0]

    genes = df[col].astype(str).str.strip()
    genes = genes[(genes != "") & (genes.str.lower() != "nan")]

    return set(genes.tolist())


def load_protein_index(path: str) -> list[str]:
    """
    Load protein order corresponding to rows/columns of a similarity matrix.
    """
    df = pd.read_csv(path)

    if "protein" in df.columns:
        col = "protein"
    else:
        col = df.columns[0]

    proteins = df[col].astype(str).str.strip()
    proteins = proteins[(proteins != "") & (proteins.str.lower() != "nan")]

    return proteins.tolist()


def load_positive_pairs(
    path: str,
    eval_proteins: set[str],
) -> set[tuple[str, str]]:
    """
    Load CORUM-derived positive co-complex pairs.

    The input should contain:
        protein_1, protein_2

    Pairs are restricted to the CORUM/proteomics evaluation universe.
    """
    df = pd.read_csv(path)

    if not {"protein_1", "protein_2"}.issubset(df.columns):
        raise ValueError(
            f"{path} must contain protein_1 and protein_2 columns. "
            f"Found columns: {df.columns.tolist()}"
        )

    positives = set()

    for _, row in df.iterrows():
        p1 = str(row["protein_1"]).strip()
        p2 = str(row["protein_2"]).strip()

        if p1 == p2:
            continue

        if p1 in eval_proteins and p2 in eval_proteins:
            positives.add(canonical_pair(p1, p2))

    return positives


def parse_gene_list_field(x) -> list[str]:
    """
    Parse a semicolon-separated gene field.
    """
    if pd.isna(x):
        return []

    return [
        g.strip()
        for g in str(x).split(";")
        if g.strip() and g.strip().lower() not in {"nan", "none"}
    ]


# ============================================================
# Evaluation pair generation
# ============================================================

def build_eval_scores(
    sim: np.ndarray,
    proteins: list[str],
    eval_proteins: set[str],
    positive_pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    """
    Build compact pair-level data for one similarity matrix.

    Evaluation universe:
        all valid pairs among proteins in proteomics ∩ CORUM ∩ matrix proteins

    label = 1:
        CORUM-positive co-complex pair

    label = 0:
        non-CORUM background pair, not guaranteed true negative

    The returned DataFrame is kept in memory only. It is not saved by default
    because it can be large.
    """
    protein_to_idx = {p: i for i, p in enumerate(proteins)}
    eval_available = sorted([p for p in eval_proteins if p in protein_to_idx])

    rows = []

    for p1, p2 in combinations(eval_available, 2):
        i = protein_to_idx[p1]
        j = protein_to_idx[p2]

        score = sim[i, j]

        if np.isnan(score):
            continue

        label = 1 if canonical_pair(p1, p2) in positive_pairs else 0

        rows.append({
            "protein_1": p1,
            "protein_2": p2,
            "score": float(score),
            "label": label,
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise ValueError("No valid evaluation pairs were created.")

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    return df


# ============================================================
# Global pair-level metrics
# ============================================================

def compute_at_k(
    eval_df: pd.DataFrame,
    k_values: list[int],
    method: str,
) -> pd.DataFrame:
    """
    Compute precision@K, recall@K, and enrichment@K.
    """
    n_total = len(eval_df)
    n_pos = int(eval_df["label"].sum())
    background_rate = n_pos / n_total if n_total > 0 else np.nan

    rows = []

    for k in k_values:
        k_eff = min(k, n_total)
        top = eval_df.head(k_eff)

        pos_at_k = int(top["label"].sum())
        precision = pos_at_k / k_eff if k_eff > 0 else np.nan
        recall = pos_at_k / n_pos if n_pos > 0 else np.nan
        enrichment = precision / background_rate if background_rate > 0 else np.nan

        rows.append({
            "method": method,
            "K": k,
            "effective_K": k_eff,
            "positive_at_K": pos_at_k,
            "precision_at_K": precision,
            "recall_at_K": recall,
            "enrichment_at_K": enrichment,
        })

    return pd.DataFrame(rows)


def compute_method_metrics(
    eval_df: pd.DataFrame,
    method: str,
    k_values: list[int],
) -> tuple[dict, pd.DataFrame]:
    """
    Compute global pair-level metrics for one method.
    """
    y_true = eval_df["label"].values.astype(int)
    y_score = eval_df["score"].values.astype(float)

    n_eval_pairs = len(eval_df)
    n_positive_pairs = int(y_true.sum())
    n_background_pairs = n_eval_pairs - n_positive_pairs
    background_positive_rate = (
        n_positive_pairs / n_eval_pairs if n_eval_pairs > 0 else np.nan
    )

    if n_positive_pairs == 0 or n_background_pairs == 0:
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

    pos_scores = eval_df.loc[eval_df["label"] == 1, "score"]
    bg_scores = eval_df.loc[eval_df["label"] == 0, "score"]
    positive_ranks = eval_df.loc[eval_df["label"] == 1, "rank"]

    at_k_df = compute_at_k(eval_df, k_values, method)

    metrics = {
        "method": method,
        "n_eval_pairs": n_eval_pairs,
        "n_positive_pairs": n_positive_pairs,
        "n_background_pairs": n_background_pairs,
        "background_positive_rate": background_positive_rate,
        "AUROC": auroc,
        "AUPRC": auprc,
        "mean_positive_score": pos_scores.mean(),
        "mean_background_score": bg_scores.mean(),
        "median_positive_score": pos_scores.median(),
        "median_background_score": bg_scores.median(),
        "mean_positive_rank": positive_ranks.mean(),
        "median_positive_rank": positive_ranks.median(),
    }

    for _, row in at_k_df.iterrows():
        k = int(row["K"])
        metrics[f"precision@{k}"] = row["precision_at_K"]
        metrics[f"recall@{k}"] = row["recall_at_K"]
        metrics[f"enrichment@{k}"] = row["enrichment_at_K"]

    return metrics, at_k_df


# ============================================================
# Complex-level validation
# ============================================================

def get_within_scores(
    genes: list[str],
    sim: np.ndarray,
    protein_to_idx: dict[str, int],
) -> np.ndarray:
    """
    Extract all valid within-set pairwise scores for a gene set.
    """
    genes = [g for g in genes if g in protein_to_idx]
    scores = []

    for p1, p2 in combinations(genes, 2):
        s = sim[protein_to_idx[p1], protein_to_idx[p2]]
        if not np.isnan(s):
            scores.append(float(s))

    return np.array(scores, dtype=float)


def complex_level_validation(
    complex_coverage_path: str,
    sim: np.ndarray,
    proteins: list[str],
    eval_proteins: set[str],
    method: str,
    min_complex_size: int,
    n_random: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Complex-level validation.

    For each well-covered CORUM complex:
        1. compute mean within-complex similarity
        2. sample random protein sets with the same size
        3. compute z-score and empirical p-value

    Random baseline:
        size-matched random protein sets from the evaluation universe
    """
    rng = np.random.default_rng(random_state)

    protein_to_idx = {p: i for i, p in enumerate(proteins)}
    eval_available = sorted([p for p in eval_proteins if p in protein_to_idx])

    if len(eval_available) < min_complex_size:
        return pd.DataFrame()

    complex_df = pd.read_csv(complex_coverage_path)

    if "matched_genes" not in complex_df.columns:
        raise ValueError(
            f"{complex_coverage_path} must contain matched_genes column. "
            f"Found columns: {complex_df.columns.tolist()}"
        )

    rows = []

    for _, row in complex_df.iterrows():
        complex_id = row.get("ComplexID", None)
        complex_name = row.get("ComplexName", None)

        genes = parse_gene_list_field(row["matched_genes"])
        genes = [g for g in genes if g in eval_proteins and g in protein_to_idx]

        n = len(genes)

        if n < min_complex_size:
            continue

        observed_scores = get_within_scores(genes, sim, protein_to_idx)

        if len(observed_scores) == 0:
            continue

        observed_mean = float(np.mean(observed_scores))
        observed_median = float(np.median(observed_scores))

        random_means = []

        for _ in range(n_random):
            random_genes = rng.choice(eval_available, size=n, replace=False).tolist()
            random_scores = get_within_scores(random_genes, sim, protein_to_idx)

            if len(random_scores) > 0:
                random_means.append(float(np.mean(random_scores)))

        random_means = np.array(random_means, dtype=float)

        random_mean = float(np.mean(random_means)) if len(random_means) > 0 else np.nan
        random_std = float(np.std(random_means)) if len(random_means) > 0 else np.nan

        if len(random_means) > 0 and random_std > 0:
            z_score = float((observed_mean - random_mean) / random_std)
        else:
            z_score = np.nan

        if len(random_means) > 0:
            empirical_p = float(
                (np.sum(random_means >= observed_mean) + 1)
                / (len(random_means) + 1)
            )
        else:
            empirical_p = np.nan

        rows.append({
            "method": method,
            "ComplexID": complex_id,
            "ComplexName": complex_name,
            "n_matched_proteins": n,
            "n_valid_pairs": len(observed_scores),
            "observed_mean_score": observed_mean,
            "observed_median_score": observed_median,
            "random_mean_score": random_mean,
            "random_std_score": random_std,
            "z_score": z_score,
            "empirical_p_value": empirical_p,
        })

    result = pd.DataFrame(rows)

    if len(result) > 0:
        result = result.sort_values(
            ["method", "z_score", "observed_mean_score"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

    return result


# ============================================================
# Method list
# ============================================================

def default_methods() -> list[dict]:
    """
    Similarity matrices to evaluate.

    Missing files are skipped automatically.
    """
    return [
        {
            "name": "pearson",
            "matrix": "outputs/similarity_correlation/pearson.npy",
            "protein_index": "outputs/similarity_correlation/protein_index.csv",
        },
        {
            "name": "spearman",
            "matrix": "outputs/similarity_correlation/spearman.npy",
            "protein_index": "outputs/similarity_correlation/protein_index.csv",
        },
        {
            "name": "pearson_adjusted",
            "matrix": "outputs/similarity_correlation/pearson_adjusted.npy",
            "protein_index": "outputs/similarity_correlation/protein_index.csv",
        },
        {
            "name": "spearman_adjusted",
            "matrix": "outputs/similarity_correlation/spearman_adjusted.npy",
            "protein_index": "outputs/similarity_correlation/protein_index.csv",
        },
        {
            "name": "combined_pearson_jaccard",
            "matrix": "outputs/similarity_correlation/combined_pearson_jaccard.npy",
            "protein_index": "outputs/similarity_correlation/protein_index.csv",
        },
        {
            "name": "combined_spearman_jaccard",
            "matrix": "outputs/similarity_correlation/combined_spearman_jaccard.npy",
            "protein_index": "outputs/similarity_correlation/protein_index.csv",
        },
        {
            "name": "intensity_rbf",
            "matrix": "outputs/similarity_correlation/intensity_rbf.npy",
            "protein_index": "outputs/similarity_correlation/protein_index.csv",
        },
        {
            "name": "composite_rbf_jaccard",
            "matrix": "outputs/similarity_correlation/composite_rbf_jaccard.npy",
            "protein_index": "outputs/similarity_correlation/protein_index.csv",
        },
        {
            "name": "autoencoder_cosine",
            "matrix": "outputs/similarity_autoencoder/autoencoder_cosine.npy",
            "protein_index": "outputs/similarity_autoencoder/protein_index.csv",
        },
        {
            "name": "transformer_cosine",
            "matrix": "outputs/similarity_transformer/transformer_cosine.npy",
            "protein_index": "outputs/similarity_transformer/protein_index.csv",
        },
    ]


# ============================================================
# Main
# ============================================================

def main(args):
    ensure_dir(args.output_dir)

    print("── Load CORUM evaluation universe ──")
    eval_proteins = read_gene_list(args.overlap_genes)
    positive_pairs = load_positive_pairs(args.positive_pairs, eval_proteins)

    print(f"  Evaluation proteins, proteomics ∩ CORUM: {len(eval_proteins)}")
    print(f"  CORUM-positive co-complex pairs: {len(positive_pairs)}")

    k_values = [int(k) for k in args.k_values.split(",")]

    metrics_rows = []
    at_k_rows = []
    complex_rows = []

    for method in default_methods():
        name = method["name"]
        matrix_path = method["matrix"]
        protein_index_path = method["protein_index"]

        if not os.path.exists(matrix_path) or not os.path.exists(protein_index_path):
            print(f"Skip {name}: missing matrix or protein_index")
            continue

        print(f"\n── Evaluate: {name} ──")

        sim = np.load(matrix_path)
        proteins = load_protein_index(protein_index_path)

        if sim.shape[0] != sim.shape[1]:
            raise ValueError(f"{name}: similarity matrix is not square: {sim.shape}")

        if sim.shape[0] != len(proteins):
            raise ValueError(
                f"{name}: matrix shape {sim.shape}, "
                f"protein index length {len(proteins)}"
            )

        eval_df = build_eval_scores(
            sim=sim,
            proteins=proteins,
            eval_proteins=eval_proteins,
            positive_pairs=positive_pairs,
        )

        n_valid_pos = int(eval_df["label"].sum())
        print(f"  Valid evaluation pairs: {len(eval_df)}")
        print(f"  Valid CORUM-positive pairs: {n_valid_pos}")

        if n_valid_pos == 0:
            print(f"  Warning: no valid CORUM-positive pairs for {name}")

        metrics, at_k = compute_method_metrics(eval_df, name, k_values)
        metrics_rows.append(metrics)
        at_k_rows.append(at_k)

        report_k = 1000 if 1000 in k_values else k_values[-1]
        print(
            f"  AUPRC={metrics['AUPRC']:.6f} | "
            f"AUROC={metrics['AUROC']:.6f} | "
            f"precision@{report_k}={metrics.get(f'precision@{report_k}', np.nan):.6f} | "
            f"enrichment@{report_k}={metrics.get(f'enrichment@{report_k}', np.nan):.2f}"
        )

        complex_df = complex_level_validation(
            complex_coverage_path=args.complex_coverage,
            sim=sim,
            proteins=proteins,
            eval_proteins=eval_proteins,
            method=name,
            min_complex_size=args.min_complex_size,
            n_random=args.n_random,
            random_state=args.random_state,
        )

        if len(complex_df) > 0:
            complex_rows.append(complex_df)
            print(f"  Complex-level rows: {len(complex_df)}")
        else:
            print("  Complex-level rows: 0")

    print("\n── Save concise outputs ──")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(args.output_dir, "method_comparison_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved {metrics_path}")

    if len(at_k_rows) > 0:
        at_k_df = pd.concat(at_k_rows, ignore_index=True)
        at_k_path = os.path.join(args.output_dir, "precision_enrichment_at_k.csv")
        at_k_df.to_csv(at_k_path, index=False)
        print(f"  Saved {at_k_path}")

    if len(complex_rows) > 0:
        complex_all = pd.concat(complex_rows, ignore_index=True)
        complex_path = os.path.join(args.output_dir, "complex_level_validation.csv")
        complex_all.to_csv(complex_path, index=False)
        print(f"  Saved {complex_path}")

    readme_path = os.path.join(args.output_dir, "README_evaluation.txt")
    with open(readme_path, "w") as f:
        f.write(
            "CORUM evaluation outputs\n"
            "========================\n\n"
            "method_comparison_metrics.csv:\n"
            "  Global pair-level metrics for each similarity method.\n\n"
            "precision_enrichment_at_k.csv:\n"
            "  Precision, recall, and enrichment at each K.\n\n"
            "complex_level_validation.csv:\n"
            "  Complex-level validation for well-covered CORUM complexes.\n\n"
            "Interpretation:\n"
            "  CORUM-positive pairs are known co-complex positives.\n"
            "  Non-CORUM pairs are background pairs, not guaranteed true negatives.\n"
            "  Enrichment@K compares top-K precision to the background CORUM-positive rate.\n"
        )
    print(f"  Saved {readme_path}")

    print(f"\nSaved concise evaluation outputs to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overlap_genes",
        type=str,
        default="outputs/dataset_overlap/overlap_proteomics_CORUM_genes.csv",
    )

    parser.add_argument(
        "--positive_pairs",
        type=str,
        default="outputs/dataset_overlap/CORUM_positive_pairs_in_proteomics.csv",
    )

    parser.add_argument(
        "--complex_coverage",
        type=str,
        default="outputs/dataset_overlap/CORUM_complex_coverage_in_proteomics.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation_CORUM",
    )

    parser.add_argument(
        "--k_values",
        type=str,
        default="100,500,1000,5000,10000",
    )

    parser.add_argument(
        "--min_complex_size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--n_random",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    main(args)