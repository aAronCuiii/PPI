"""Rank protein-pair similarity matrices into edge tables.

This script supports two common network construction views:

1. top-K ranked edges, e.g. the top 100,000 protein pairs.
2. thresholded edges, e.g. all pairs with similarity >= 0.85.

The implementation scans the upper triangle of the matrix and avoids building
all possible protein pairs as one large DataFrame.

python -m src.rank.rank_similarity \
  --output-dir outputs/network/rank \
  --top-k 100000 \
  --thresholds 0.85
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = (0.85,)
DEFAULT_SIMILARITY_DIRS = (
    Path("outputs/similarity_correlation"),
    Path("outputs/similarity_autoencoder"),
    Path("outputs/similarity_transformer"),
    Path("outputs/similarity"),
)


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not thresholds:
        raise argparse.ArgumentTypeError("At least one threshold is required.")
    return sorted(set(thresholds), reverse=True)


def load_protein_index(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "protein" in df.columns:
        proteins = df["protein"]
    elif len(df.columns) == 1:
        proteins = df.iloc[:, 0]
    else:
        raise ValueError(f"Expected one protein column in {path}, got {df.columns.tolist()}")
    return proteins.astype(str).tolist()


def resolve_protein_index(matrix_path: Path, protein_index: Path | None) -> Path:
    if protein_index is not None:
        return protein_index

    candidates = [
        matrix_path.parent / "protein_index.csv",
        Path("outputs/similarity/protein_index.csv"),
        Path("data/processed/protein_index.csv"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not infer protein_index.csv for {matrix_path}. "
        "Pass --protein-index explicitly."
    )


def method_name_from_path(path: Path) -> str:
    return path.stem


def format_threshold(threshold: float) -> str:
    text = f"{threshold:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p")


def strategy_name_for_threshold(threshold: float) -> str:
    return f"threshold_{format_threshold(threshold)}"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def top_k_edges(
    sim: np.ndarray,
    proteins: list[str],
    top_k: int,
    score_name: str,
) -> pd.DataFrame:
    """Return the top-K upper-triangle edges by score."""
    heap: list[tuple[float, int, int]] = []
    n = sim.shape[0]

    for i in range(n - 1):
        row_scores = sim[i, i + 1 :]
        valid = np.isfinite(row_scores)

        if len(heap) >= top_k:
            valid &= row_scores > heap[0][0]

        offsets = np.flatnonzero(valid)
        if offsets.size == 0:
            continue

        candidate_scores = row_scores[offsets]
        row_keep = min(offsets.size, top_k)
        if row_keep < offsets.size:
            keep = np.argpartition(candidate_scores, -row_keep)[-row_keep:]
            offsets = offsets[keep]
            candidate_scores = candidate_scores[keep]

        for offset, score in zip(offsets, candidate_scores, strict=False):
            item = (float(score), i, i + int(offset) + 1)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    ranked = sorted(heap, reverse=True)
    edge_df = pd.DataFrame(
        {
            "rank": np.arange(1, len(ranked) + 1),
            "protein_1": [proteins[i] for _, i, _ in ranked],
            "protein_2": [proteins[j] for _, _, j in ranked],
            score_name: [score for score, _, _ in ranked],
        }
    )
    return edge_df


def write_threshold_edges(
    sim: np.ndarray,
    proteins: list[str],
    threshold: float,
    score_name: str,
    output_path: Path,
    chunk_size: int,
) -> dict[str, float | int | str]:
    """Write all upper-triangle edges whose score is >= threshold."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = sim.shape[0]
    n_edges = 0
    min_score = np.inf
    max_score = -np.inf
    header = True
    buffer: list[dict[str, float | str]] = []

    def flush() -> None:
        nonlocal header, buffer
        if not buffer:
            return
        pd.DataFrame(buffer).to_csv(
            output_path,
            mode="w" if header else "a",
            header=header,
            index=False,
        )
        header = False
        buffer = []

    for i in range(n - 1):
        row_scores = sim[i, i + 1 :]
        mask = np.isfinite(row_scores) & (row_scores >= threshold)
        offsets = np.flatnonzero(mask)
        if offsets.size == 0:
            continue

        scores = row_scores[offsets]
        n_edges += int(offsets.size)
        min_score = min(min_score, float(scores.min()))
        max_score = max(max_score, float(scores.max()))

        for offset, score in zip(offsets, scores, strict=False):
            buffer.append(
                {
                    "protein_1": proteins[i],
                    "protein_2": proteins[i + int(offset) + 1],
                    score_name: float(score),
                    "threshold": threshold,
                }
            )
            if len(buffer) >= chunk_size:
                flush()

    flush()

    if n_edges == 0:
        pd.DataFrame(columns=["protein_1", "protein_2", score_name, "threshold"]).to_csv(
            output_path,
            index=False,
        )

    return {
        "threshold": threshold,
        "n_edges": n_edges,
        "min_score": np.nan if n_edges == 0 else min_score,
        "max_score": np.nan if n_edges == 0 else max_score,
        "output_path": str(output_path),
        "edges_path": str(output_path),
    }


def rank_one_matrix(
    matrix_path: Path,
    protein_index_path: Path,
    output_dir: Path,
    top_k: int,
    thresholds: list[float],
    chunk_size: int,
    score_name: str | None = None,
) -> list[dict[str, float | int | str]]:
    method = method_name_from_path(matrix_path)
    score_name = score_name or method
    output_dir.mkdir(parents=True, exist_ok=True)

    proteins = load_protein_index(protein_index_path)
    sim = np.load(matrix_path, mmap_mode="r")

    if sim.shape[0] != sim.shape[1]:
        raise ValueError(f"{matrix_path} is not square: {sim.shape}")
    if sim.shape[0] != len(proteins):
        raise ValueError(
            f"{matrix_path} has shape {sim.shape}, but {protein_index_path} "
            f"contains {len(proteins)} proteins."
        )

    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, float | int | str]] = []

    print(f"  Top {top_k:,}: {matrix_path}")
    top_df = top_k_edges(sim=sim, proteins=proteins, top_k=top_k, score_name=score_name)
    top_dir = method_dir / f"top{top_k}"
    top_path = top_dir / "edges.csv"
    top_dir.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(top_path, index=False)

    kth_score = float(top_df[score_name].iloc[-1]) if len(top_df) else np.nan
    top_summary = {
        "method": method,
        "strategy": f"top{top_k}",
        "matrix_path": str(matrix_path),
        "protein_index_path": str(protein_index_path),
        "top_k": top_k,
        "n_top_edges": len(top_df),
        "top_k_score_cutoff": kth_score,
        "top_output_path": str(top_path),
        "edges_path": str(top_path),
        "score_column": score_name,
    }
    write_json(top_dir / "summary.json", top_summary)
    output_rows.append(top_summary)

    threshold_rows = []
    for threshold in thresholds:
        strategy = strategy_name_for_threshold(threshold)
        threshold_dir = method_dir / strategy
        threshold_path = threshold_dir / "edges.csv"
        print(f"  Threshold >= {threshold:g}: {threshold_path}")
        row = write_threshold_edges(
            sim=sim,
            proteins=proteins,
            threshold=threshold,
            score_name=score_name,
            output_path=threshold_path,
            chunk_size=chunk_size,
        )
        row.update({
            "method": method,
            "strategy": strategy,
            "matrix_path": str(matrix_path),
            "protein_index_path": str(protein_index_path),
            "score_column": score_name,
        })
        write_json(threshold_dir / "summary.json", row)
        threshold_rows.append(row)
        output_rows.append(row)

    pd.DataFrame(output_rows).to_csv(method_dir / "rank_summary.csv", index=False)
    return output_rows


def discover_matrices(similarity_dirs: list[Path]) -> list[Path]:
    skip = {"overlap.npy"}
    paths = []
    seen = set()

    for similarity_dir in similarity_dirs:
        if not similarity_dir.exists():
            continue
        for path in sorted(similarity_dir.glob("*.npy")):
            if path.name in skip:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)

    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create top-K and thresholded edge tables from similarity matrices."
    )
    parser.add_argument(
        "--matrices",
        nargs="*",
        type=Path,
        default=None,
        help="Specific .npy similarity matrices. Defaults to all .npy files in --similarity-dir except overlap.npy.",
    )
    parser.add_argument(
        "--similarity-dir",
        type=Path,
        default=None,
        help="Single directory to scan for matrices when --matrices is omitted.",
    )
    parser.add_argument(
        "--similarity-dirs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_SIMILARITY_DIRS),
        help=(
            "Directories to scan when --matrices is omitted. Defaults cover "
            "correlation, autoencoder, transformer, and legacy similarity outputs."
        ),
    )
    parser.add_argument(
        "--protein-index",
        type=Path,
        default=None,
        help="Protein index CSV. Defaults to matrix directory protein_index.csv, then outputs/similarity/protein_index.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/network/rank"),
        help="Directory for ranked edge outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100_000,
        help="Number of top-ranked edges to write.",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=list(DEFAULT_THRESHOLDS),
        help=(
            "Comma-separated thresholds for edge output. Default: 0.85."
        ),
    )
    parser.add_argument(
        "--score-name",
        type=str,
        default=None,
        help="Score column name. Defaults to each matrix filename stem.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows buffered before appending threshold edge CSVs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    similarity_dirs = [args.similarity_dir] if args.similarity_dir else args.similarity_dirs
    matrix_paths = args.matrices or discover_matrices(similarity_dirs)
    if not matrix_paths:
        raise FileNotFoundError("No .npy similarity matrices found.")

    manifest_rows = []
    for matrix_path in matrix_paths:
        if not matrix_path.exists():
            raise FileNotFoundError(matrix_path)

        protein_index_path = resolve_protein_index(matrix_path, args.protein_index)
        rows = rank_one_matrix(
            matrix_path=matrix_path,
            protein_index_path=protein_index_path,
            output_dir=args.output_dir,
            top_k=args.top_k,
            thresholds=args.thresholds,
            chunk_size=args.chunk_size,
            score_name=args.score_name,
        )
        manifest_rows.extend(rows)

    pd.DataFrame(manifest_rows).to_csv(args.output_dir / "rank_manifest.csv", index=False)


if __name__ == "__main__":
    main()
