"""Build and visualize protein similarity graphs from ranked edge tables.

python -m src.graph.build_graph \
  --rank-dir outputs/network/rank \
  --graph-root outputs/network/graph

python -m src.graph.build_graph \
  --edges outputs/network/rank/autoencoder_cosine/threshold_0p85/edges.csv \
  --output-dir outputs/network/graph/autoencoder_cosine/threshold_0p85 \
  --score-col autoencoder_cosine

python -m src.graph.build_graph \
  --build-top-per-node \
  --per-node-k 10 \
  --graph-root outputs/network/graph
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.rank.rank_similarity import (
    DEFAULT_SIMILARITY_DIRS,
    discover_matrices,
    load_protein_index,
    method_name_from_path,
    resolve_protein_index,
)


RESERVED_EDGE_COLUMNS = {
    "protein_1",
    "protein_2",
    "rank",
    "threshold",
    "shared_patients",
    "label",
    "CORUM_positive",
    "selected_by_count",
    "per_node_k",
}


def infer_score_column(df: pd.DataFrame, score_col: str | None) -> str:
    if score_col is not None:
        if score_col not in df.columns:
            raise ValueError(f"Score column {score_col!r} not found in edge table.")
        return score_col

    candidates = [
        col
        for col in df.columns
        if col not in RESERVED_EDGE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not candidates:
        raise ValueError(
            "Could not infer a numeric score column. Pass --score-col explicitly."
        )
    return candidates[0]


def load_edges(
    edge_path: Path,
    score_col: str | None,
    source_col: str,
    target_col: str,
    min_score: float | None,
    top_n: int | None,
) -> tuple[pd.DataFrame, str]:
    edge_df = pd.read_csv(edge_path)
    required = {source_col, target_col}
    missing = required - set(edge_df.columns)
    if missing:
        raise ValueError(f"{edge_path} missing required columns: {sorted(missing)}")

    score_col = infer_score_column(edge_df, score_col)
    edge_df = edge_df[[source_col, target_col, score_col] + [
        c for c in edge_df.columns if c not in {source_col, target_col, score_col}
    ]].copy()
    edge_df[score_col] = pd.to_numeric(edge_df[score_col], errors="coerce")
    edge_df = edge_df.dropna(subset=[source_col, target_col, score_col])

    if min_score is not None:
        edge_df = edge_df[edge_df[score_col] >= min_score].copy()

    edge_df = edge_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    if top_n is not None:
        edge_df = edge_df.head(top_n).copy()

    return edge_df, score_col


def build_graph(
    edge_df: pd.DataFrame,
    score_col: str,
    source_col: str,
    target_col: str,
) -> nx.Graph:
    graph = nx.Graph()

    for row in edge_df.itertuples(index=False):
        row_dict = row._asdict()
        u = str(row_dict[source_col])
        v = str(row_dict[target_col])
        score = float(row_dict[score_col])

        if u == v:
            continue

        graph.add_edge(u, v, weight=score, score=score)

    return graph


def graph_summary(graph: nx.Graph, edge_df: pd.DataFrame, score_col: str) -> dict:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest_component_size = len(components[0]) if components else 0
    degrees = dict(graph.degree())
    weighted_degrees = dict(graph.degree(weight="weight"))

    density = nx.density(graph) if n_nodes > 1 else 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "n_connected_components": len(components),
        "largest_component_size": largest_component_size,
        "largest_component_fraction": (
            largest_component_size / n_nodes if n_nodes else np.nan
        ),
        "mean_degree": float(np.mean(list(degrees.values()))) if degrees else 0.0,
        "median_degree": float(np.median(list(degrees.values()))) if degrees else 0.0,
        "max_degree": int(max(degrees.values())) if degrees else 0,
        "mean_weighted_degree": (
            float(np.mean(list(weighted_degrees.values()))) if weighted_degrees else 0.0
        ),
        "score_min": float(edge_df[score_col].min()) if len(edge_df) else np.nan,
        "score_median": float(edge_df[score_col].median()) if len(edge_df) else np.nan,
        "score_max": float(edge_df[score_col].max()) if len(edge_df) else np.nan,
    }


def node_metrics(graph: nx.Graph) -> pd.DataFrame:
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    component_id = {}

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    for idx, component in enumerate(components, start=1):
        for node in component:
            component_id[node] = idx

    return (
        pd.DataFrame(
            {
                "protein": list(graph.nodes()),
                "degree": [degree[n] for n in graph.nodes()],
                "weighted_degree": [weighted_degree[n] for n in graph.nodes()],
                "component_id": [component_id[n] for n in graph.nodes()],
            }
        )
        .sort_values(["degree", "weighted_degree"], ascending=False)
        .reset_index(drop=True)
    )


def graph_for_plot(
    graph: nx.Graph,
    max_nodes: int,
    max_edges: int,
) -> nx.Graph:
    if graph.number_of_nodes() <= max_nodes and graph.number_of_edges() <= max_edges:
        return graph.copy()

    largest = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest).copy()

    if subgraph.number_of_nodes() > max_nodes:
        ranked_nodes = sorted(
            subgraph.degree(weight="weight"),
            key=lambda x: x[1],
            reverse=True,
        )
        keep_nodes = [node for node, _ in ranked_nodes[:max_nodes]]
        subgraph = subgraph.subgraph(keep_nodes).copy()

    if subgraph.number_of_edges() > max_edges:
        ranked_edges = sorted(
            subgraph.edges(data=True),
            key=lambda x: x[2].get("weight", 0.0),
            reverse=True,
        )
        keep_edges = ranked_edges[:max_edges]
        plot_graph = nx.Graph()
        plot_graph.add_nodes_from(subgraph.nodes(data=True))
        for u, v, data in keep_edges:
            plot_graph.add_edge(u, v, **data)
        subgraph = plot_graph

    isolates = list(nx.isolates(subgraph))
    subgraph.remove_nodes_from(isolates)
    return subgraph


def draw_graph_preview(
    graph: nx.Graph,
    output_path: Path,
    seed: int,
    max_nodes: int,
    max_edges: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib.pyplot as plt

    plot_graph = graph_for_plot(graph, max_nodes=max_nodes, max_edges=max_edges)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if plot_graph.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No edges to plot", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    pos = nx.spring_layout(plot_graph, seed=seed, weight="weight", iterations=100)
    degrees = np.asarray([plot_graph.degree(n) for n in plot_graph.nodes()], dtype=float)
    node_sizes = 20 + 120 * (degrees / degrees.max()) if degrees.max() > 0 else 40
    weights = np.asarray(
        [data.get("weight", 1.0) for _, _, data in plot_graph.edges(data=True)],
        dtype=float,
    )
    if len(weights):
        edge_widths = 0.2 + 1.8 * (
            (weights - weights.min()) / max(weights.max() - weights.min(), 1e-8)
        )
    else:
        edge_widths = 0.5

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(
        plot_graph,
        pos,
        ax=ax,
        alpha=0.18,
        width=edge_widths,
        edge_color="#4d4d4d",
    )
    nx.draw_networkx_nodes(
        plot_graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=degrees,
        cmap="viridis",
        linewidths=0.2,
        edgecolors="white",
    )

    top_label_nodes = [
        node
        for node, _ in sorted(
            plot_graph.degree(weight="weight"),
            key=lambda x: x[1],
            reverse=True,
        )[:25]
    ]
    labels = {node: node for node in top_label_nodes}
    nx.draw_networkx_labels(plot_graph, pos, labels=labels, ax=ax, font_size=7)

    ax.set_title(
        f"Graph preview: {plot_graph.number_of_nodes():,} nodes, "
        f"{plot_graph.number_of_edges():,} edges"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_graph_outputs(
    edge_path: Path,
    output_dir: Path,
    score_col: str | None,
    source_col: str,
    target_col: str,
    min_score: float | None,
    top_n: int | None,
    layout_seed: int,
    max_plot_nodes: int,
    max_plot_edges: int,
    write_graphml: bool,
    write_gexf: bool,
    write_preview: bool,
    copy_edges: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_df, resolved_score_col = load_edges(
        edge_path=edge_path,
        score_col=score_col,
        source_col=source_col,
        target_col=target_col,
        min_score=min_score,
        top_n=top_n,
    )

    graph = build_graph(
        edge_df=edge_df,
        score_col=resolved_score_col,
        source_col=source_col,
        target_col=target_col,
    )

    if copy_edges:
        edge_df.to_csv(output_dir / "edges.csv", index=False)

    summary = graph_summary(graph=graph, edge_df=edge_df, score_col=resolved_score_col)
    summary.update(
        {
            "source_edge_path": str(edge_path),
            "score_column": resolved_score_col,
            "output_dir": str(output_dir),
            "copied_edges": copy_edges,
            "edge_storage": "source_edge_path" if not copy_edges else "edges.csv",
            "min_score_filter": min_score,
            "top_n_limit": top_n,
        }
    )
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    node_metrics(graph).to_csv(output_dir / "nodes.csv", index=False)

    if write_graphml:
        nx.write_graphml(graph, output_dir / "graph.graphml")
    if write_gexf:
        nx.write_gexf(graph, output_dir / "graph.gexf")
    if write_preview:
        draw_graph_preview(
            graph=graph,
            output_path=output_dir / "preview.png",
            seed=layout_seed,
            max_nodes=max_plot_nodes,
            max_edges=max_plot_edges,
        )

    return summary


def top_per_node_edges(
    sim: np.ndarray,
    proteins: list[str],
    per_node_k: int,
    score_col: str,
) -> pd.DataFrame:
    """Build an undirected union of each protein's top-k finite neighbors."""
    if per_node_k <= 0:
        raise ValueError("--per-node-k must be positive.")

    n = sim.shape[0]
    edges: dict[tuple[int, int], dict[str, object]] = {}

    for i in range(n):
        row_scores = np.asarray(sim[i])
        valid = np.isfinite(row_scores)
        valid[i] = False
        neighbor_idx = np.flatnonzero(valid)
        if neighbor_idx.size == 0:
            continue

        neighbor_scores = row_scores[neighbor_idx]
        keep_n = min(per_node_k, neighbor_idx.size)
        if keep_n < neighbor_idx.size:
            keep_pos = np.argpartition(neighbor_scores, -keep_n)[-keep_n:]
        else:
            keep_pos = np.arange(neighbor_idx.size)

        selected_idx = neighbor_idx[keep_pos]
        selected_scores = neighbor_scores[keep_pos]
        order = np.argsort(selected_scores)[::-1]

        for j, score in zip(selected_idx[order], selected_scores[order], strict=False):
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            edge = edges.setdefault(
                (a, b),
                {
                    "score": float(score),
                    "selected_by": set(),
                },
            )
            edge["score"] = max(float(edge["score"]), float(score))
            selected_by = edge["selected_by"]
            if isinstance(selected_by, set):
                selected_by.add(proteins[i])

    rows = []
    ranked_edges = sorted(edges.items(), key=lambda item: item[1]["score"], reverse=True)
    for rank, ((i, j), data) in enumerate(ranked_edges, start=1):
        selected_by = data["selected_by"]
        if not isinstance(selected_by, set):
            selected_by = set()
        rows.append(
            {
                "rank": rank,
                "protein_1": proteins[i],
                "protein_2": proteins[j],
                score_col: float(data["score"]),
                "selected_by_count": len(selected_by),
                "selected_by": ";".join(sorted(selected_by)),
                "per_node_k": per_node_k,
            }
        )

    return pd.DataFrame(rows)


def write_graph_from_edges(
    edge_df: pd.DataFrame,
    output_dir: Path,
    score_col: str,
    source_col: str,
    target_col: str,
    layout_seed: int,
    max_plot_nodes: int,
    max_plot_edges: int,
    write_graphml: bool,
    write_gexf: bool,
    write_preview: bool,
    summary_extra: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_graph(
        edge_df=edge_df,
        score_col=score_col,
        source_col=source_col,
        target_col=target_col,
    )

    edge_df.to_csv(output_dir / "edges.csv", index=False)
    node_metrics(graph).to_csv(output_dir / "nodes.csv", index=False)

    summary = graph_summary(graph=graph, edge_df=edge_df, score_col=score_col)
    summary.update(
        {
            "score_column": score_col,
            "output_dir": str(output_dir),
            "edge_storage": "edges.csv",
        }
    )
    if summary_extra:
        summary.update(summary_extra)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if write_graphml:
        nx.write_graphml(graph, output_dir / "graph.graphml")
    if write_gexf:
        nx.write_gexf(graph, output_dir / "graph.gexf")
    if write_preview:
        draw_graph_preview(
            graph=graph,
            output_path=output_dir / "preview.png",
            seed=layout_seed,
            max_nodes=max_plot_nodes,
            max_edges=max_plot_edges,
        )

    return summary


def write_graph_manifest(graph_root: Path, rows: list[dict]) -> Path:
    graph_root.mkdir(parents=True, exist_ok=True)
    manifest_path = graph_root / "graph_manifest.csv"
    new_df = pd.DataFrame(rows)

    if manifest_path.exists():
        old_df = pd.read_csv(manifest_path)
        combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df

    if {"method", "strategy"}.issubset(combined.columns):
        combined = combined.drop_duplicates(["method", "strategy"], keep="last")
        combined = combined.sort_values(["method", "strategy"]).reset_index(drop=True)

    combined.to_csv(manifest_path, index=False)
    return manifest_path


def build_top_per_node_graphs(
    matrix_paths: list[Path],
    protein_index_override: Path | None,
    graph_root: Path,
    per_node_k: int,
    layout_seed: int,
    max_plot_nodes: int,
    max_plot_edges: int,
    write_graphml: bool,
    write_gexf: bool,
    write_preview: bool,
) -> list[dict]:
    summaries = []

    for matrix_path in matrix_paths:
        if not matrix_path.exists():
            raise FileNotFoundError(matrix_path)

        method = method_name_from_path(matrix_path)
        score_col = method
        protein_index_path = resolve_protein_index(matrix_path, protein_index_override)
        proteins = load_protein_index(protein_index_path)
        sim = np.load(matrix_path, mmap_mode="r")

        if sim.shape[0] != sim.shape[1]:
            raise ValueError(f"{matrix_path} is not square: {sim.shape}")
        if sim.shape[0] != len(proteins):
            raise ValueError(
                f"{matrix_path} has shape {sim.shape}, but {protein_index_path} "
                f"contains {len(proteins)} proteins."
            )

        strategy = f"top{per_node_k}perNodes"
        output_dir = graph_root / method / strategy
        print(f"Build graph: {method}/{strategy}")

        edge_df = top_per_node_edges(
            sim=sim,
            proteins=proteins,
            per_node_k=per_node_k,
            score_col=score_col,
        )
        summary = write_graph_from_edges(
            edge_df=edge_df,
            output_dir=output_dir,
            score_col=score_col,
            source_col="protein_1",
            target_col="protein_2",
            layout_seed=layout_seed,
            max_plot_nodes=max_plot_nodes,
            max_plot_edges=max_plot_edges,
            write_graphml=write_graphml,
            write_gexf=write_gexf,
            write_preview=write_preview,
            summary_extra={
                "method": method,
                "strategy": strategy,
                "matrix_path": str(matrix_path),
                "protein_index_path": str(protein_index_path),
                "per_node_k": per_node_k,
                "edge_definition": (
                    "Undirected union of each protein's top-k finite similarity neighbors."
                ),
            },
        )
        summaries.append(summary)

    return summaries


def discover_rank_edge_files(rank_dir: Path) -> list[tuple[str, str, Path]]:
    edge_files = []
    if not rank_dir.exists():
        raise FileNotFoundError(rank_dir)

    for edge_path in sorted(rank_dir.glob("*/*/edges.csv")):
        strategy = edge_path.parent.name
        method = edge_path.parent.parent.name
        edge_files.append((method, strategy, edge_path))

    return edge_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build NetworkX graph outputs from ranked/thresholded edge CSVs."
    )
    parser.add_argument("--edges", type=Path, default=None, help="Single ranked edge CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for --edges mode.")
    parser.add_argument(
        "--rank-dir",
        type=Path,
        default=None,
        help="Organized rank directory created by src.rank.rank_similarity.",
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("outputs/network/graph"),
        help="Root graph output directory for --rank-dir mode.",
    )
    parser.add_argument("--score-col", type=str, default=None, help="Score column.")
    parser.add_argument("--source-col", type=str, default="protein_1")
    parser.add_argument("--target-col", type=str, default="protein_2")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--layout-seed", type=int, default=42)
    parser.add_argument("--max-plot-nodes", type=int, default=500)
    parser.add_argument("--max-plot-edges", type=int, default=2000)
    parser.add_argument(
        "--copy-edges",
        action="store_true",
        help=(
            "Copy ranked/thresholded source edges into the graph output directory. "
            "Default keeps graph storage lean and records source_edge_path instead."
        ),
    )
    parser.add_argument(
        "--write-graphml",
        action="store_true",
        help="Optionally write graph.graphml. Disabled by default to reduce storage.",
    )
    parser.add_argument(
        "--write-gexf",
        action="store_true",
        help="Optionally write graph.gexf. Disabled by default to reduce storage.",
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--build-top-per-node",
        action="store_true",
        help="Build a graph where each protein keeps its top-k similarity neighbors.",
    )
    parser.add_argument(
        "--per-node-k",
        type=int,
        default=10,
        help="Number of strongest neighbors kept per protein for --build-top-per-node.",
    )
    parser.add_argument(
        "--matrices",
        nargs="*",
        type=Path,
        default=None,
        help="Specific .npy similarity matrices for --build-top-per-node.",
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
        help="Directories to scan for matrices for --build-top-per-node.",
    )
    parser.add_argument(
        "--protein-index",
        type=Path,
        default=None,
        help="Protein index CSV override for --build-top-per-node.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_rows = []

    if args.build_top_per_node:
        similarity_dirs = [args.similarity_dir] if args.similarity_dir else args.similarity_dirs
        matrix_paths = args.matrices or discover_matrices(similarity_dirs)
        if not matrix_paths:
            raise FileNotFoundError("No .npy similarity matrices found.")

        summaries = build_top_per_node_graphs(
            matrix_paths=matrix_paths,
            protein_index_override=args.protein_index,
            graph_root=args.graph_root,
            per_node_k=args.per_node_k,
            layout_seed=args.layout_seed,
            max_plot_nodes=args.max_plot_nodes,
            max_plot_edges=args.max_plot_edges,
            write_graphml=args.write_graphml,
            write_gexf=args.write_gexf,
            write_preview=not args.no_preview,
        )
        manifest_rows.extend(summaries)

    if args.rank_dir is not None:
        edge_files = discover_rank_edge_files(args.rank_dir)
        if not edge_files:
            raise FileNotFoundError(f"No organized edge files found in {args.rank_dir}")

        for method, strategy, edge_path in edge_files:
            output_dir = args.graph_root / method / strategy
            print(f"Build graph: {method}/{strategy}")
            summary = build_graph_outputs(
                edge_path=edge_path,
                output_dir=output_dir,
                score_col=args.score_col,
                source_col=args.source_col,
                target_col=args.target_col,
                min_score=args.min_score,
                top_n=args.top_n,
                layout_seed=args.layout_seed,
                max_plot_nodes=args.max_plot_nodes,
                max_plot_edges=args.max_plot_edges,
                write_graphml=args.write_graphml,
                write_gexf=args.write_gexf,
                write_preview=not args.no_preview,
                copy_edges=args.copy_edges,
            )
            summary.update({"method": method, "strategy": strategy})
            manifest_rows.append(summary)

        manifest_path = write_graph_manifest(args.graph_root, manifest_rows)
        print(f"Saved graph manifest: {manifest_path}")
        return

    if args.build_top_per_node:
        manifest_path = write_graph_manifest(args.graph_root, manifest_rows)
        print(f"Saved graph manifest: {manifest_path}")
        return

    if args.edges is None or args.output_dir is None:
        raise ValueError(
            "Use --build-top-per-node, --rank-dir, or both --edges and --output-dir."
        )

    summary = build_graph_outputs(
        edge_path=args.edges,
        output_dir=args.output_dir,
        score_col=args.score_col,
        source_col=args.source_col,
        target_col=args.target_col,
        min_score=args.min_score,
        top_n=args.top_n,
        layout_seed=args.layout_seed,
        max_plot_nodes=args.max_plot_nodes,
        max_plot_edges=args.max_plot_edges,
        write_graphml=args.write_graphml,
        write_gexf=args.write_gexf,
        write_preview=not args.no_preview,
        copy_edges=args.copy_edges,
    )

    print("Saved graph outputs:")
    if args.copy_edges:
        print(f"  {args.output_dir / 'edges.csv'}")
    else:
        print(f"  source edges: {args.edges}")
    print(f"  {args.output_dir / 'nodes.csv'}")
    print(f"  {args.output_dir / 'summary.json'}")
    if not args.no_preview:
        print(f"  {args.output_dir / 'preview.png'}")


if __name__ == "__main__":
    main()
