"""Detect protein network modules with Leiden and dense cores with MCODE.

The script consumes graph folders produced by ``src.graph.build_graph`` and
keeps outputs organized by method and graph strategy.

python -m src.modules.detect_modules \
  --graph-root outputs/network/graph \
  --output-root outputs/network/modules

python -m src.modules.detect_modules \
  --methods transformer_cosine autoencoder_cosine \
  --strategies top10perNodes threshold_0p85
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

from src.graph.build_graph import build_graph, load_edges


DEFAULT_GRAPH_ROOT = Path("outputs/network/graph")
DEFAULT_OUTPUT_ROOT = Path("outputs/network/modules")


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def finite_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def discover_graph_runs(
    graph_root: Path,
    methods: set[str] | None,
    strategies: set[str] | None,
) -> list[tuple[str, str, Path]]:
    manifest_path = graph_root / "graph_manifest.csv"
    runs: list[tuple[str, str, Path]] = []

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        for row in manifest.itertuples(index=False):
            method = str(getattr(row, "method"))
            strategy = str(getattr(row, "strategy"))
            if methods is not None and method not in methods:
                continue
            if strategies is not None and strategy not in strategies:
                continue
            graph_dir = graph_root / method / strategy
            if (graph_dir / "summary.json").exists():
                runs.append((method, strategy, graph_dir))
        return sorted(set(runs))

    for summary_path in sorted(graph_root.glob("*/*/summary.json")):
        strategy = summary_path.parent.name
        method = summary_path.parent.parent.name
        if methods is not None and method not in methods:
            continue
        if strategies is not None and strategy not in strategies:
            continue
        runs.append((method, strategy, summary_path.parent))

    return runs


def load_graph_run_edges(
    graph_dir: Path,
    score_col: str | None,
    source_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, str, dict]:
    summary_path = graph_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = read_json(summary_path)
    local_edges = graph_dir / "edges.csv"
    if local_edges.exists():
        edge_path = local_edges
        min_score = None
        top_n = None
    else:
        edge_path = Path(summary["source_edge_path"])
        min_score = summary.get("min_score_filter")
        top_n = finite_int(summary.get("top_n_limit"))

    edge_df, resolved_score_col = load_edges(
        edge_path=edge_path,
        score_col=score_col or summary.get("score_column"),
        source_col=source_col,
        target_col=target_col,
        min_score=min_score,
        top_n=top_n,
    )
    return edge_df, resolved_score_col, summary


def networkx_to_igraph(graph: nx.Graph) -> tuple[ig.Graph, list[str]]:
    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
    weights = [float(data.get("weight", 1.0)) for _, _, data in graph.edges(data=True)]

    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_graph.vs["name"] = nodes
    if weights:
        ig_graph.es["weight"] = weights
    return ig_graph, nodes


def run_leiden(
    graph: nx.Graph,
    objective: str,
    resolution: float,
    beta: float,
    n_iterations: int,
) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["protein", "module_id"])

    ig_graph, nodes = networkx_to_igraph(graph)
    weights = "weight" if ig_graph.ecount() else None

    partition = ig_graph.community_leiden(
        objective_function=objective,
        weights=weights,
        resolution=resolution,
        beta=beta,
        n_iterations=n_iterations,
    )

    raw_membership = partition.membership
    module_sizes = pd.Series(raw_membership).value_counts().sort_values(ascending=False)
    raw_to_stable = {
        raw_id: f"L{rank:04d}" for rank, raw_id in enumerate(module_sizes.index, start=1)
    }

    return pd.DataFrame(
        {
            "protein": nodes,
            "module_id": [raw_to_stable[module] for module in raw_membership],
        }
    )


def top_key_proteins(subgraph: nx.Graph, n: int) -> str:
    if subgraph.number_of_nodes() == 0:
        return ""

    weighted_degree = dict(subgraph.degree(weight="weight"))
    degree = dict(subgraph.degree())
    ranked = sorted(
        subgraph.nodes(),
        key=lambda node: (weighted_degree.get(node, 0.0), degree.get(node, 0), str(node)),
        reverse=True,
    )
    return ";".join(ranked[:n])


def summarize_modules(
    graph: nx.Graph,
    assignments: pd.DataFrame,
    method: str,
    strategy: str,
    score_col: str,
    key_protein_count: int,
    min_module_size: int,
) -> pd.DataFrame:
    rows = []
    for module_id, group in assignments.groupby("module_id"):
        proteins = group["protein"].astype(str).tolist()
        size = len(proteins)
        if size < min_module_size:
            continue

        subgraph = graph.subgraph(proteins).copy()
        weights = [
            float(data.get("weight", np.nan))
            for _, _, data in subgraph.edges(data=True)
        ]
        rows.append(
            {
                "method": method,
                "strategy": strategy,
                "module_id": module_id,
                "size": size,
                "n_edges": subgraph.number_of_edges(),
                "density": nx.density(subgraph) if size > 1 else 0.0,
                "mean_weight": float(np.nanmean(weights)) if weights else np.nan,
                "median_weight": float(np.nanmedian(weights)) if weights else np.nan,
                "max_weight": float(np.nanmax(weights)) if weights else np.nan,
                "total_weight": float(np.nansum(weights)) if weights else 0.0,
                "mean_degree": (
                    float(np.mean([d for _, d in subgraph.degree()])) if size else 0.0
                ),
                "max_degree": (
                    int(max([d for _, d in subgraph.degree()])) if size else 0
                ),
                "key_proteins": top_key_proteins(subgraph, key_protein_count),
                "score_column": score_col,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["size", "total_weight", "density"], ascending=False)
        .reset_index(drop=True)
    )


def annotate_assignments(
    graph: nx.Graph,
    assignments: pd.DataFrame,
    module_summary: pd.DataFrame,
    method: str,
    strategy: str,
) -> pd.DataFrame:
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    module_sizes = assignments["module_id"].value_counts().to_dict()
    module_rank = {
        module_id: rank
        for rank, module_id in enumerate(module_summary["module_id"].tolist(), start=1)
    }

    out = assignments.copy()
    out.insert(0, "method", method)
    out.insert(1, "strategy", strategy)
    out["module_size"] = out["module_id"].map(module_sizes).astype(int)
    out["module_rank_by_size"] = out["module_id"].map(module_rank)
    out["degree"] = out["protein"].map(degree).fillna(0).astype(int)
    out["weighted_degree"] = out["protein"].map(weighted_degree).fillna(0.0)
    return out.sort_values(
        ["module_size", "module_id", "weighted_degree"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def size_distribution(assignments: pd.DataFrame, min_module_size: int) -> dict:
    sizes = assignments["module_id"].value_counts().to_numpy()
    if sizes.size == 0:
        return {
            "n_modules": 0,
            "n_modules_ge_min": 0,
            "n_singletons": 0,
            "module_size_min": np.nan,
            "module_size_q25": np.nan,
            "module_size_median": np.nan,
            "module_size_mean": np.nan,
            "module_size_q75": np.nan,
            "module_size_max": np.nan,
        }

    return {
        "n_modules": int(sizes.size),
        "n_modules_ge_min": int(np.sum(sizes >= min_module_size)),
        "n_singletons": int(np.sum(sizes == 1)),
        "module_size_min": int(np.min(sizes)),
        "module_size_q25": float(np.quantile(sizes, 0.25)),
        "module_size_median": float(np.median(sizes)),
        "module_size_mean": float(np.mean(sizes)),
        "module_size_q75": float(np.quantile(sizes, 0.75)),
        "module_size_max": int(np.max(sizes)),
    }


def mcode_vertex_weights(graph: nx.Graph) -> dict[str, float]:
    weights: dict[str, float] = {}

    for node in graph.nodes():
        neighborhood = set(graph.neighbors(node))
        neighborhood.add(node)
        subgraph = graph.subgraph(neighborhood)
        n_nodes = subgraph.number_of_nodes()
        if n_nodes < 2 or subgraph.number_of_edges() == 0:
            weights[node] = 0.0
            continue

        core_numbers = nx.core_number(subgraph)
        max_core = max(core_numbers.values()) if core_numbers else 0
        if max_core == 0:
            weights[node] = 0.0
            continue

        core_nodes = [n for n, core in core_numbers.items() if core >= max_core]
        core_graph = subgraph.subgraph(core_nodes)
        density = nx.density(core_graph) if core_graph.number_of_nodes() > 1 else 0.0
        weights[node] = float(max_core * density)

    return weights


def haircut_cluster(graph: nx.Graph, nodes: set[str]) -> set[str]:
    cluster = set(nodes)
    changed = True
    while changed:
        changed = False
        subgraph = graph.subgraph(cluster)
        remove = [node for node, degree in subgraph.degree() if degree <= 1]
        if remove:
            cluster.difference_update(remove)
            changed = True
    return cluster


def apply_k_core_filter(graph: nx.Graph, nodes: set[str], k_core: int) -> set[str]:
    if not nodes or k_core <= 1:
        return set(nodes)
    subgraph = graph.subgraph(nodes)
    if subgraph.number_of_nodes() <= k_core:
        return set()
    try:
        core_numbers = nx.core_number(subgraph)
    except nx.NetworkXError:
        return set()
    return {node for node, core in core_numbers.items() if core >= k_core}


def grow_mcode_cluster(
    graph: nx.Graph,
    seed: str,
    vertex_weights: dict[str, float],
    node_score_cutoff: float,
    expansion: str,
) -> set[str]:
    """Create a MCODE-style seed neighborhood.

    Cytoscape MCODE recursively expands from a seed. On connected protein kNN
    graphs, recursive expansion can balloon into broad low-density components,
    so the default keeps the seed's immediate high-scoring neighborhood and
    lets the downstream haircut/k-core/density filters define the dense core.
    """
    seed_weight = vertex_weights.get(seed, 0.0)
    min_weight = seed_weight * (1.0 - node_score_cutoff)
    cluster = {seed}

    if expansion == "local":
        for neighbor in graph.neighbors(seed):
            if vertex_weights.get(neighbor, 0.0) >= min_weight:
                cluster.add(neighbor)
    elif expansion == "recursive":
        queue: deque[str] = deque([seed])
        while queue:
            node = queue.popleft()
            for neighbor in graph.neighbors(node):
                if neighbor in cluster:
                    continue
                if vertex_weights.get(neighbor, 0.0) >= min_weight:
                    cluster.add(neighbor)
                    queue.append(neighbor)
    else:
        raise ValueError(f"Unknown MCODE expansion mode: {expansion}")

    return cluster


def run_mcode(
    graph: nx.Graph,
    method: str,
    strategy: str,
    node_score_cutoff: float,
    k_core: int,
    min_core_size: int,
    min_density: float,
    min_score: float,
    min_seed_score: float,
    expansion: str,
    key_protein_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if graph.number_of_nodes() == 0:
        empty_cores = pd.DataFrame()
        empty_members = pd.DataFrame()
        return empty_cores, empty_members

    vertex_weights = mcode_vertex_weights(graph)
    ranked_nodes = sorted(
        graph.nodes(),
        key=lambda node: (vertex_weights.get(node, 0.0), graph.degree(node), str(node)),
        reverse=True,
    )

    seen: set[str] = set()
    core_records = []

    for seed in ranked_nodes:
        seed_weight = vertex_weights.get(seed, 0.0)
        if seed_weight < min_seed_score:
            break
        if seed in seen or seed_weight <= 0:
            continue

        cluster = grow_mcode_cluster(
            graph=graph,
            seed=seed,
            vertex_weights=vertex_weights,
            node_score_cutoff=node_score_cutoff,
            expansion=expansion,
        )
        cluster = haircut_cluster(graph, cluster)
        cluster = apply_k_core_filter(graph, cluster, k_core=k_core)
        if len(cluster) < min_core_size:
            seen.add(seed)
            continue

        subgraph = graph.subgraph(cluster).copy()
        size = subgraph.number_of_nodes()
        density = nx.density(subgraph) if size > 1 else 0.0
        mcode_score = float(size * density)
        if density < min_density or mcode_score < min_score:
            seen.add(seed)
            continue

        seen.update(cluster)
        weights = [
            float(data.get("weight", np.nan))
            for _, _, data in subgraph.edges(data=True)
        ]

        core_records.append(
            {
                "method": method,
                "strategy": strategy,
                "seed_protein": seed,
                "seed_score": float(vertex_weights.get(seed, 0.0)),
                "size": size,
                "n_edges": subgraph.number_of_edges(),
                "density": density,
                "mcode_score": mcode_score,
                "mean_weight": float(np.nanmean(weights)) if weights else np.nan,
                "median_weight": float(np.nanmedian(weights)) if weights else np.nan,
                "max_weight": float(np.nanmax(weights)) if weights else np.nan,
                "key_proteins": top_key_proteins(subgraph, key_protein_count),
                "protein_preview": ";".join(sorted(cluster)[:key_protein_count]),
                "members": sorted(cluster),
            }
        )

    core_records = sorted(
        core_records,
        key=lambda row: (row["mcode_score"], row["size"], row["density"]),
        reverse=True,
    )

    core_rows = []
    member_rows = []
    for core_idx, record in enumerate(core_records, start=1):
        core_id = f"MC{core_idx:04d}"
        members = record.pop("members")
        core_rows.append({"core_id": core_id, **record})

        for protein in members:
            member_rows.append(
                {
                    "method": method,
                    "strategy": strategy,
                    "core_id": core_id,
                    "protein": protein,
                    "seed_protein": record["seed_protein"],
                    "vertex_weight": float(vertex_weights.get(protein, 0.0)),
                    "degree": int(graph.degree(protein)),
                    "weighted_degree": float(graph.degree(protein, weight="weight")),
                }
            )

    cores = pd.DataFrame(core_rows)
    members = pd.DataFrame(member_rows)
    return cores, members


def detect_one_graph(
    method: str,
    strategy: str,
    graph_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict:
    edge_df, score_col, graph_summary = load_graph_run_edges(
        graph_dir=graph_dir,
        score_col=args.score_col,
        source_col=args.source_col,
        target_col=args.target_col,
    )
    graph = build_graph(
        edge_df=edge_df,
        score_col=score_col,
        source_col=args.source_col,
        target_col=args.target_col,
    )

    assignments = run_leiden(
        graph=graph,
        objective=args.leiden_objective,
        resolution=args.leiden_resolution,
        beta=args.leiden_beta,
        n_iterations=args.leiden_iterations,
    )
    module_summary = summarize_modules(
        graph=graph,
        assignments=assignments,
        method=method,
        strategy=strategy,
        score_col=score_col,
        key_protein_count=args.key_proteins,
        min_module_size=args.min_module_size,
    )
    assignment_out = annotate_assignments(
        graph=graph,
        assignments=assignments,
        module_summary=module_summary,
        method=method,
        strategy=strategy,
    )

    mcode_cores, mcode_members = run_mcode(
        graph=graph,
        method=method,
        strategy=strategy,
        node_score_cutoff=args.mcode_node_score_cutoff,
        k_core=args.mcode_k_core,
        min_core_size=args.mcode_min_core_size,
        min_density=args.mcode_min_density,
        min_score=args.mcode_min_score,
        min_seed_score=args.mcode_min_seed_score,
        expansion=args.mcode_expansion,
        key_protein_count=args.key_proteins,
    )

    output_dir = output_root / method / strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_out.to_csv(output_dir / "module_assignments.csv", index=False)
    module_summary.to_csv(output_dir / "module_summary.csv", index=False)
    mcode_cores.to_csv(output_dir / "mcode_cores.csv", index=False)
    mcode_members.to_csv(output_dir / "mcode_core_members.csv", index=False)

    size_stats = size_distribution(assignments, min_module_size=args.min_module_size)
    top_module_key_proteins = (
        module_summary["key_proteins"].iloc[0] if len(module_summary) else ""
    )
    top_mcode_key_proteins = (
        mcode_cores["key_proteins"].iloc[0] if len(mcode_cores) else ""
    )

    run_summary = {
        "method": method,
        "strategy": strategy,
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "score_column": score_col,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "leiden_objective": args.leiden_objective,
        "leiden_resolution": args.leiden_resolution,
        "leiden_beta": args.leiden_beta,
        "leiden_iterations": args.leiden_iterations,
        "min_module_size": args.min_module_size,
        "mcode_expansion": args.mcode_expansion,
        "mcode_node_score_cutoff": args.mcode_node_score_cutoff,
        "mcode_k_core": args.mcode_k_core,
        "mcode_min_core_size": args.mcode_min_core_size,
        "mcode_min_density": args.mcode_min_density,
        "mcode_min_score": args.mcode_min_score,
        "mcode_min_seed_score": args.mcode_min_seed_score,
        "n_mcode_cores": int(len(mcode_cores)),
        "top_module_id": (
            module_summary["module_id"].iloc[0] if len(module_summary) else ""
        ),
        "top_module_size": (
            int(module_summary["size"].iloc[0]) if len(module_summary) else 0
        ),
        "top_module_key_proteins": top_module_key_proteins,
        "top_mcode_core_id": (
            mcode_cores["core_id"].iloc[0] if len(mcode_cores) else ""
        ),
        "top_mcode_size": int(mcode_cores["size"].iloc[0]) if len(mcode_cores) else 0,
        "top_mcode_key_proteins": top_mcode_key_proteins,
        "source_graph_summary": graph_summary,
    }
    run_summary.update(size_stats)
    write_json(output_dir / "run_summary.json", run_summary)
    return run_summary


def write_overviews(output_root: Path, run_rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    overview = pd.DataFrame(run_rows)
    if not overview.empty:
        drop_cols = [c for c in ["source_graph_summary"] if c in overview.columns]
        overview.drop(columns=drop_cols).to_csv(
            output_root / "module_detection_overview.csv",
            index=False,
        )

    module_frames = []
    mcode_frames = []
    for row in run_rows:
        output_dir = Path(row["output_dir"])
        module_path = output_dir / "module_summary.csv"
        mcode_path = output_dir / "mcode_cores.csv"
        if module_path.exists():
            module_df = pd.read_csv(module_path)
            if not module_df.empty:
                module_frames.append(module_df.head(10))
        if mcode_path.exists():
            mcode_df = pd.read_csv(mcode_path)
            if not mcode_df.empty:
                mcode_frames.append(mcode_df.head(10))

    if module_frames:
        pd.concat(module_frames, ignore_index=True).to_csv(
            output_root / "top_modules_overview.csv",
            index=False,
        )
    if mcode_frames:
        pd.concat(mcode_frames, ignore_index=True).to_csv(
            output_root / "top_mcode_cores_overview.csv",
            index=False,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Leiden module detection and MCODE dense-core detection."
    )
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--strategies", nargs="*", default=None)
    parser.add_argument("--score-col", type=str, default=None)
    parser.add_argument("--source-col", type=str, default="protein_1")
    parser.add_argument("--target-col", type=str, default="protein_2")
    parser.add_argument(
        "--leiden-objective",
        choices=["modularity", "CPM"],
        default="modularity",
        help="Use modularity for weighted graph modules, or CPM for resolution sweeps.",
    )
    parser.add_argument("--leiden-resolution", type=float, default=3.0)
    parser.add_argument("--leiden-beta", type=float, default=0.01)
    parser.add_argument(
        "--leiden-iterations",
        type=int,
        default=-1,
        help="Negative value runs Leiden until no further quality improvement.",
    )
    parser.add_argument("--min-module-size", type=int, default=3)
    parser.add_argument("--key-proteins", type=int, default=10)
    parser.add_argument("--mcode-node-score-cutoff", type=float, default=0.2)
    parser.add_argument(
        "--mcode-expansion",
        choices=["local", "recursive"],
        default="local",
        help=(
            "local keeps the seed's immediate high-scoring neighborhood; "
            "recursive follows Cytoscape-style expansion and may be slow on connected kNN graphs."
        ),
    )
    parser.add_argument("--mcode-k-core", type=int, default=2)
    parser.add_argument("--mcode-min-core-size", type=int, default=5)
    parser.add_argument(
        "--mcode-min-density",
        type=float,
        default=0.5,
        help="Discard MCODE clusters below this internal edge density.",
    )
    parser.add_argument(
        "--mcode-min-score",
        type=float,
        default=5.0,
        help="Discard MCODE clusters with size*density below this value.",
    )
    parser.add_argument(
        "--mcode-min-seed-score",
        type=float,
        default=1.0,
        help="Stop MCODE seed expansion once vertex weights fall below this score.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    methods = set(args.methods) if args.methods else None
    strategies = set(args.strategies) if args.strategies else None
    runs = discover_graph_runs(
        graph_root=args.graph_root,
        methods=methods,
        strategies=strategies,
    )
    if not runs:
        raise FileNotFoundError(f"No graph runs found under {args.graph_root}")

    run_rows = []
    for method, strategy, graph_dir in runs:
        print(f"Detect modules: {method}/{strategy}")
        run_rows.append(
            detect_one_graph(
                method=method,
                strategy=strategy,
                graph_dir=graph_dir,
                output_root=args.output_root,
                args=args,
            )
        )

    write_overviews(args.output_root, run_rows)
    print(f"Saved module detection outputs under {args.output_root}")


if __name__ == "__main__":
    main()
