"""Create an interactive Plotly HTML network from a ranked edge table."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.graph.build_graph import build_graph, graph_for_plot, load_edges


def make_plotly_figure(
    graph: nx.Graph,
    title: str,
    seed: int,
    max_nodes: int,
    max_edges: int,
):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly is not installed. Install it with `pip install plotly`, "
            "or install this project's requirements."
        ) from exc

    plot_graph = graph_for_plot(graph, max_nodes=max_nodes, max_edges=max_edges)
    pos = nx.spring_layout(plot_graph, seed=seed, weight="weight", iterations=120)

    edge_x = []
    edge_y = []
    for u, v in plot_graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 0.5, "color": "rgba(90, 90, 90, 0.35)"},
        hoverinfo="none",
        name="edges",
    )

    degree = dict(plot_graph.degree())
    weighted_degree = dict(plot_graph.degree(weight="weight"))
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    hover_text = []

    max_degree = max(degree.values()) if degree else 1
    for node in plot_graph.nodes():
        x, y = pos[node]
        deg = degree[node]
        wdeg = weighted_degree[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(deg)
        node_size.append(7 + 20 * deg / max(max_degree, 1))
        hover_text.append(
            f"protein: {node}<br>"
            f"degree: {deg}<br>"
            f"weighted degree: {wdeg:.3f}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker={
            "size": node_size,
            "color": node_color,
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": "Degree"},
            "line": {"width": 0.5, "color": "white"},
        },
        text=hover_text,
        hoverinfo="text",
        name="proteins",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title={
                "text": (
                    f"{title}<br>"
                    f"<sup>{plot_graph.number_of_nodes():,} nodes, "
                    f"{plot_graph.number_of_edges():,} edges shown</sup>"
                ),
                "x": 0.02,
            },
            showlegend=False,
            hovermode="closest",
            margin={"b": 20, "l": 5, "r": 5, "t": 60},
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            plot_bgcolor="white",
        ),
    )

    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an interactive Plotly HTML network from edge CSV."
    )
    parser.add_argument("--edges", type=Path, required=True, help="Ranked edge CSV.")
    parser.add_argument("--output-html", type=Path, required=True, help="Output HTML path.")
    parser.add_argument("--score-col", type=str, default=None, help="Score column.")
    parser.add_argument("--source-col", type=str, default="protein_1")
    parser.add_argument("--target-col", type=str, default="protein_2")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=5000)
    parser.add_argument("--max-nodes", type=int, default=800)
    parser.add_argument("--max-edges", type=int, default=5000)
    parser.add_argument("--layout-seed", type=int, default=42)
    parser.add_argument("--title", type=str, default="Protein similarity network")
    parser.add_argument(
        "--include-plotlyjs",
        type=str,
        default="cdn",
        choices=["cdn", "directory", "inline"],
        help=(
            "How to include Plotly JS in the HTML. Use inline for a fully "
            "self-contained file; cdn keeps the file smaller."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_html.parent.mkdir(parents=True, exist_ok=True)

    edge_df, score_col = load_edges(
        edge_path=args.edges,
        score_col=args.score_col,
        source_col=args.source_col,
        target_col=args.target_col,
        min_score=args.min_score,
        top_n=args.top_n,
    )
    graph = build_graph(
        edge_df=edge_df,
        score_col=score_col,
        source_col=args.source_col,
        target_col=args.target_col,
    )

    fig = make_plotly_figure(
        graph=graph,
        title=args.title,
        seed=args.layout_seed,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
    )

    include_plotlyjs: str | bool
    include_plotlyjs = True if args.include_plotlyjs == "inline" else args.include_plotlyjs
    fig.write_html(args.output_html, include_plotlyjs=include_plotlyjs)

    print(f"Saved interactive Plotly graph: {args.output_html}")


if __name__ == "__main__":
    main()

