"""Utility to explore Formula 1 relational data as a heterogeneous graph.

This script builds a multi-type graph from drivers, constructors, and circuits
using the public CSV sources that ship with the project. It computes basic
centrality metrics to highlight influential drivers, teams, and venues.

Usage (from repo root):
    python analytics/graph_analysis.py --base-path data --top-k 10

Optional flags allow filtering by year range and exporting the ranked tables to
CSV for further analysis.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import networkx as nx
import pandas as pd

NODE_TYPES = {"driver", "constructor", "circuit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F1 graph analytics")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("data"),
        help="Root directory containing the CSV files (default: data)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Minimum race year to include",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Maximum race year to include",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top records to display per node type",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional directory to export ranked tables as CSV",
    )
    return parser.parse_args()


def load_tables(base_path: Path) -> Dict[str, pd.DataFrame]:
    def _read(name: str) -> pd.DataFrame:
        path = base_path / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required dataset: {path}")
        return pd.read_csv(path)

    tables = {
        "drivers": _read("drivers"),
        "constructors": _read("constructors"),
        "circuits": _read("circuits"),
        "races": _read("races"),
        "results": _read("results"),
    }
    return tables


def filter_results(tables: Dict[str, pd.DataFrame], min_year: int | None, max_year: int | None) -> pd.DataFrame:
    races = tables["races"][['raceId', 'year', 'circuitId', 'name']].rename(columns={'name': 'race_name'})
    merged = tables["results"].merge(races, on="raceId", how="inner")
    if min_year is not None:
        merged = merged[merged["year"] >= min_year]
    if max_year is not None:
        merged = merged[merged["year"] <= max_year]
    return merged


def build_graph(tables: Dict[str, pd.DataFrame], filtered_results: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    # Add nodes with labels for readability
    for _, row in tables["drivers"].iterrows():
        G.add_node(
            f"driver_{row.driverId}",
            type="driver",
            label=f"{row.forename} {row.surname}",
            nationality=row.nationality,
        )
    for _, row in tables["constructors"].iterrows():
        G.add_node(
            f"constructor_{row.constructorId}",
            type="constructor",
            label=row.name,
            nationality=row.nationality,
        )
    for _, row in tables["circuits"].iterrows():
        G.add_node(
            f"circuit_{row.circuitId}",
            type="circuit",
            label=row.name,
            country=row.country,
        )

    # Helper to add weighted edges
    def _add_edges(pairs: Iterable[Tuple[str, str]], attr_name: str) -> None:
        for source, target in pairs:
            if G.has_edge(source, target):
                G[source][target]["weight"] += 1
                G[source][target]["relations"].add(attr_name)
            else:
                G.add_edge(source, target, weight=1, relations={attr_name})

    # Driver <-> Constructor edges
    driver_constructor_pairs = (
        (f"driver_{row.driverId}", f"constructor_{row.constructorId}")
        for row in filtered_results.itertuples()
    )
    _add_edges(driver_constructor_pairs, "driver_constructor")

    # Driver <-> Circuit edges
    driver_circuit_pairs = (
        (f"driver_{row.driverId}", f"circuit_{row.circuitId}")
        for row in filtered_results.itertuples()
    )
    _add_edges(driver_circuit_pairs, "driver_circuit")

    # Constructor <-> Circuit edges
    constructor_circuit_pairs = (
        (f"constructor_{row.constructorId}", f"circuit_{row.circuitId}")
        for row in filtered_results.itertuples()
    )
    _add_edges(constructor_circuit_pairs, "constructor_circuit")

    return G


def _rank_nodes(G: nx.Graph, node_type: str, top_k: int) -> pd.DataFrame:
    if node_type not in NODE_TYPES:
        raise ValueError(f"Unsupported node type: {node_type}")
    nodes = [n for n, data in G.nodes(data=True) if data.get("type") == node_type]
    if not nodes:
        return pd.DataFrame()

    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="weight")

    rows = []
    for node in nodes:
        data = G.nodes[node]
        rows.append(
            {
                "node": node,
                "label": data.get("label"),
                "degree": G.degree(node),
                "weighted_degree": G.degree(node, weight="weight"),
                "degree_centrality": degree_cent.get(node, 0.0),
                "betweenness": betweenness.get(node, 0.0),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["weighted_degree", "degree_centrality"], ascending=False)
    return df.head(top_k)


def export_tables(export_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        out_path = export_dir / f"graph_ranking_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    tables = load_tables(args.base_path)
    filtered_results = filter_results(tables, args.min_year, args.max_year)

    if filtered_results.empty:
        year_filter = f" between {args.min_year} and {args.max_year}" if args.min_year or args.max_year else ""
        raise ValueError(f"No race results found{year_filter}. Adjust filters and retry.")

    graph = build_graph(tables, filtered_results)
    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    rankings = {
        "drivers": _rank_nodes(graph, "driver", args.top_k),
        "constructors": _rank_nodes(graph, "constructor", args.top_k),
        "circuits": _rank_nodes(graph, "circuit", args.top_k),
    }

    for label, df in rankings.items():
        if df.empty:
            print(f"No nodes of type {label} found for the current filter")
            continue
        print("\n=== Top", args.top_k, label, "===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.export_dir is not None:
        export_tables(args.export_dir, rankings)


if __name__ == "__main__":
    main()
