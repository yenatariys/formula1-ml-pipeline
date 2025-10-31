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
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple, Any, Optional, List

import networkx as nx
import pandas as pd

NODE_TYPES = {"driver", "constructor", "circuit"}


def _chunked(items: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """Yield successive lists of length *size* from *items*."""
    batch: List[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


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
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=None,
        help="Bolt URI for Neo4j (e.g. bolt://localhost:7687). Enables Neo4j export.",
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default=None,
        help="Neo4j username (required when --neo4j-uri is provided)",
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=None,
        help="Neo4j password (required when --neo4j-uri is provided)",
    )
    parser.add_argument(
        "--neo4j-wipe",
        action="store_true",
        help="If set, clears existing nodes and relationships before loading.",
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


def export_to_neo4j(nx_graph: nx.Graph, uri: str, user: str, password: str, wipe: bool) -> None:
    try:
        from py2neo import Graph as NeoGraph
    except ImportError as exc:
        raise RuntimeError(
            "py2neo is required for Neo4j export. Install it via 'pip install py2neo'."
        ) from exc

    neo = NeoGraph(uri, auth=(user, password))
    if wipe:
        neo.run("MATCH (n) DETACH DELETE n")

    batch_size = 500

    def _sanitize_label(raw: str) -> str:
        cleaned = "".join(ch for ch in raw if ch.isalnum() or ch == "_")
        return cleaned or "Entity"

    nodes_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node_id, data in nx_graph.nodes(data=True):
        label = _sanitize_label(data.get("type", "Entity").capitalize())
        props = {"id": node_id, "name": data.get("label")}
        for key, value in data.items():
            if key in {"type", "label"}:
                continue
            props[key] = value
        nodes_by_label[label].append({"id": node_id, "props": props})

    for label, rows in nodes_by_label.items():
        if not rows:
            continue
        query = (
            f"UNWIND $rows AS row "
            f"MERGE (n:{label} {{id: row.id}}) "
            "SET n += row.props"
        )
        for batch in _chunked(rows, batch_size):
            neo.run(query, rows=batch)

    rel_batches: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for source, target, attr in nx_graph.edges(data=True):
        relations = attr.get("relations", {"RELATED"})
        weight = attr.get("weight", 1)
        relation_types = sorted(relations)
        ordered_source, ordered_target = sorted([source, target])
        for rel in relation_types:
            rel_type = _sanitize_label(rel.upper())
            rel_batches[rel_type].append(
                {
                    "source_id": ordered_source,
                    "target_id": ordered_target,
                    "weight": weight,
                    "relation_types": relation_types,
                }
            )

    for rel_type, rows in rel_batches.items():
        if not rows:
            continue
        query = (
            "UNWIND $rows AS row "
            "MATCH (a {id: row.source_id}), (b {id: row.target_id}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r.weight = row.weight, r.relation_types = row.relation_types"
        )
        for batch in _chunked(rows, batch_size):
            neo.run(query, rows=batch)
    print("Neo4j export completed.")


def run_graph_analysis(
    base_path: Path | str = Path("data"),
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    top_k: int = 10,
    export_dir: Optional[Path | str] = None,
    neo4j_config: Optional[Dict[str, Any]] = None,
    echo: bool = True,
) -> Dict[str, pd.DataFrame]:
    base_path = Path(base_path)
    export_path = Path(export_dir) if export_dir is not None else None

    tables = load_tables(base_path)
    filtered_results = filter_results(tables, min_year, max_year)

    if filtered_results.empty:
        year_filter = ""
        if min_year or max_year:
            year_filter = f" between {min_year} and {max_year}"
        raise ValueError(f"No race results found{year_filter}. Adjust filters and retry.")

    graph = build_graph(tables, filtered_results)
    if echo:
        print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    rankings = {
        "drivers": _rank_nodes(graph, "driver", top_k),
        "constructors": _rank_nodes(graph, "constructor", top_k),
        "circuits": _rank_nodes(graph, "circuit", top_k),
    }

    if echo:
        for label, df in rankings.items():
            if df.empty:
                print(f"No nodes of type {label} found for the current filter")
                continue
            print("\n=== Top", top_k, label, "===")
            print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if export_path is not None:
        export_tables(export_path, rankings)

    if neo4j_config is not None:
        required_keys = {"uri", "user", "password", "wipe"}
        missing = required_keys - set(neo4j_config)
        if missing:
            raise ValueError(f"Neo4j config missing keys: {', '.join(sorted(missing))}")
        export_to_neo4j(
            graph,
            uri=neo4j_config["uri"],
            user=neo4j_config["user"],
            password=neo4j_config["password"],
            wipe=bool(neo4j_config.get("wipe", False)),
        )

    return rankings


def main() -> None:
    args = parse_args()
    neo4j_config = None
    if args.neo4j_uri is not None:
        if not args.neo4j_user or not args.neo4j_password:
            raise ValueError("--neo4j-user and --neo4j-password are required when --neo4j-uri is set")
        neo4j_config = {
            "uri": args.neo4j_uri,
            "user": args.neo4j_user,
            "password": args.neo4j_password,
            "wipe": args.neo4j_wipe,
        }

    run_graph_analysis(
        base_path=args.base_path,
        min_year=args.min_year,
        max_year=args.max_year,
        top_k=args.top_k,
        export_dir=args.export_dir,
        neo4j_config=neo4j_config,
        echo=True,
    )


if __name__ == "__main__":
    main()
