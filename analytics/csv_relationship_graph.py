"""
CSV Relationship Graph Analysis for F1 ETL Pipeline

This script analyzes the relationships between CSV files in the data directory
by examining their columns and foreign key relationships. It creates a visual
graph showing how the tables are connected and provides insights into the
data model structure.

Usage:
    python analytics/csv_relationship_graph.py
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json

# Define the data directory
DATA_DIR = Path("data")
OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Known relationships based on F1 database schema
RELATIONSHIPS = {
    "results.csv": {
        "raceId": "races.csv",
        "driverId": "drivers.csv",
        "constructorId": "constructors.csv",
        "statusId": "status.csv"
    },
    "races.csv": {
        "circuitId": "circuits.csv",
        "year": "seasons.csv"
    },
    "qualifying.csv": {
        "raceId": "races.csv",
        "driverId": "drivers.csv",
        "constructorId": "constructors.csv"
    },
    "constructor_results.csv": {
        "raceId": "races.csv",
        "constructorId": "constructors.csv"
    },
    "constructor_standings.csv": {
        "raceId": "races.csv",
        "constructorId": "constructors.csv"
    },
    "driver_standings.csv": {
        "raceId": "races.csv",
        "driverId": "drivers.csv"
    },
    "lap_times.csv": {
        "raceId": "races.csv",
        "driverId": "drivers.csv"
    },
    "pit_stops.csv": {
        "raceId": "races.csv",
        "driverId": "drivers.csv"
    },
    "sprint_results.csv": {
        "raceId": "races.csv",
        "driverId": "drivers.csv",
        "constructorId": "constructors.csv"
    }
}

# Define table categories
TABLE_CATEGORIES = {
    "Core Entities": ["drivers.csv", "constructors.csv", "circuits.csv", "races.csv", "seasons.csv"],
    "Race Results": ["results.csv", "sprint_results.csv", "qualifying.csv"],
    "Standings": ["driver_standings.csv", "constructor_standings.csv"],
    "Race Details": ["lap_times.csv", "pit_stops.csv"],
    "Reference": ["status.csv", "constructor_results.csv"],
    "Transformed": ["f1_results_joined.csv"]
}


def load_csv_metadata() -> Dict[str, Dict]:
    """Load metadata about each CSV file."""
    metadata = {}
    
    for csv_file in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, nrows=5)  # Just read first 5 rows for metadata
            metadata[csv_file.name] = {
                "columns": list(df.columns),
                "row_count": len(pd.read_csv(csv_file)),
                "file_size_kb": csv_file.stat().st_size / 1024
            }
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
    
    return metadata


def build_relationship_graph() -> nx.DiGraph:
    """Build a directed graph of CSV relationships."""
    G = nx.DiGraph()
    
    # Add all CSV files as nodes with their category
    for category, tables in TABLE_CATEGORIES.items():
        for table in tables:
            if (DATA_DIR / table).exists():
                G.add_node(table, category=category)
    
    # Add edges based on foreign key relationships
    for source_table, foreign_keys in RELATIONSHIPS.items():
        for fk_column, target_table in foreign_keys.items():
            if (DATA_DIR / source_table).exists() and (DATA_DIR / target_table).exists():
                G.add_edge(target_table, source_table, 
                          relationship=fk_column,
                          edge_type="foreign_key")
    
    return G


def analyze_table_dependencies(G: nx.DiGraph) -> Dict:
    """Analyze dependencies between tables."""
    analysis = {
        "most_referenced": [],
        "most_dependent": [],
        "core_tables": [],
        "leaf_tables": []
    }
    
    # Tables that are most referenced (have most incoming edges)
    in_degree = dict(G.in_degree())
    analysis["most_referenced"] = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Tables that depend on most others (have most outgoing edges)
    out_degree = dict(G.out_degree())
    analysis["most_dependent"] = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Core tables (high in-degree, low out-degree)
    analysis["core_tables"] = [(node, in_degree[node]) 
                               for node in G.nodes() 
                               if in_degree[node] >= 2 and out_degree[node] <= 1]
    
    # Leaf tables (no incoming edges or high out-degree)
    analysis["leaf_tables"] = [(node, out_degree[node]) 
                               for node in G.nodes() 
                               if in_degree[node] == 0 or out_degree[node] >= 3]
    
    return analysis


def visualize_graph(G: nx.DiGraph, output_file: Path):
    """Create a visual representation of the CSV relationships."""
    plt.figure(figsize=(16, 12))
    
    # Define colors for each category
    category_colors = {
        "Core Entities": "#3498db",      # Blue
        "Race Results": "#e74c3c",       # Red
        "Standings": "#f39c12",          # Orange
        "Race Details": "#9b59b6",       # Purple
        "Reference": "#95a5a6",          # Gray
        "Transformed": "#2ecc71"         # Green
    }
    
    # Assign colors to nodes based on category
    node_colors = []
    for node in G.nodes():
        category = G.nodes[node].get('category', 'Reference')
        node_colors.append(category_colors.get(category, '#95a5a6'))
    
    # Use hierarchical layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=3000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, 
                          arrowstyle='->', width=1.5,
                          connectionstyle='arc3,rad=0.1')
    
    # Add edge labels (foreign key columns)
    edge_labels = nx.get_edge_attributes(G, 'relationship')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, label=cat)
                      for cat, color in category_colors.items()]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.title("Formula 1 ETL CSV Relationship Graph", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Graph visualization saved to: {output_file}")
    plt.close()


def generate_dependency_report(G: nx.DiGraph, metadata: Dict, analysis: Dict):
    """Generate a detailed text report of the relationships."""
    report_file = OUTPUT_DIR / "csv_dependency_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FORMULA 1 ETL CSV RELATIONSHIP ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overview
        f.write("OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total CSV files: {len(G.nodes())}\n")
        f.write(f"Total relationships: {len(G.edges())}\n")
        f.write(f"Total data size: {sum(m['file_size_kb'] for m in metadata.values()):.2f} KB\n")
        f.write(f"Total records: {sum(m['row_count'] for m in metadata.values()):,}\n\n")
        
        # Most referenced tables
        f.write("MOST REFERENCED TABLES (Core Tables)\n")
        f.write("-" * 80 + "\n")
        for table, count in analysis["most_referenced"]:
            f.write(f"  {table:30} - Referenced by {count} other tables\n")
            if table in metadata:
                f.write(f"    Rows: {metadata[table]['row_count']:,}, ")
                f.write(f"Size: {metadata[table]['file_size_kb']:.2f} KB\n")
        f.write("\n")
        
        # Most dependent tables
        f.write("MOST DEPENDENT TABLES (Complex Joins)\n")
        f.write("-" * 80 + "\n")
        for table, count in analysis["most_dependent"]:
            f.write(f"  {table:30} - Depends on {count} other tables\n")
            # Show what it depends on
            predecessors = list(G.predecessors(table))
            if predecessors:
                f.write(f"    Dependencies: {', '.join(predecessors)}\n")
        f.write("\n")
        
        # Table details by category
        f.write("TABLES BY CATEGORY\n")
        f.write("-" * 80 + "\n")
        for category, tables in TABLE_CATEGORIES.items():
            f.write(f"\n{category}:\n")
            for table in tables:
                if table in metadata:
                    f.write(f"  {table:30}")
                    f.write(f" | Rows: {metadata[table]['row_count']:>8,}")
                    f.write(f" | Columns: {len(metadata[table]['columns']):>3}")
                    f.write(f" | Size: {metadata[table]['file_size_kb']:>8.2f} KB\n")
        f.write("\n")
        
        # Column details for key tables
        f.write("KEY TABLE COLUMN DETAILS\n")
        f.write("-" * 80 + "\n")
        key_tables = ["results.csv", "races.csv", "drivers.csv", "constructors.csv"]
        for table in key_tables:
            if table in metadata:
                f.write(f"\n{table}:\n")
                f.write(f"  Columns ({len(metadata[table]['columns'])}): ")
                f.write(", ".join(metadata[table]['columns']))
                f.write("\n")
    
    print(f"✅ Dependency report saved to: {report_file}")


def generate_json_schema(G: nx.DiGraph, metadata: Dict):
    """Generate a JSON schema representation of the data model."""
    schema = {
        "tables": {},
        "relationships": []
    }
    
    # Add table information
    for node in G.nodes():
        if node in metadata:
            schema["tables"][node] = {
                "columns": metadata[node]["columns"],
                "row_count": metadata[node]["row_count"],
                "file_size_kb": round(metadata[node]["file_size_kb"], 2),
                "category": G.nodes[node].get('category', 'Unknown')
            }
    
    # Add relationship information
    for source, target, data in G.edges(data=True):
        schema["relationships"].append({
            "from": source,
            "to": target,
            "foreign_key": data.get("relationship", ""),
            "type": data.get("edge_type", "")
        })
    
    schema_file = OUTPUT_DIR / "csv_schema.json"
    with open(schema_file, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"✅ JSON schema saved to: {schema_file}")


def find_data_flow_paths(G: nx.DiGraph):
    """Find all possible data flow paths from core tables to results."""
    print("\n" + "=" * 80)
    print("DATA FLOW PATHS")
    print("=" * 80)
    
    # Find paths from core tables to results
    core_tables = ["drivers.csv", "constructors.csv", "circuits.csv"]
    target_tables = ["results.csv", "f1_results_joined.csv"]
    
    for source in core_tables:
        for target in target_tables:
            if source in G.nodes() and target in G.nodes():
                try:
                    paths = list(nx.all_simple_paths(G, source, target, cutoff=4))
                    if paths:
                        print(f"\n{source} → {target}:")
                        for path in paths[:3]:  # Show first 3 paths
                            print(f"  {' → '.join(path)}")
                except nx.NetworkXNoPath:
                    pass


def main():
    """Main execution function."""
    print("🏎️  Formula 1 ETL CSV Relationship Analysis")
    print("=" * 80)
    
    # Load metadata
    print("\n📊 Loading CSV metadata...")
    metadata = load_csv_metadata()
    print(f"   Found {len(metadata)} CSV files")
    
    # Build relationship graph
    print("\n🔗 Building relationship graph...")
    G = build_relationship_graph()
    print(f"   Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Analyze dependencies
    print("\n🔍 Analyzing table dependencies...")
    analysis = analyze_table_dependencies(G)
    
    # Generate visualizations and reports
    print("\n📈 Generating outputs...")
    visualize_graph(G, OUTPUT_DIR / "csv_relationship_graph.png")
    generate_dependency_report(G, metadata, analysis)
    generate_json_schema(G, metadata)
    
    # Find data flow paths
    find_data_flow_paths(G)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✅ Analysis complete!")
    print(f"\nMost Referenced Tables (Core):")
    for table, count in analysis["most_referenced"][:3]:
        print(f"  • {table:30} ({count} references)")
    
    print(f"\nMost Complex Tables (Most Dependencies):")
    for table, count in analysis["most_dependent"][:3]:
        print(f"  • {table:30} ({count} dependencies)")
    
    print(f"\n📁 Output files saved to: {OUTPUT_DIR}/")
    print(f"   - csv_relationship_graph.png")
    print(f"   - csv_dependency_report.txt")
    print(f"   - csv_schema.json")


if __name__ == "__main__":
    main()
