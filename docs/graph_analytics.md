# Graph Analytics Quickstart

This guide shows how to explore the Formula 1 datasets as a graph using the
new `analytics/graph_analysis.py` utility. The goal is to highlight influential
drivers, constructors, and circuits based on their historical relationships.

## 1. Install dependencies (if running locally)
If you are using the Docker images, the requirements will be installed during
build. For a host Python environment run:

```powershell
pip install -r requirements.txt
```

## 2. Run the script
Execute from the project root:

```powershell
python analytics/graph_analysis.py --base-path data --top-k 10
```

You can filter by season range or export CSV results:

```powershell
python analytics/graph_analysis.py --min-year 2015 --max-year 2020 --export-dir artifacts/graph_outputs
```

The script prints the top drivers, constructors, and circuits ranked by weighted
degree and betweenness centrality. Exported CSVs contain the same tables for
further analysis.

### Optional: push results to Neo4j
Provide your Neo4j connection details to load the same graph into a live
database (requires the `py2neo` dependency already listed in `requirements.txt`):

```powershell
python analytics/graph_analysis.py `
  --neo4j-uri bolt://localhost:7687 `
  --neo4j-user neo4j `
  --neo4j-password s3cr3t `
  --neo4j-wipe
```

The `--neo4j-wipe` flag clears existing nodes and relationships first. Omit it
if you prefer to merge into an existing dataset.

## 3. Next steps
- Build additional edges (for example, driver-to-driver rivalry edges based on
  wheel-to-wheel battles or podium co-appearances).
- Load the exported CSVs into your BI or graph database tool for interactive
  exploration.
- Pair this analysis with the strategy models to uncover which circuits or
  teams yield the highest impact for pit-stop decisions.
