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

### Optional: automate via ETL
The ETL container can kick off the graph job automatically once data loads
complete. Set the following environment variables on the `etl` service (for
example, inside `docker-compose.yml` or your deployment platform):

```yaml
environment:
  RUN_GRAPH_ANALYTICS: "true"
  GRAPH_BASE_PATH: data  # optional override
  GRAPH_EXPORT_DIR: artifacts/graph_outputs  # optional
  GRAPH_TOP_K: "10"
  GRAPH_MIN_YEAR: "2015"  # optional year filter
  GRAPH_MAX_YEAR: "2020"  # optional year filter
  GRAPH_NEO4J_URI: bolt://neo4j:7687  # optional Neo4j export
  GRAPH_NEO4J_USER: neo4j
  GRAPH_NEO4J_PASSWORD: s3cr3t
  GRAPH_NEO4J_WIPE: "true"
```

Leave the Neo4j variables unset if you only need local CSV outputs. Failures in
this optional step are logged but will not stop the ETL pipeline.

### Optional: push results to Neo4j
Spin up the bundled Neo4j container (data persists in `neo4j/` within the repo):

```powershell
docker-compose up -d neo4j
```

Set a stronger password by exporting `NEO4J_AUTH=neo4j/<your-secret>` in a
`.env` file before starting the container if desired.

With the service running, provide your Neo4j connection details to load the
same graph into the live database (requires the `py2neo` dependency already
listed in `requirements.txt`):

```powershell
python analytics/graph_analysis.py `
  --neo4j-uri bolt://localhost:7687 `
  --neo4j-user neo4j `
  --neo4j-password neo4j123 `
  --neo4j-wipe
```

The `--neo4j-wipe` flag clears existing nodes and relationships first. Omit it
if you prefer to merge into an existing dataset. Access the Neo4j Browser at
`http://localhost:7474` (or replace `localhost` with your LAN IP when sharing
on the network).

## 3. Next steps
- Build additional edges (for example, driver-to-driver rivalry edges based on
  wheel-to-wheel battles or podium co-appearances).
- Load the exported CSVs into your BI or graph database tool for interactive
  exploration.
- Pair this analysis with the strategy models to uncover which circuits or
  teams yield the highest impact for pit-stop decisions.
