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

### Explore in Neo4j Browser

Once the graph is loaded, try the Cypher snippets below. Run them in the Neo4j
Browser and switch to the Graph or Table panes depending on the insight you
need.

1. **Driver career breadth** – list every circuit a driver has raced on.
  ```cypher
  MATCH (d:Driver {name: "Lewis Hamilton"})-[:DRIVER_CIRCUIT]-(c:Circuit)
  RETURN d, c;
  ```
  Switch the Neo4j Browser to Graph view to visualise the driver node linked to each circuit. If you only need a tabular list, append `RETURN d.name AS driver, collect(DISTINCT c.name) AS circuits ORDER BY size(circuits) DESC` instead. See `docs/graph_sample_insights.md` for a captured example and deeper commentary.

2. **Circuit popularity leaderboard** – rank tracks by unique participating drivers and total starts.
  ```cypher
  MATCH (d:Driver)-[r:DRIVER_CIRCUIT]-(c:Circuit)
  RETURN c.name AS circuit,
       count(DISTINCT d) AS driverCount,
       sum(r.weight) AS starts
  ORDER BY driverCount DESC
  LIMIT 10;
  ```
  Highlights venues that appear most across the dataset and the breadth of
  driver attendance at each.

3. **Driver–constructor partnerships** – show which constructor a driver raced for at a specific circuit.
  ```cypher
  MATCH (d:Driver)-[:DRIVER_CONSTRUCTOR]-(con:Constructor),
    (d)-[:DRIVER_CIRCUIT]-(c:Circuit)
  WHERE c.name = "Silverstone Circuit"
  RETURN d.name AS driver, con.name AS constructor
  ORDER BY driver;
  ```
  Useful for spotting historical team allegiances at individual venues.

4. **Shared circuit experience** – find other drivers who have raced the same tracks as your target driver.
  ```cypher
  MATCH (d1:Driver {name: "Fernando Alonso"})-[:DRIVER_CIRCUIT]-(c:Circuit)
  MATCH (d2:Driver)-[:DRIVER_CIRCUIT]-(c)
  WHERE d1 <> d2
  RETURN d2.name AS peer, collect(DISTINCT c.name) AS sharedCircuits,
       size(sharedCircuits) AS overlap
  ORDER BY overlap DESC
  LIMIT 10;
  ```
  Identifies comparable drivers based on overlapping race venues.

5. **Time-filtered circuit activity** – constrain relationships to recent seasons.
  ```cypher
  MATCH (d:Driver)-[r:DRIVER_CIRCUIT {season: 2020}]-(c:Circuit)
  RETURN c.name AS circuit,
       count(DISTINCT d) AS driverCount,
       sum(r.weight) AS raceStarts2020
  ORDER BY raceStarts2020 DESC
  LIMIT 5;
  ```
  Requires the relationship to retain a `season` property; adjust the year or
  extend the WHERE clause for ranges.

## 3. Next steps
- Build additional edges (for example, driver-to-driver rivalry edges based on
  wheel-to-wheel battles or podium co-appearances).
- Load the exported CSVs into your BI or graph database tool for interactive
  exploration.
- Pair this analysis with the strategy models to uncover which circuits or
  teams yield the highest impact for pit-stop decisions.
