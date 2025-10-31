# Graph Analytics Sample View

This example uses the query below inside the Neo4j Browser to visualise a slice of the Formula 1 knowledge graph built by `analytics/graph_analysis.py`:

```cypher
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 50;
```

## What the view shows

- **Two node labels are present**: `Driver` (orange) and `Circuit` (purple). Every displayed relationship is of type `DRIVER_CIRCUIT`, meaning the driver has raced on that circuit at least once.
- **Driver nodes with high degree** (for example Oscar Larrauri, Philippe Streiff, Adrián Campos, Joachim Winkelhock) sit near the centre because they share edges to many circuits in the first 50 relationships returned. These are not necessarily the globally most connected drivers—only the most connected within this sample.
- **Circuit nodes** form the outer halo. Circuits with higher weighted degree have more incident driver edges, which indicates venues that appeared frequently in the filtered subset.
- **Edge labels** indicate the aggregated relationship type (`DRIVER_CIRCUIT`). When you switch to the table view you can see the `weight` property; a higher weight means the same driver–circuit pair appears in multiple race results.

## Interpreting it in context

1. Use the Browser’s legend (right-hand panel) to filter by label or relationship type. This helps isolate a single driver or circuit to inspect its local neighbourhood.
2. Toggle to the table view to see the quantitative values (weights, relation sets) corresponding to the visual edges.
3. Run additional queries to focus on subsets, e.g. limit to a specific driver or year:
   ```cypher
   MATCH (d:Driver {name: "Lewis Hamilton"})-[:DRIVER_CIRCUIT]->(c:Circuit)
   RETURN d, c
   ```
4. Use the export button in the Browser to download the rendered PNG if you want to embed the current snapshot into reports. Save the file in a project directory such as `docs/figures/`.

## Tips for richer analysis

- Create more selective queries (by season, constructor, or championship position) to reduce clutter and highlight patterns relevant to your investigation.
- Combine the graph view with the rankings emitted by the script to cross-check which entities have the highest centrality versus which only appear locally dense in the visualisation.
- Consider enabling Neo4j Bloom for narrative exploration with natural-language-style searches once the dataset is loaded.
