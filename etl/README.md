# ETL Module (Step 02)

This folder contains the Spark-assisted extract/transform/load routines that hydrate Postgres with race history data.

## Key scripts
- `etl_pipeline.py` – orchestrates extract → transform → load inside the ETL container.
- `extract_data.py` – reads raw CSVs from the `data/` directory.
- `transform_data.py` – uses Spark to clean and typecast records before loading.
- `load_data.py` – writes the final dataframe to Postgres.
- `export_joined_results.py` – optional helper that materialises `data/f1_results_joined.csv` for the Spark big data job.

## Execution order
1. Run `etl_pipeline.py` (automatically executed when the `etl` container starts).
2. Optionally call `export_joined_results.py` after the ETL run to produce the joined dataset consumed by Spark.

Refer to `docs/runbooks/02-data-ingest-etl.md` for detailed commands and troubleshooting steps.
