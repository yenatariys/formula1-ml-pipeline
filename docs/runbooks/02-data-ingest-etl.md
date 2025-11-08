# 02 – Data Ingest & ETL

Run the ETL service to transform raw CSVs into the `f1_results_transformed` table that powers downstream pipelines.

## 1. Stage raw datasets
Ensure all CSV sources exist under `data/` (e.g., `data/results.csv`, `data/races.csv`). Replace or append new seasons as needed before each ETL run.

## 2. Start the ETL container
```powershell
docker-compose up etl
```
The container waits for Postgres to become healthy, executes `etl/etl_pipeline.py`, then terminates. When the run completes you should see `ETL pipeline finished successfully!`.

## 3. Confirm table creation
```powershell
docker-compose exec postgres psql -U admin -d f1_data -c "\d f1_results_transformed"
```
You should see all expected columns (position, points, driverId, etc.). If the table is missing, check ETL logs with `docker-compose logs etl`.

## 4. Export joined dataset for Spark (optional)
Some big data workflows expect `data/f1_results_joined.csv`. Generate it after the ETL run:
```powershell
docker-compose run --rm etl python etl/export_joined_results.py
```
This command reuses the ETL image and writes the combined dataset into `data/` for Spark feature engineering.
