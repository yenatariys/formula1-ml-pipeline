# Classic Machine Learning Pipelines

This directory contains the lightweight, scikit-learn based trainers that operate on the curated race history stored in PostgreSQL. They are designed for fast iteration on a single machine or within the `ml_train` Docker service.

## Contents
- `train.py` – trains a class-balanced RandomForest model and writes predictions to the `f1_predictions` table.
- `train_comparison.py` – runs the RandomForest baseline and an XGBoost model with a compact grid-search, then stores comparison tables (`f1_predictions_rf`, `f1_predictions_xgb`, `f1_model_comparison`).

## Prerequisites
1. Docker Compose stack running Postgres and the ETL job:
   - `docker-compose up -d postgres`
   - `docker-compose up etl`
2. PostgreSQL connection variables exported in your shell (PowerShell example):
   ```powershell
   $env:DB_HOST = "localhost"
   $env:DB_NAME = "f1_data"
   $env:DB_USER = "admin"
   $env:DB_PASSWORD = "admin123"
   ```
3. A Python environment with the classic dependencies. When executing outside Docker, install from the repo root:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install pandas scikit-learn psycopg2-binary sqlalchemy xgboost
   ```

## Running the trainers
### Option A: inside the `ml_train` container (recommended)
```powershell
docker-compose run --rm -e DB_HOST=f1_postgres -e DB_NAME=f1_data -e DB_USER=admin -e DB_PASSWORD=admin123 ml_train python pipelines/classic/train_comparison.py
```
Replace the script name with `pipelines/classic/train.py` if you only need the RandomForest baseline. Results are written back to Postgres and logged to stdout.

### Option B: directly on your host machine
```powershell
python pipelines/classic/train_comparison.py
```
Ensure the database environment variables are set beforehand. Predictions and comparison metrics will be saved to the tables listed above.

## Output locations
- Model metrics appear in the terminal log.
- Prediction tables and comparison summaries are written to PostgreSQL, enabling dashboards or notebooks to consume the results immediately.

## Troubleshooting
- If you see auth errors, verify the environment variables and that Postgres is reachable.
- Missing tables usually indicates the ETL pipeline has not populated `f1_results_transformed` yet.
- XGBoost can take several minutes; tune the grid in `train_comparison.py` if you need faster feedback.
