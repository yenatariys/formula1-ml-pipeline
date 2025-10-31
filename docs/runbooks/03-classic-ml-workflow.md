# 03 – Classic ML Workflow

Train the fast scikit-learn and XGBoost models that write predictions back to Postgres for dashboard exploration.

## 1. Ensure prerequisites
- Postgres and the ETL pipeline have run (see steps 01 and 02).
- Classic dependencies are available (handled automatically inside the `ml_train` container).

## 2. Run the comparison trainer (recommended)
```powershell
docker-compose run --rm `
  -e DB_HOST=f1_postgres `
  -e DB_NAME=f1_data `
  -e DB_USER=admin `
  -e DB_PASSWORD=admin123 `
  ml_train python pipelines/classic/train_comparison.py
```
Outputs:
- `f1_predictions_rf`, `f1_predictions_xgb`, and `f1_model_comparison` tables in Postgres.
- Console logs summarising accuracy, ROC-AUC, and feature importances.

## 3. Run the baseline RandomForest only (optional)
```powershell
python pipelines/classic/train.py
```
Execute this from an activated virtual environment if you prefer running on the host.

## 4. Verify results
```powershell
docker-compose exec postgres psql -U admin -d f1_data -c "SELECT * FROM f1_model_comparison;"
```
The table should list the accuracy and ROC-AUC for each model.

## 5. Update dashboards
Once tables exist you can start the classic dashboard with:
```powershell
docker-compose up -d dashboard_classic
```
Navigate to `http://localhost:8501` to explore predictions and evaluation metrics.
