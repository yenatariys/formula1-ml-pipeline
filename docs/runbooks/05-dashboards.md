# 05 – Dashboards & Monitoring

Two Streamlit applications surface insights from the classic pipeline (Postgres-backed) and the big data pipeline (file artefact-backed).

## Classic dashboard (Postgres focus)
```powershell
docker-compose up -d dashboard_classic
```
- URL: `http://localhost:8501`
- Expects `f1_predictions`, `f1_predictions_rf`, `f1_predictions_xgb`, and `f1_model_comparison` tables.
- Provides race analytics, head-to-head comparisons, and model evaluation charts.

## Big data dashboard (Artefact focus)
```powershell
docker-compose up -d dashboard_bigdata
```
- URL: `http://localhost:8502`
- Reads Spark feature store shards and TensorFlow/Spark evaluation JSON files from `artifacts/`.
- Visualises feature distributions, model metrics, and artefact freshness.

## Operational tips
- Restart dashboards after retraining to refresh cached data (`docker-compose restart dashboard_classic dashboard_bigdata`).
- Dashboards mount the repo via bind volume; artefact updates become visible without rebuilding images.
- To view logs: `docker-compose logs -f dashboard_classic` or `docker-compose logs -f dashboard_bigdata`.
