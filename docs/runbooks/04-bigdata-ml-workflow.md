# 04 – Big Data ML Workflow

Execute the Spark feature engineering job and the TensorFlow trainer that consume the shared feature store.

## 1. Confirm prerequisites
- Spark master and workers are running (`docker-compose ps`).
- The joined dataset (`data/f1_results_joined.csv`) exists – generate it via runbook 02 if missing.

## 2. Submit the Spark job
```powershell
docker-compose exec spark-master \
  spark-submit /app/pipelines/bigdata/spark/train_driver_win_mllib.py
```
Key outputs:
- Feature store shards under `artifacts/feature_store/` (Parquet + CSV).
- Spark model artefact in `artifacts/models/mllib_driver_win_rf`.
- Evaluation JSON in `artifacts/evaluations/mllib_driver_win_metrics.json`.

## 3. Run the TensorFlow trainer
Activate a Python environment with TensorFlow (or reuse a container) and execute:
```powershell
python pipelines/bigdata/tensorflow/train_tensorflow_bigdata.py
```
Outputs:
- Saved Keras model at `artifacts/models/tf_driver_win`.
- Training history (`tf_driver_win_history.json`) and evaluation metrics (`tf_driver_win_metrics.json`) in `artifacts/evaluations/`.

## 4. Explore artefacts
Start the big data dashboard to inspect metrics and feature samples:
```powershell
docker-compose up -d dashboard_bigdata
```
Visit `http://localhost:8502` for live charts and artefact status indicators.

## 5. Troubleshooting tips
- Missing feature shards: rerun the Spark job and ensure `F1_FEATURE_SOURCE` points to the correct CSV.
- TensorFlow dataset errors: confirm the feature store path matches the Spark output and that shards contain data.
- Slow training: lower `F1_TF_BATCH_SIZE` or reduce epochs via environment variables before executing the trainer.
