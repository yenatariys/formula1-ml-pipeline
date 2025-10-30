# Big Data ML Upgrade

This document captures the updated machine learning workflow that extends the
Formula 1 pipeline with Spark MLlib and TensorFlow on large datasets. The
original scikit-learn based training scripts remain available (`ml/train.py`),
while the new code paths live alongside them so you can compare the
before/after implementations.

## Overview

```
+-------------------------+        +---------------------------+
|   Raw / curated data    |        |   Feature store shards    |
| (CSV, Parquet, Postgres)|        | (Parquet + CSV shards)    |
+------------+------------+        +--------------+------------+
             |                                     |
             v                                     v
   spark/train_driver_win_mllib.py        ml/train_tensorflow_bigdata.py
             |                                     |
     Spark MLlib model                    TensorFlow distributed model
     (RandomForest)                        (Deep neural network)
             |                                     |
             v                                     v
    artifacts/models/...                  artifacts/models/...
    artifacts/evaluations/...             artifacts/evaluations/...
```

Key additions:

- `spark/train_driver_win_mllib.py`: Spark job that engineers historical
  driver features using window functions, trains an MLlib RandomForest at
  cluster scale, and writes a reusable feature store (Parquet & CSV shards).
- `ml/train_tensorflow_bigdata.py`: TensorFlow pipeline that streams the
  Spark-generated CSV shards through `tf.data`, supports `tf.distribute`
  strategies, and saves both metrics and the trained model.
- Output directories in `/app/artifacts` (configurable via environment
  variables) to keep models, evaluations, and feature store files side by
  side for downstream consumers.

## Running the Spark MLlib job

1. Make sure the Spark cluster in `docker-compose.yml` is up (`spark-master`
   + workers if configured) and the feature source CSV is available at the
   path referenced by `F1_FEATURE_SOURCE` (defaults to
   `/app/f1_results_transformed.csv`).
2. Submit the job from the repo root (WSL/Docker container):

   ```bash
   spark-submit \
     --master spark://spark-master:7077 \
     --deploy-mode client \
     spark/train_driver_win_mllib.py
   ```

   Optional environment variables:

   - `F1_FEATURE_SOURCE`: Path or URI to the input CSV.
   - `F1_FEATURE_STORE_PATH`: Where engineered features are stored.
   - `F1_MLLIB_MODEL_PATH`, `F1_MLLIB_PREDICTIONS_PATH`.
   - `F1_RF_NUM_TREES`, `F1_RF_MAX_DEPTH`, `F1_TRAIN_FRACTION`.

3. Outputs:

   - Feature store shards in `${F1_FEATURE_STORE_PATH}` (Parquet + CSV).
   - Model artefact in `${F1_MLLIB_MODEL_PATH}`.
   - Evaluation JSON (`mllib_driver_win_metrics.json`).

## Running the TensorFlow (big data) pipeline

1. Ensure the Spark job above has materialised the feature store (the TF
   script validates that the CSV shards exist).
2. Install TensorFlow (CPU or GPU build) and optional accelerators
   (see `requirements.txt`).
3. Execute the training script:

   ```bash
   python ml/train_tensorflow_bigdata.py
   ```

   Important environment variables:

   - `F1_FEATURE_STORE_PATH`: Feature shard directory (defaults to
     `/app/artifacts/feature_store`).
   - `F1_TF_BATCH_SIZE`, `F1_TF_EPOCHS`, `F1_TF_LR`.
   - `F1_TF_MODEL_PATH`, `F1_TF_EVAL_PATH` for outputs.
   - `TF_CONFIG`: Provide cluster spec JSON for multi-worker runs. When set,
     the script automatically switches to `MultiWorkerMirroredStrategy`; it
     defaults to `MirroredStrategy` for single-host multi-GPU/CPU training.

4. Outputs:

   - Saved Keras model in `${F1_TF_MODEL_PATH}`.
   - Training history (`tf_driver_win_history.json`) and final evaluation
     metrics (`tf_driver_win_metrics.json`).

## Notes

- The Spark job purposely keeps the engineered dataset in distributed
  storage so both MLlib and TensorFlow stages can consume the same source
  without copying data through the driver.
- The TensorFlow pipeline splits batches deterministically using the batch
  index (0-9) to create 70/20/10 train/validation/test splits, enabling
  streaming over arbitrarily large shard sets.
- Extend either pipeline with additional features by updating the Spark
  feature engineering logic; the TensorFlow reader automatically adapts as
  long as new columns are appended before the label column.
- For production, wire these scripts into your orchestrator (Airflow,
  Kubeflow, Databricks Jobs, etc.) and track artefacts with MLflow or similar
  registry tooling.
