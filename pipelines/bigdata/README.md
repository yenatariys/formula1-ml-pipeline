# Big Data Pipelines

This folder houses the distributed workflows that power the Formula 1 feature store and deep-learning models. The Spark job engineers large-scale driver features, and the TensorFlow trainer consumes those artefacts for neural-network experiments.

## Directory layout
- `spark/` – Spark MLlib training job, supporting utilities, and the base Docker image used by the cluster.
- `tensorflow/` – TensorFlow training script that streams feature shards via `tf.data` and supports `tf.distribute` strategies.
- `docs/` – placeholder for architecture notes and future design diagrams.

## Prerequisites
1. Docker Compose stack with Postgres, Spark master, and Spark workers from `docker-compose.yml`.
   - Ensure the Spark services point to this directory as their build context. If needed, run:
     ```powershell
     docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1 spark-master spark-worker-1 spark-worker-2
     ```
     or update `docker-compose.yml` to reference `pipelines/bigdata/spark`.
2. Base data staged under `data/` (default source is `data/f1_results_joined.csv`).
3. Python environment with TensorFlow 2.16+ (host or dedicated container). From the repo root:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
4. Optional: update environment variables to tweak locations or model hyperparameters (see table below).

## End-to-end workflow
1. **Prepare infrastructure and data**
   - `docker-compose up -d postgres spark-master spark-worker-1 spark-worker-2`
   - Run the ETL service at least once so artefacts in `data/` and the `f1_results_transformed` table are current.

2. **Engineer features with Spark MLlib**
   ```powershell
   docker-compose exec spark-master spark-submit /app/pipelines/bigdata/spark/train_driver_win_mllib.py
   ```
   - The job reads from `F1_FEATURE_SOURCE` (default `/app/data/f1_results_joined.csv`).
   - Engineered features are saved to `/app/artifacts/feature_store` (Parquet + CSV) and metrics/models land in `/app/artifacts/evaluations` and `/app/artifacts/models`.

3. **Train the TensorFlow model**
   - Run locally after activating your virtual environment **or** from a GPU-enabled container:
     ```powershell
     python pipelines/bigdata/tensorflow/train_tensorflow_bigdata.py
     ```
   - The script streams the CSV shards written by Spark, trains with `tf.distribute.MirroredStrategy`, and stores models/metrics under `artifacts/`.

4. **Inspect outputs**
   - Feature store: `artifacts/feature_store/driver_features_{parquet,csv}`
   - Spark metrics: `artifacts/evaluations/mllib_driver_win_metrics.json`
   - TensorFlow metrics: `artifacts/evaluations/tf_driver_win_metrics.json`
   - Models: `artifacts/models/`

## Key environment variables
| Variable | Default | Purpose |
| --- | --- | --- |
| `F1_FEATURE_SOURCE` | `/app/data/f1_results_joined.csv` | Source data for Spark feature engineering. |
| `F1_FEATURE_STORE_PATH` | `/app/artifacts/feature_store` | Destination for shared feature artefacts. |
| `F1_RF_NUM_TREES` | `200` | Number of trees in the Spark RandomForest. |
| `F1_RF_MAX_DEPTH` | `8` | Maximum tree depth for Spark RandomForest. |
| `F1_TRAIN_FRACTION` | `0.8` | Train/test split fraction in Spark. |
| `F1_TF_BATCH_SIZE` | `512` | Batch size for TensorFlow dataset streaming. |
| `F1_TF_EPOCHS` | `15` | Training epochs for the TensorFlow model. |
| `F1_TF_MODEL_PATH` | `/app/artifacts/models/tf_driver_win` | Destination directory for the saved Keras model. |

Set variables inline when running commands, for example:
```powershell
$env:F1_FEATURE_SOURCE = "/app/data/custom_results.csv"
docker-compose exec spark-master spark-submit /app/pipelines/bigdata/spark/train_driver_win_mllib.py
```

## Troubleshooting
- Spark job fails with `FileNotFoundError`: confirm the feature source path exists inside the container and that volumes are mounted correctly.
- Empty TensorFlow datasets: ensure the Spark step completed and wrote CSV shards under `artifacts/feature_store/driver_features_csv`.
- Slow TensorFlow training: lower `F1_TF_BATCH_SIZE` or reduce epochs while prototyping.
- Cluster connectivity issues: check that Spark master and workers are healthy (`docker-compose ps`) before submitting jobs.
