"""Spark MLlib training pipeline for Formula 1 driver win prediction.

This module demonstrates a scalable feature engineering and model training
workflow that can be submitted to a Spark cluster. It consumes the
``f1_results_transformed.csv`` dataset (or another CSV specified via the
``F1_FEATURE_SOURCE`` environment variable), computes rolling performance
features per driver using Spark window functions, trains a Random Forest
classifier with Spark MLlib, and persists both the fitted PipelineModel and
its evaluation artefacts to disk.

The script intentionally avoids collecting the dataset to the driver node so
it can operate on large inputs stored on shared volumes (e.g. HDFS, S3, or a
Kubernetes persistent volume). Output artefacts are written back to a shared
location so downstream consumers (TensorFlow training, dashboards, etc.) can
reuse the engineered dataset.
"""

from __future__ import annotations

import json
import os
from typing import Dict

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def _build_spark_session() -> SparkSession:
    """Create or reuse a Spark session configured for cluster execution."""

    spark_master = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
    app_name = os.getenv("SPARK_APP_NAME", "F1DriverWinSparkMLlib")

    spark = (
        SparkSession.builder.appName(app_name)
        .master(spark_master)
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SQL_PARTITIONS", "200"))
        .getOrCreate()
    )

    return spark


def _load_source_dataframe(spark: SparkSession) -> DataFrame:
    """Load the source dataset from CSV (or another format in the future)."""

    source_path = os.getenv("F1_FEATURE_SOURCE", "/app/data/f1_results_joined.csv")
    if not source_path:
        raise ValueError("F1_FEATURE_SOURCE environment variable must be set")

    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(source_path)
    )

    if "position" not in df.columns:
        raise ValueError("Expected column 'position' not present in source dataset")

    required_columns = {"driverId", "points", "year", "round"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {sorted(missing)}")

    return df


def _engineer_features(df: DataFrame) -> DataFrame:
    """Engineer rolling driver performance features using Spark window ops."""

    ordering = Window.partitionBy("driverId").orderBy("year", "round")
    cumulative_excluding_current = ordering.rowsBetween(Window.unboundedPreceding, -1)

    df_enriched = (
        df.withColumn("win", F.when(F.col("position") == 1, F.lit(1)).otherwise(F.lit(0)))
        .withColumn("points", F.col("points").cast("double"))
        .withColumn("year", F.col("year").cast("int"))
        .withColumn("round", F.col("round").cast("int"))
        .withColumn("races_so_far", F.row_number().over(ordering) - F.lit(1))
        .withColumn(
            "wins_so_far",
            F.coalesce(F.sum("win").over(cumulative_excluding_current), F.lit(0)),
        )
        .withColumn(
            "points_so_far",
            F.coalesce(F.sum("points").over(cumulative_excluding_current), F.lit(0.0)),
        )
        .withColumn(
            "win_rate",
            F.when(F.col("races_so_far") > 0, F.col("wins_so_far") / F.col("races_so_far"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "avg_points",
            F.when(F.col("races_so_far") > 0, F.col("points_so_far") / F.col("races_so_far"))
            .otherwise(F.lit(0.0)),
        )
        .filter(F.col("races_so_far") >= F.lit(5))
        .select("driverId", "year", "round", "win_rate", "avg_points", "win")
    )

    return df_enriched


def _persist_feature_store(features: DataFrame, base_path: str) -> None:
    """Persist engineered features to Parquet and CSV shards for reuse."""

    parquet_path = os.path.join(base_path, "driver_features_parquet")
    csv_path = os.path.join(base_path, "driver_features_csv")

    features.write.mode("overwrite").parquet(parquet_path)
    (
        features.coalesce(int(os.getenv("F1_FEATURE_CSV_SHARDS", "1")))
        .write.mode("overwrite")
        .option("header", "true")
        .csv(csv_path)
    )


def _train_model(features: DataFrame) -> Dict[str, float]:
    """Train a Spark MLlib RandomForest classifier and evaluate it."""

    assembler = VectorAssembler(
        inputCols=["year", "round", "win_rate", "avg_points"], outputCol="features"
    )
    classifier = RandomForestClassifier(
        labelCol="win",
        featuresCol="features",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        numTrees=int(os.getenv("F1_RF_NUM_TREES", "200")),
        maxDepth=int(os.getenv("F1_RF_MAX_DEPTH", "8")),
        seed=int(os.getenv("F1_RANDOM_SEED", "42")),
    )
    pipeline = Pipeline(stages=[assembler, classifier])

    train_fraction = float(os.getenv("F1_TRAIN_FRACTION", "0.8"))
    train_data, test_data = features.randomSplit([train_fraction, 1 - train_fraction], 42)
    train_data = train_data.cache()
    test_data = test_data.cache()

    model = pipeline.fit(train_data)

    metrics = _evaluate_model(model, test_data)
    metrics["train_records"] = train_data.count()
    metrics["test_records"] = test_data.count()

    _persist_model(model)
    return metrics


def _evaluate_model(pipeline_model: Pipeline, test_data: DataFrame) -> Dict[str, float]:
    """Evaluate accuracy and ROC-AUC on held-out data."""

    predictions = pipeline_model.transform(test_data)
    predictions.cache()

    accuracy_eval = MulticlassClassificationEvaluator(
        labelCol="win", predictionCol="prediction", metricName="accuracy"
    )
    auc_eval = BinaryClassificationEvaluator(
        labelCol="win", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )

    accuracy = accuracy_eval.evaluate(predictions)
    roc_auc = auc_eval.evaluate(predictions)

    feature_importances = pipeline_model.stages[-1].featureImportances
    feature_map = {
        "year": float(feature_importances[0]),
        "round": float(feature_importances[1]),
        "win_rate": float(feature_importances[2]),
        "avg_points": float(feature_importances[3]),
    }

    evaluation_path = os.getenv("F1_MLLIB_EVAL_PATH", "/app/artifacts/evaluations")
    os.makedirs(evaluation_path, exist_ok=True)

    metrics = {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "feature_importances": feature_map,
    }

    predictions_output = os.getenv(
        "F1_MLLIB_PREDICTIONS_PATH", "/app/artifacts/predictions/mllib_driver_win"
    )
    os.makedirs(os.path.dirname(predictions_output), exist_ok=True)
    (
        predictions.select("driverId", "year", "round", "prediction", "probability", "win")
        .write.mode("overwrite")
        .parquet(predictions_output)
    )

    with open(os.path.join(evaluation_path, "mllib_driver_win_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    return metrics


def _persist_model(pipeline_model: Pipeline) -> None:
    """Persist the trained model for later inference."""

    model_output = os.getenv(
        "F1_MLLIB_MODEL_PATH", "/app/artifacts/models/mllib_driver_win_rf"
    )
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    pipeline_model.write().overwrite().save(model_output)


def main() -> None:
    spark = _build_spark_session()
    try:
        source_df = _load_source_dataframe(spark)
        features_df = _engineer_features(source_df)

        feature_store_path = os.getenv("F1_FEATURE_STORE_PATH", "/app/artifacts/feature_store")
        os.makedirs(feature_store_path, exist_ok=True)
        _persist_feature_store(features_df, feature_store_path)

        metrics = _train_model(features_df)

        print("Spark MLlib training finished successfully.")
        print("Evaluation metrics:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value:.4f}")
            else:
                print(f"  {key}: {value}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
