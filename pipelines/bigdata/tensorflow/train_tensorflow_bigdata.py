"""TensorFlow (distributed-ready) training pipeline for Formula 1 predictions.

This script consumes the driver feature store exported by
``pipelines/bigdata/spark/train_driver_win_mllib.py`` and trains a neural
network using TensorFlow's ``tf.data`` API together with ``tf.distribute`` so
it can scale from a single machine to a multi-worker cluster without code
changes. The input reader streams CSV shards directly from disk, making it
suitable for large datasets stored on shared volumes or cloud object
storage.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Tuple

import tensorflow as tf

FEATURE_COLUMNS = ["driverId", "year", "round", "win_rate", "avg_points"]
LABEL_COLUMN = "win"
COLUMN_DEFAULTS = [0, 0, 0, 0.0, 0.0, 0]


def _feature_store_glob() -> str:
    base_path = os.getenv("F1_FEATURE_STORE_PATH", "/app/artifacts/feature_store")
    csv_dir = os.path.join(base_path, "driver_features_csv")
    if not tf.io.gfile.exists(csv_dir):
        raise FileNotFoundError(
            "Spark feature store shards not found. Run pipelines/bigdata/spark/"
            "train_driver_win_mllib.py first or set F1_FEATURE_STORE_PATH to a directory "
            f"containing CSV shards (missing: {csv_dir})."
        )
    return os.path.join(csv_dir, "*")


def _make_base_dataset(batch_size: int) -> tf.data.Dataset:
    file_pattern = _feature_store_glob()
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        column_names=FEATURE_COLUMNS + [LABEL_COLUMN],
        column_defaults=COLUMN_DEFAULTS,
        label_name=LABEL_COLUMN,
        num_epochs=1,
        shuffle=True,
        shuffle_buffer_size=batch_size * 25,
        num_parallel_reads=tf.data.AUTOTUNE,
        sloppy=True,
    )
    dataset = dataset.map(_pack_features, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.enumerate()
    dataset = dataset.cache()
    return dataset


def _pack_features(features: dict, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    stacked = tf.stack(
        [tf.cast(features[name], tf.float32) for name in FEATURE_COLUMNS], axis=-1
    )
    return stacked, tf.cast(label, tf.float32)


def _split_datasets(dataset: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    def strip_index(_idx: tf.Tensor, data: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        return data

    def is_train(idx: tf.Tensor, _data) -> tf.Tensor:
        return tf.math.less(tf.math.floormod(idx, 10), 7)

    def is_val(idx: tf.Tensor, _data) -> tf.Tensor:
        mod = tf.math.floormod(idx, 10)
        return tf.math.logical_and(tf.math.greater_equal(mod, 7), tf.math.less(mod, 9))

    def is_test(idx: tf.Tensor, _data) -> tf.Tensor:
        return tf.math.equal(tf.math.floormod(idx, 10), 9)

    train = dataset.filter(is_train).map(strip_index).cache().prefetch(tf.data.AUTOTUNE)
    val = dataset.filter(is_val).map(strip_index).cache().prefetch(tf.data.AUTOTUNE)
    test = dataset.filter(is_test).map(strip_index).cache().prefetch(tf.data.AUTOTUNE)

    return train, val, test


def _build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(FEATURE_COLUMNS),)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(os.getenv("F1_TF_LR", "0.001"))),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )
    return model


def _train_and_evaluate(
    strategy: tf.distribute.Strategy,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    epochs: int,
) -> Tuple[tf.keras.Model, dict, Iterable[Tuple[str, float]]]:
    with strategy.scope():
        model = _build_model()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    test_metrics = model.evaluate(test_ds, return_dict=True)
    return model, history.history, test_metrics.items()


def main() -> None:
    batch_size = int(os.getenv("F1_TF_BATCH_SIZE", "512"))
    epochs = int(os.getenv("F1_TF_EPOCHS", "15"))

    base_dataset = _make_base_dataset(batch_size=batch_size)
    train_ds, val_ds, test_ds = _split_datasets(base_dataset)

    for name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        cardinality = tf.data.experimental.cardinality(dataset)
        if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
            raise RuntimeError(
                f"Unable to determine cardinality for {name} dataset. Check feature shards."
            )
        if cardinality.numpy() == 0:
            raise RuntimeError(
                f"{name} dataset is empty. Ensure Spark feature export produced enough batches."
            )
        print(f"{name.capitalize()} batches: {cardinality.numpy()}")

    if os.getenv("TF_CONFIG"):
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        strategy = tf.distribute.MirroredStrategy()

    model, history, test_metrics = _train_and_evaluate(
        strategy, train_ds, val_ds, test_ds, epochs
    )

    model_dir = os.getenv("F1_TF_MODEL_PATH", "/app/artifacts/models/tf_driver_win")
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)

    metrics_dir = os.getenv("F1_TF_EVAL_PATH", "/app/artifacts/evaluations")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "tf_driver_win_history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, sort_keys=True)
    with open(os.path.join(metrics_dir, "tf_driver_win_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({k: float(v) for k, v in test_metrics}, handle, indent=2, sort_keys=True)

    print("TensorFlow training complete. Test metrics:")
    for key, value in test_metrics:
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
