import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]


def _resolve_path(env_key: str, default_value: str) -> Path:
    raw_path = os.getenv(env_key, default_value)
    path = Path(raw_path)
    if not path.is_absolute():
        path = BASE_DIR / raw_path
    return path


FEATURE_STORE_ROOT = _resolve_path("F1_FEATURE_STORE_PATH", "artifacts/feature_store")
FEATURE_CSV_DIR = FEATURE_STORE_ROOT / "driver_features_csv"
SPARK_EVAL_DIR = _resolve_path("F1_MLLIB_EVAL_PATH", "artifacts/evaluations")
SPARK_METRICS_PATH = SPARK_EVAL_DIR / "mllib_driver_win_metrics.json"
PREDICTIONS_DIR = _resolve_path("F1_MLLIB_PREDICTIONS_PATH", "artifacts/predictions/mllib_driver_win")
TF_EVAL_DIR = _resolve_path("F1_TF_EVAL_PATH", "artifacts/evaluations")
TF_HISTORY_PATH = TF_EVAL_DIR / "tf_driver_win_history.json"
TF_METRICS_PATH = TF_EVAL_DIR / "tf_driver_win_metrics.json"


st.set_page_config(page_title="F1 Big Data Pipeline Dashboard", layout="wide")
st.title("🧠 Formula 1 Big Data Pipeline Dashboard")
st.caption("Monitoring artefacts and metrics produced by the Spark + TensorFlow workflows.")


@st.cache_data(show_spinner=False)
def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def _load_feature_sample(max_rows: int = 5000) -> pd.DataFrame:
    if not FEATURE_CSV_DIR.exists():
        return pd.DataFrame()

    shards = [p for p in FEATURE_CSV_DIR.iterdir() if p.suffix == ".csv"]
    if not shards:
        return pd.DataFrame()

    frames = []
    rows_loaded = 0
    for shard in sorted(shards):
        chunk = pd.read_csv(shard)
        frames.append(chunk)
        rows_loaded += len(chunk)
        if rows_loaded >= max_rows:
            break

    if not frames:
        return pd.DataFrame()

    sample = pd.concat(frames, ignore_index=True)
    return sample.head(max_rows)


def _format_timestamp(path: Path) -> str:
    if not path.exists():
        return "—"
    ts = datetime.fromtimestamp(path.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _artefact_status(path: Path) -> str:
    return "✅ Available" if path.exists() else "⚠️ Missing"


# Artefact status overview
st.header("Artefact Status")
col_feature, col_spark, col_tf = st.columns(3)

with col_feature:
    st.metric("Feature Store CSV", _artefact_status(FEATURE_CSV_DIR))
    st.caption(f"Location: {FEATURE_CSV_DIR}")

with col_spark:
    st.metric("Spark Metrics", _artefact_status(SPARK_METRICS_PATH))
    st.caption(f"Last update: {_format_timestamp(SPARK_METRICS_PATH)}")

with col_tf:
    st.metric("TensorFlow Metrics", _artefact_status(TF_METRICS_PATH))
    st.caption(f"Last update: {_format_timestamp(TF_METRICS_PATH)}")

spark_metrics = _load_json(SPARK_METRICS_PATH)
tf_metrics = _load_json(TF_METRICS_PATH)
tf_history = _load_json(TF_HISTORY_PATH)
feature_sample = _load_feature_sample()

if spark_metrics:
    total_records = spark_metrics.get("train_records", 0) + spark_metrics.get("test_records", 0)
else:
    total_records = len(feature_sample)

st.divider()

# Feature store insights
st.header("Feature Store Preview")
if feature_sample.empty:
    st.info("No feature shards detected yet. Run the Spark pipeline to populate `artifacts/feature_store`.\n")
else:
    st.metric("Sample Rows Loaded", len(feature_sample))
    st.metric("Approx. Training Rows", total_records)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Win Rate Distribution")
        fig_win_rate = px.histogram(
            feature_sample,
            x="win_rate",
            nbins=40,
            title="Distribution of Rolling Win Rate",
            labels={"win_rate": "Win Rate"},
        )
        st.plotly_chart(fig_win_rate, use_container_width=True)

    with col_b:
        st.subheader("Average Points vs Win Rate")
        fig_scatter = px.scatter(
            feature_sample,
            x="avg_points",
            y="win_rate",
            color="win",
            title="Driver Form Snapshot",
            labels={"avg_points": "Average Points", "win_rate": "Win Rate", "win": "Label"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Sample Records")
    st.dataframe(feature_sample.head(200), use_container_width=True)

st.divider()

# Spark model metrics
st.header("Spark MLlib Model Metrics")
if not spark_metrics:
    st.warning("Spark evaluation artefacts not found. Re-run `pipelines/bigdata/spark/train_driver_win_mllib.py`.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{spark_metrics.get('accuracy', 0):.3f}")
    col2.metric("ROC-AUC", f"{spark_metrics.get('roc_auc', 0):.3f}")
    col3.metric("Train/Test Split", f"{spark_metrics.get('train_records', 0)} / {spark_metrics.get('test_records', 0)}")

    feature_importances = spark_metrics.get("feature_importances", {})
    if feature_importances:
        fi_df = pd.DataFrame(
            sorted(feature_importances.items(), key=lambda item: item[1], reverse=True),
            columns=["Feature", "Importance"],
        )
        st.subheader("Feature Importances")
        fig_fi = px.bar(fi_df, x="Feature", y="Importance", title="Spark RandomForest Feature Importances")
        st.plotly_chart(fig_fi, use_container_width=True)

st.divider()

# TensorFlow metrics
st.header("TensorFlow Model Metrics")
if not tf_metrics:
    st.info("TensorFlow evaluation artefacts not found. Run `pipelines/bigdata/tensorflow/train_tensorflow_bigdata.py` after the Spark step.")
else:
    metric_cols = st.columns(len(tf_metrics))
    for (name, value), column in zip(tf_metrics.items(), metric_cols):
        column.metric(name.replace("_", " ").title(), f"{value:.3f}")

    if tf_history:
        history_df = pd.DataFrame(tf_history)
        history_df["epoch"] = history_df.index + 1
        st.subheader("Training History")
        metric_names = [col for col in history_df.columns if col != "epoch"]
        fig_history = px.line(
            history_df,
            x="epoch",
            y=metric_names,
            title="TensorFlow Training Curves",
            labels={"epoch": "Epoch", "value": "Metric"},
        )
        st.plotly_chart(fig_history, use_container_width=True)

        st.caption("Source: artifacts/evaluations/tf_driver_win_history.json")

st.divider()

st.header("Prediction Artefacts")
if not PREDICTIONS_DIR.exists():
    st.info("Spark prediction parquet artefacts not found yet.")
else:
    shards = [p for p in PREDICTIONS_DIR.glob("*.parquet")]
    st.write(f"Found {len(shards)} parquet shard(s) under {PREDICTIONS_DIR}")
    st.caption("Use Spark or pandas with pyarrow to explore full prediction outputs as needed.")
