"""Unified Formula 1 ML Dashboard - Classic ML + Big Data Pipeline"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="F1 ML Dashboard - Unified",
    layout="wide",
    initial_sidebar_state="expanded"
)


def _resolve_path(env_key: str, default_value: str) -> Path:
    """Resolve artifact paths from environment or default."""
    raw_path = os.getenv(env_key, default_value)
    path = Path(raw_path)
    if not path.is_absolute():
        path = BASE_DIR / raw_path
    return path


# Big Data Pipeline Paths
FEATURE_STORE_ROOT = _resolve_path("F1_FEATURE_STORE_PATH", "artifacts/feature_store")
FEATURE_CSV_DIR = FEATURE_STORE_ROOT / "driver_features_csv"
SPARK_EVAL_DIR = _resolve_path("F1_MLLIB_EVAL_PATH", "artifacts/evaluations")
SPARK_METRICS_PATH = SPARK_EVAL_DIR / "mllib_driver_win_metrics.json"
TF_EVAL_DIR = _resolve_path("F1_TF_EVAL_PATH", "artifacts/evaluations")
TF_HISTORY_PATH = TF_EVAL_DIR / "tf_driver_win_history.json"
TF_METRICS_PATH = TF_EVAL_DIR / "tf_driver_win_metrics.json"

# Database connection for Classic ML
try:
    engine = create_engine("postgresql+psycopg2://admin:admin123@f1_postgres:5432/f1_data")
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False


# ============================================================================
# Utility Functions
# ============================================================================

@st.cache_data(show_spinner=False)
def _load_json(path: Path) -> Optional[Dict]:
    """Load JSON file if it exists."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def _load_feature_sample(max_rows: int = 5000) -> pd.DataFrame:
    """Load sample of engineered features from CSV shards."""
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


@st.cache_data
def load_results():
    """Load race results from database."""
    if not DB_AVAILABLE:
        return pd.DataFrame()
    try:
        return pd.read_sql("SELECT * FROM f1_results_transformed", engine)
    except Exception:
        return pd.DataFrame()


def _format_timestamp(path: Path) -> str:
    """Format file modification timestamp."""
    if not path.exists():
        return "—"
    ts = datetime.fromtimestamp(path.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _artefact_status(path: Path) -> str:
    """Return status emoji for artifact."""
    return "✅ Available" if path.exists() else "⚠️ Missing"


# ============================================================================
# Big Data Pipeline Dashboard Components
# ============================================================================

def render_bigdata_overview():
    """Render big data pipeline overview section."""
    st.header("📊 Big Data Pipeline Overview")
    
    col_feature, col_spark, col_tf = st.columns(3)
    
    with col_feature:
        st.metric("Feature Store CSV", _artefact_status(FEATURE_CSV_DIR))
        st.caption(f"Location: {FEATURE_CSV_DIR}")
    
    with col_spark:
        st.metric("Spark MLlib Metrics", _artefact_status(SPARK_METRICS_PATH))
        st.caption(f"Last update: {_format_timestamp(SPARK_METRICS_PATH)}")
    
    with col_tf:
        st.metric("TensorFlow Metrics", _artefact_status(TF_METRICS_PATH))
        st.caption(f"Last update: {_format_timestamp(TF_METRICS_PATH)}")


def render_spark_mllib_metrics():
    """Render Spark MLlib model metrics."""
    st.header("⚡ Spark MLlib Model Metrics")
    
    spark_metrics = _load_json(SPARK_METRICS_PATH)
    
    if not spark_metrics:
        st.warning(
            "Spark evaluation artefacts not found. Re-run "
            "`pipelines/bigdata/spark/train_driver_win_mllib.py`."
        )
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{spark_metrics.get('accuracy', 0):.4f}")
    
    with col2:
        st.metric("ROC-AUC", f"{spark_metrics.get('roc_auc', 0):.4f}")
    
    with col3:
        st.metric("Train Records", spark_metrics.get("train_records", "N/A"))
    
    with col4:
        st.metric("Test Records", spark_metrics.get("test_records", "N/A"))
    
    # Feature importances
    if "feature_importances" in spark_metrics:
        st.subheader("Feature Importances")
        
        importances = spark_metrics["feature_importances"]
        importance_df = pd.DataFrame(
            list(importances.items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=False)
        
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importances",
            color="Importance",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_tensorflow_metrics():
    """Render TensorFlow model metrics."""
    st.header("🧠 TensorFlow Model Metrics")
    
    tf_metrics = _load_json(TF_METRICS_PATH)
    tf_history = _load_json(TF_HISTORY_PATH)
    
    if not tf_metrics:
        st.warning(
            "TensorFlow evaluation artefacts not found. Re-run "
            "`pipelines/bigdata/tensorflow/train_tensorflow_bigdata.py`."
        )
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Loss", f"{tf_metrics.get('loss', 0):.3f}")
    
    with col2:
        st.metric("Precision", f"{tf_metrics.get('precision', 0):.3f}")
    
    with col3:
        st.metric("Recall", f"{tf_metrics.get('recall', 0):.3f}")
    
    with col4:
        st.metric("ROC-AUC", f"{tf_metrics.get('roc_auc', 0):.3f}")
    
    # Training history
    if tf_history:
        st.subheader("Training History")
        
        # Prepare data for plotting
        epochs = list(range(1, len(tf_history.get("loss", [])) + 1))
        
        # Loss plot
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs,
            y=tf_history.get("loss", []),
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='blue')
        ))
        if "val_loss" in tf_history:
            fig_loss.add_trace(go.Scatter(
                x=epochs,
                y=tf_history.get("val_loss", []),
                mode='lines+markers',
                name='Validation Loss',
                line=dict(color='red', dash='dash')
            ))
        fig_loss.update_layout(
            title="Loss Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode='x unified'
        )
        
        # ROC-AUC plot
        fig_auc = go.Figure()
        fig_auc.add_trace(go.Scatter(
            x=epochs,
            y=tf_history.get("roc_auc", []),
            mode='lines+markers',
            name='Training ROC-AUC',
            line=dict(color='green')
        ))
        if "val_roc_auc" in tf_history:
            fig_auc.add_trace(go.Scatter(
                x=epochs,
                y=tf_history.get("val_roc_auc", []),
                mode='lines+markers',
                name='Validation ROC-AUC',
                line=dict(color='orange', dash='dash')
            ))
        fig_auc.update_layout(
            title="ROC-AUC Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="ROC-AUC",
            hovermode='x unified'
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_loss, use_container_width=True)
        with col2:
            st.plotly_chart(fig_auc, use_container_width=True)


def render_feature_store():
    """Render feature store sample."""
    st.header("📦 Feature Store Sample")
    
    features_df = _load_feature_sample()
    
    if features_df.empty:
        st.warning("Feature store data not available.")
        return
    
    st.write(f"Showing {len(features_df):,} records from feature store")
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(features_df):,}")
    
    with col2:
        wins = features_df["win"].sum()
        st.metric("Wins", f"{wins:,}")
    
    with col3:
        win_rate = (features_df["win"].mean() * 100)
        st.metric("Win Rate", f"{win_rate:.2f}%")
    
    # Show sample data
    st.dataframe(features_df.head(100), use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(
            features_df,
            x="win_rate",
            nbins=50,
            title="Distribution of Driver Win Rate",
            color_discrete_sequence=["#636EFA"]
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.histogram(
            features_df,
            x="avg_points",
            nbins=50,
            title="Distribution of Average Points",
            color_discrete_sequence=["#EF553B"]
        )
        st.plotly_chart(fig2, use_container_width=True)


def render_model_comparison():
    """Render comparison between all models."""
    st.header("🔄 Model Comparison")
    
    spark_metrics = _load_json(SPARK_METRICS_PATH)
    tf_metrics = _load_json(TF_METRICS_PATH)
    
    if not spark_metrics and not tf_metrics:
        st.warning("No model metrics available for comparison.")
        return
    
    # Create comparison dataframe
    comparison_data = []
    
    if spark_metrics:
        comparison_data.append({
            "Model": "Spark MLlib (Random Forest)",
            "Accuracy": spark_metrics.get("accuracy", 0),
            "ROC-AUC": spark_metrics.get("roc_auc", 0),
            "Type": "Big Data - Tree-based"
        })
    
    if tf_metrics:
        comparison_data.append({
            "Model": "TensorFlow (Neural Network)",
            "Accuracy": tf_metrics.get("loss", 0),  # Note: TF doesn't track accuracy directly
            "ROC-AUC": tf_metrics.get("roc_auc", 0),
            "Precision": tf_metrics.get("precision", 0),
            "Recall": tf_metrics.get("recall", 0),
            "Type": "Big Data - Neural Network"
        })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # ROC-AUC comparison
        fig = px.bar(
            df_comparison,
            x="Model",
            y="ROC-AUC",
            title="Model Performance Comparison (ROC-AUC)",
            color="Type",
            text="ROC-AUC"
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("Detailed Metrics")
        st.dataframe(df_comparison, use_container_width=True)


# ============================================================================
# Classic ML Dashboard Components
# ============================================================================

def render_classic_overview():
    """Render classic ML overview with race results."""
    st.header("🏎️ Formula 1 Race Results & Classic ML")
    
    if not DB_AVAILABLE:
        st.error("Database connection not available.")
        return
    
    df = load_results()
    
    if df.empty:
        st.warning("No race results data available.")
        return
    
    # Show total records
    st.metric("Total Race Results", f"{len(df):,}")
    
    # Year filter
    year = st.selectbox("Select Season", sorted(df["year"].unique(), reverse=True))
    filtered = df[df["year"] == year].copy()

    # Ensure 'position' is numeric for aggregation
    filtered["position"] = pd.to_numeric(filtered["position"], errors="coerce")

    col1, col2 = st.columns(2)

    with col1:
        # Wins by driver (position 1)
        winners = filtered[filtered["position"] == 1].groupby("surname")["raceId"].count().reset_index()
        winners.columns = ["Driver", "Wins"]
        fig1 = px.bar(
            winners,
            x="Driver",
            y="Wins",
            title=f"Race Wins by Driver ({year})",
            color="Wins",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Points distribution
        points_by_driver = (
            filtered.groupby("surname")["points"]
            .sum()
            .reset_index()
            .sort_values("points", ascending=False)
            .head(10)
        )
        points_by_driver.columns = ["Driver", "Total Points"]
        fig2 = px.pie(
            points_by_driver,
            values="Total Points",
            names="Driver",
            title=f"Top 10 Drivers by Points ({year})"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Detailed results table
    st.subheader(f"Detailed Results - {year}")
    display_cols = ["round", "name", "surname", "position", "points", "grid", "laps", "statusName"]
    available_cols = [col for col in display_cols if col in filtered.columns]
    st.dataframe(filtered[available_cols].head(50), use_container_width=True)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main dashboard application."""
    
    # Sidebar navigation
    st.sidebar.title("🏁 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Dashboard",
        [
            "🏠 Home",
            "🏎️ Classic ML & Race Data",
            "⚡ Spark MLlib Model",
            "🧠 TensorFlow Model",
            "📦 Feature Store",
            "🔄 Model Comparison"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Unified F1 ML Dashboard**\n\n"
        "This dashboard combines:\n"
        "- Classic ML models\n"
        "- Big Data Spark MLlib\n"
        "- TensorFlow Deep Learning\n"
        "- Feature Engineering Pipeline"
    )
    
    # Main content area
    if page == "🏠 Home":
        st.title("🏁 Formula 1 ML Dashboard - Unified View")
        st.markdown("""
        Welcome to the unified Formula 1 Machine Learning Dashboard! This dashboard provides
        a comprehensive view of all ML pipelines and models in the project.
        
        ## 📊 Available Sections
        
        ### 🏎️ Classic ML & Race Data
        - View race results and statistics
        - Explore traditional ML model performance
        - Analyze driver and constructor performance
        
        ### ⚡ Spark MLlib Model
        - Random Forest classifier metrics
        - Feature importance analysis
        - Big data pipeline artifacts
        
        ### 🧠 TensorFlow Model
        - Neural network performance
        - Training history visualization
        - Deep learning metrics
        
        ### 📦 Feature Store
        - Engineered features from Spark
        - Feature distributions
        - Data quality metrics
        
        ### 🔄 Model Comparison
        - Side-by-side model performance
        - ROC-AUC comparison
        - Best model identification
        """)
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📈 Quick Stats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Big Data Pipeline Status",
                _artefact_status(FEATURE_CSV_DIR)
            )
        
        with col2:
            spark_metrics = _load_json(SPARK_METRICS_PATH)
            if spark_metrics:
                st.metric(
                    "Spark MLlib ROC-AUC",
                    f"{spark_metrics.get('roc_auc', 0):.4f}"
                )
            else:
                st.metric("Spark MLlib ROC-AUC", "N/A")
        
        with col3:
            tf_metrics = _load_json(TF_METRICS_PATH)
            if tf_metrics:
                st.metric(
                    "TensorFlow ROC-AUC",
                    f"{tf_metrics.get('roc_auc', 0):.4f}"
                )
            else:
                st.metric("TensorFlow ROC-AUC", "N/A")
    
    elif page == "🏎️ Classic ML & Race Data":
        render_classic_overview()
    
    elif page == "⚡ Spark MLlib Model":
        render_bigdata_overview()
        st.markdown("---")
        render_spark_mllib_metrics()
    
    elif page == "🧠 TensorFlow Model":
        render_bigdata_overview()
        st.markdown("---")
        render_tensorflow_metrics()
    
    elif page == "📦 Feature Store":
        render_feature_store()
    
    elif page == "🔄 Model Comparison":
        render_model_comparison()


# Always run main() - compatible with both direct execution and imports
main()
