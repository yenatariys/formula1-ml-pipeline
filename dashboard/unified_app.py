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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
import numpy as np
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


@st.cache_data(show_spinner=False)
def load_csv_data(filename: str) -> pd.DataFrame:
    """Load CSV file from data directory."""
    csv_path = BASE_DIR / "data" / filename
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_lap_times() -> pd.DataFrame:
    """Load and enrich lap times data."""
    lap_times = load_csv_data("lap_times.csv")
    if lap_times.empty:
        return pd.DataFrame()
    
    # Load additional data for enrichment
    drivers = load_csv_data("drivers.csv")
    races = load_csv_data("races.csv")
    
    # Merge with driver and race info
    if not drivers.empty:
        lap_times = lap_times.merge(
            drivers[['driverId', 'surname', 'forename']], 
            on='driverId', 
            how='left'
        )
        lap_times['driver_name'] = lap_times['forename'] + ' ' + lap_times['surname']
    
    if not races.empty:
        lap_times = lap_times.merge(
            races[['raceId', 'year', 'name', 'round']], 
            on='raceId', 
            how='left'
        )
        lap_times.rename(columns={'name': 'race_name'}, inplace=True)
    
    return lap_times


@st.cache_data(show_spinner=False)
def load_pit_stops() -> pd.DataFrame:
    """Load and enrich pit stop data."""
    pit_stops = load_csv_data("pit_stops.csv")
    if pit_stops.empty:
        return pd.DataFrame()
    
    # Load additional data for enrichment
    drivers = load_csv_data("drivers.csv")
    races = load_csv_data("races.csv")
    constructors = load_csv_data("constructors.csv")
    results = load_csv_data("results.csv")
    
    # Merge with driver info
    if not drivers.empty:
        pit_stops = pit_stops.merge(
            drivers[['driverId', 'surname', 'forename']], 
            on='driverId', 
            how='left'
        )
        pit_stops['driver_name'] = pit_stops['forename'] + ' ' + pit_stops['surname']
    
    # Merge with race info
    if not races.empty:
        pit_stops = pit_stops.merge(
            races[['raceId', 'year', 'name', 'round']], 
            on='raceId', 
            how='left'
        )
        # Rename race name column first
        pit_stops.rename(columns={'name': 'race_name'}, inplace=True)
    
    # Get constructor info from results
    if not results.empty and not constructors.empty:
        driver_constructors = results[['raceId', 'driverId', 'constructorId']].drop_duplicates()
        pit_stops = pit_stops.merge(driver_constructors, on=['raceId', 'driverId'], how='left')
        pit_stops = pit_stops.merge(
            constructors[['constructorId', 'name']], 
            on='constructorId', 
            how='left'
        )
        # Rename constructor name column
        pit_stops.rename(columns={'name': 'constructor_name'}, inplace=True)
    
    return pit_stops


@st.cache_data(ttl=60)
def load_predictions():
    """Load ML predictions from database."""
    if not DB_AVAILABLE:
        return pd.DataFrame()
    try:
        return pd.read_sql("SELECT * FROM f1_predictions", engine)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_rf_predictions():
    """Load Random Forest predictions from database."""
    if not DB_AVAILABLE:
        return pd.DataFrame()
    try:
        return pd.read_sql("SELECT * FROM f1_predictions_rf", engine)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_xgb_predictions():
    """Load XGBoost predictions from database."""
    if not DB_AVAILABLE:
        return pd.DataFrame()
    try:
        return pd.read_sql("SELECT * FROM f1_predictions_xgb", engine)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_model_comparison():
    """Load model comparison data from database."""
    if not DB_AVAILABLE:
        return pd.DataFrame()
    try:
        return pd.read_sql("SELECT * FROM f1_model_comparison", engine)
    except Exception:
        return pd.DataFrame()


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
# CSV-Based Analysis Components
# ============================================================================

def render_lap_time_analysis():
    """Render lap time analysis from CSV data."""
    st.header("⏱️ Lap Time Analysis")
    
    lap_times = load_lap_times()
    
    if lap_times.empty:
        st.warning("Lap times data not available.")
        return
    
    st.write(f"📊 Loaded {len(lap_times):,} lap time records")
    
    # Year filter
    available_years = sorted(lap_times['year'].dropna().unique(), reverse=True)
    year = st.selectbox("Select Season", available_years, key="lap_year")
    
    year_data = lap_times[lap_times['year'] == year]
    
    # Race filter
    available_races = year_data[['round', 'race_name']].drop_duplicates().sort_values('round')
    race_names = [f"Round {row['round']}: {row['race_name']}" for _, row in available_races.iterrows()]
    
    if not race_names:
        st.info(f"No race data available for {year}")
        return
    
    selected_race_display = st.selectbox("Select Race", race_names, key="lap_race")
    selected_round = int(selected_race_display.split(':')[0].replace('Round ', ''))
    
    race_data = year_data[year_data['round'] == selected_round]
    
    if race_data.empty:
        st.info("No lap time data for selected race")
        return
    
    st.subheader(f"📈 {selected_race_display}")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Laps Recorded", f"{len(race_data):,}")
    
    with col2:
        fastest_lap_ms = race_data['milliseconds'].min()
        fastest_lap_sec = fastest_lap_ms / 1000
        st.metric("Fastest Lap", f"{fastest_lap_sec:.3f}s")
    
    with col3:
        unique_drivers = race_data['driver_name'].nunique()
        st.metric("Drivers", unique_drivers)
    
    with col4:
        max_lap = race_data['lap'].max()
        st.metric("Race Distance", f"{max_lap} laps")
    
    # Lap time progression
    st.subheader("🏁 Lap Time Progression")
    
    # Get top 10 drivers by average lap time
    avg_lap_times = race_data.groupby('driver_name')['milliseconds'].mean().nsmallest(10)
    top_drivers = avg_lap_times.index.tolist()
    
    selected_drivers = st.multiselect(
        "Select Drivers to Compare",
        options=sorted(race_data['driver_name'].dropna().unique()),
        default=top_drivers[:5] if len(top_drivers) >= 5 else top_drivers,
        key="lap_drivers"
    )
    
    if selected_drivers:
        driver_data = race_data[race_data['driver_name'].isin(selected_drivers)]
        driver_data['lap_time_seconds'] = driver_data['milliseconds'] / 1000
        
        fig_progression = px.line(
            driver_data,
            x='lap',
            y='lap_time_seconds',
            color='driver_name',
            title='Lap Time Progression',
            labels={'lap': 'Lap Number', 'lap_time_seconds': 'Lap Time (seconds)', 'driver_name': 'Driver'},
            markers=True
        )
        fig_progression.update_layout(hovermode='x unified')
        st.plotly_chart(fig_progression, use_container_width=True)
    
    # Driver performance comparison
    st.subheader("📊 Driver Performance Summary")
    
    driver_stats = race_data.groupby('driver_name').agg({
        'milliseconds': ['mean', 'min', 'max', 'std'],
        'lap': 'count'
    }).reset_index()
    
    driver_stats.columns = ['Driver', 'Avg Lap Time (ms)', 'Fastest Lap (ms)', 'Slowest Lap (ms)', 'Std Dev', 'Laps Completed']
    driver_stats = driver_stats.sort_values('Avg Lap Time (ms)')
    
    # Convert to seconds for better readability
    driver_stats['Avg Lap Time (s)'] = driver_stats['Avg Lap Time (ms)'] / 1000
    driver_stats['Fastest Lap (s)'] = driver_stats['Fastest Lap (ms)'] / 1000
    
    st.dataframe(
        driver_stats[['Driver', 'Laps Completed', 'Avg Lap Time (s)', 'Fastest Lap (s)', 'Std Dev']].head(15),
        use_container_width=True
    )
    
    # Fastest lap distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Fastest laps by driver
        fastest_by_driver = race_data.groupby('driver_name')['milliseconds'].min().nsmallest(10).reset_index()
        fastest_by_driver['seconds'] = fastest_by_driver['milliseconds'] / 1000
        
        fig_fastest = px.bar(
            fastest_by_driver,
            x='driver_name',
            y='seconds',
            title='Top 10 Fastest Laps',
            labels={'driver_name': 'Driver', 'seconds': 'Lap Time (seconds)'},
            color='seconds',
            color_continuous_scale='Viridis_r'
        )
        st.plotly_chart(fig_fastest, use_container_width=True)
    
    with col2:
        # Lap time consistency (std dev)
        consistency = race_data.groupby('driver_name')['milliseconds'].std().nsmallest(10).reset_index()
        consistency.columns = ['Driver', 'Std Dev (ms)']
        
        fig_consistency = px.bar(
            consistency,
            x='Driver',
            y='Std Dev (ms)',
            title='Top 10 Most Consistent Drivers',
            labels={'Driver': 'Driver', 'Std Dev (ms)': 'Standard Deviation (ms)'},
            color='Std Dev (ms)',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_consistency, use_container_width=True)
    
    # ========================================================================
    # ML ANALYSIS SECTION
    # ========================================================================
    st.markdown("---")
    st.header("🤖 Machine Learning Analysis")
    
    ml_tab1, ml_tab2, ml_tab3 = st.tabs([
        "🎯 Lap Time Prediction", 
        "👥 Driver Clustering", 
        "⚠️ Anomaly Detection"
    ])
    
    with ml_tab1:
        st.subheader("Lap Time Prediction Model")
        
        # Prepare data for prediction
        ml_data = year_data[['driver_name', 'race_name', 'lap', 'position', 'milliseconds']].copy()
        ml_data = ml_data.dropna()
        
        if len(ml_data) > 100:
            # Create features
            ml_data['driver_encoded'] = pd.factorize(ml_data['driver_name'])[0]
            ml_data['race_encoded'] = pd.factorize(ml_data['race_name'])[0]
            ml_data['lap_time_seconds'] = ml_data['milliseconds'] / 1000
            
            X = ml_data[['driver_encoded', 'race_encoded', 'lap', 'position']]
            y = ml_data['lap_time_seconds']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            with st.spinner("Training Random Forest model..."):
                rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Absolute Error", f"{mae:.3f}s")
                st.metric("R² Score", f"{r2:.4f}")
            
            with col2:
                # Feature importance
                feature_names = ['Driver', 'Race', 'Lap Number', 'Position']
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Prediction vs Actual
            comparison_df = pd.DataFrame({
                'Actual': y_test.values[:100],
                'Predicted': y_pred[:100]
            })
            
            fig_pred = px.scatter(
                comparison_df,
                x='Actual',
                y='Predicted',
                title='Predicted vs Actual Lap Times (Sample)',
                labels={'Actual': 'Actual Lap Time (s)', 'Predicted': 'Predicted Lap Time (s)'}
            )
            fig_pred.add_trace(go.Scatter(
                x=[comparison_df['Actual'].min(), comparison_df['Actual'].max()],
                y=[comparison_df['Actual'].min(), comparison_df['Actual'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.success(f"✅ Model trained on {len(X_train):,} laps, tested on {len(X_test):,} laps")
        else:
            st.warning("Insufficient data for ML modeling (need >100 records)")
    
    with ml_tab2:
        st.subheader("Driver Performance Clustering")
        
        # Aggregate driver statistics
        driver_agg = year_data.groupby('driver_name').agg({
            'milliseconds': ['mean', 'std', 'min'],
            'lap': 'count',
            'position': 'mean'
        }).reset_index()
        
        driver_agg.columns = ['driver_name', 'avg_time', 'std_time', 'best_time', 'total_laps', 'avg_position']
        driver_agg = driver_agg[driver_agg['total_laps'] >= 10]  # Filter drivers with enough laps
        
        if len(driver_agg) >= 5:
            # Prepare features for clustering
            features = driver_agg[['avg_time', 'std_time', 'avg_position']].copy()
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Determine optimal clusters (2-5)
            n_clusters = min(4, len(driver_agg) // 3)
            n_clusters = max(2, n_clusters)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            driver_agg['cluster'] = kmeans.fit_predict(features_scaled)
            
            silhouette = silhouette_score(features_scaled, driver_agg['cluster'])
            
            st.metric("Silhouette Score", f"{silhouette:.3f}")
            st.info(f"Drivers grouped into {n_clusters} performance clusters")
            
            # Visualize clusters
            fig_cluster = px.scatter(
                driver_agg,
                x='avg_time',
                y='std_time',
                color='cluster',
                size='total_laps',
                hover_data=['driver_name', 'avg_position'],
                title='Driver Clustering: Speed vs Consistency',
                labels={
                    'avg_time': 'Average Lap Time (ms)',
                    'std_time': 'Lap Time Std Dev (ms)',
                    'cluster': 'Cluster'
                },
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Cluster characteristics
            st.subheader("Cluster Characteristics")
            cluster_summary = driver_agg.groupby('cluster').agg({
                'avg_time': 'mean',
                'std_time': 'mean',
                'avg_position': 'mean',
                'driver_name': 'count'
            }).reset_index()
            cluster_summary.columns = ['Cluster', 'Avg Lap Time (ms)', 'Consistency (ms)', 'Avg Position', 'Drivers']
            cluster_summary['Cluster'] = cluster_summary['Cluster'].astype(int)
            
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Show drivers in each cluster
            for cluster_id in sorted(driver_agg['cluster'].unique()):
                with st.expander(f"Cluster {cluster_id} Drivers"):
                    cluster_drivers = driver_agg[driver_agg['cluster'] == cluster_id][
                        ['driver_name', 'avg_time', 'std_time', 'avg_position', 'total_laps']
                    ].sort_values('avg_time')
                    cluster_drivers.columns = ['Driver', 'Avg Time (ms)', 'Std Dev (ms)', 'Avg Position', 'Total Laps']
                    st.dataframe(cluster_drivers, use_container_width=True)
        else:
            st.warning("Insufficient drivers for clustering analysis")
    
    with ml_tab3:
        st.subheader("Anomaly Detection - Unusual Lap Times")
        
        if len(year_data) > 50:
            # Calculate Z-scores for lap times
            year_data_copy = year_data.copy()
            year_data_copy['lap_time_seconds'] = year_data_copy['milliseconds'] / 1000
            
            # Group by race and calculate z-scores
            year_data_copy['z_score'] = year_data_copy.groupby('race_name')['lap_time_seconds'].transform(
                lambda x: np.abs((x - x.mean()) / x.std())
            )
            
            # Find anomalies (z-score > 3)
            anomalies = year_data_copy[year_data_copy['z_score'] > 3].copy()
            anomalies = anomalies.sort_values('z_score', ascending=False)
            
            st.metric("Anomalous Laps Detected", len(anomalies))
            
            if len(anomalies) > 0:
                # Show top anomalies
                st.subheader("Top Anomalies")
                top_anomalies = anomalies.head(20)[
                    ['driver_name', 'race_name', 'lap', 'lap_time_seconds', 'position', 'z_score']
                ].copy()
                top_anomalies.columns = ['Driver', 'Race', 'Lap', 'Lap Time (s)', 'Position', 'Z-Score']
                st.dataframe(top_anomalies, use_container_width=True)
                
                # Visualize anomalies
                sample_race = anomalies['race_name'].value_counts().index[0]
                race_with_anomalies = year_data_copy[year_data_copy['race_name'] == sample_race]
                
                fig_anomaly = px.scatter(
                    race_with_anomalies,
                    x='lap',
                    y='lap_time_seconds',
                    color=race_with_anomalies['z_score'] > 3,
                    title=f'Lap Time Anomalies - {sample_race}',
                    labels={'lap': 'Lap Number', 'lap_time_seconds': 'Lap Time (s)', 'color': 'Is Anomaly'},
                    hover_data=['driver_name', 'position'],
                    color_discrete_map={True: 'red', False: 'blue'}
                )
                st.plotly_chart(fig_anomaly, use_container_width=True)
                
                st.info("🔍 Anomalies may indicate: Safety car periods, pit stops, crashes, weather changes, or data errors")
            else:
                st.success("No significant anomalies detected in this season")
        else:
            st.warning("Insufficient data for anomaly detection")


def render_pit_stop_analysis():
    """Render pit stop strategy analysis from CSV data."""
    st.header("⛽ Pit Stop Strategy Analysis")
    
    pit_stops = load_pit_stops()
    
    if pit_stops.empty:
        st.warning("Pit stop data not available.")
        return
    
    st.write(f"📊 Loaded {len(pit_stops):,} pit stop records")
    
    # Year filter
    available_years = sorted(pit_stops['year'].dropna().unique(), reverse=True)
    year = st.selectbox("Select Season", available_years, key="pit_year")
    
    year_data = pit_stops[pit_stops['year'] == year]
    
    # Overall season statistics
    st.subheader(f"🏁 {year} Season Pit Stop Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_stops = len(year_data)
        st.metric("Total Pit Stops", f"{total_stops:,}")
    
    with col2:
        avg_duration = year_data['milliseconds'].mean() / 1000
        st.metric("Avg Stop Duration", f"{avg_duration:.3f}s")
    
    with col3:
        fastest_stop = year_data['milliseconds'].min() / 1000
        st.metric("Fastest Stop", f"{fastest_stop:.3f}s")
    
    with col4:
        slowest_stop = year_data['milliseconds'].max() / 1000
        st.metric("Slowest Stop", f"{slowest_stop:.3f}s")
    
    # Team performance
    st.subheader("🏎️ Pit Crew Performance by Team")
    
    if 'constructor_name' in year_data.columns:
        team_stats = year_data.groupby('constructor_name').agg({
            'milliseconds': ['mean', 'min', 'count'],
            'stop': 'max'
        }).reset_index()
        
        team_stats.columns = ['Team', 'Avg Duration (ms)', 'Fastest Stop (ms)', 'Total Stops', 'Max Stops per Race']
        team_stats['Avg Duration (s)'] = team_stats['Avg Duration (ms)'] / 1000
        team_stats['Fastest Stop (s)'] = team_stats['Fastest Stop (ms)'] / 1000
        team_stats = team_stats.sort_values('Avg Duration (ms)')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_team_avg = px.bar(
                team_stats.head(10),
                x='Team',
                y='Avg Duration (s)',
                title='Top 10 Teams by Average Pit Stop Duration',
                color='Avg Duration (s)',
                color_continuous_scale='RdYlGn_r',
                labels={'Avg Duration (s)': 'Avg Duration (seconds)'}
            )
            st.plotly_chart(fig_team_avg, use_container_width=True)
        
        with col2:
            fig_team_count = px.bar(
                team_stats.sort_values('Total Stops', ascending=False).head(10),
                x='Team',
                y='Total Stops',
                title='Top 10 Teams by Total Pit Stops',
                color='Total Stops',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_team_count, use_container_width=True)
        
        st.dataframe(
            team_stats[['Team', 'Total Stops', 'Avg Duration (s)', 'Fastest Stop (s)']],
            use_container_width=True
        )
    
    # Race-specific analysis
    st.subheader("📍 Race-Specific Analysis")
    
    available_races = year_data[['round', 'race_name']].drop_duplicates().sort_values('round')
    race_names = [f"Round {row['round']}: {row['race_name']}" for _, row in available_races.iterrows()]
    
    if race_names:
        selected_race_display = st.selectbox("Select Race", race_names, key="pit_race")
        selected_round = int(selected_race_display.split(':')[0].replace('Round ', ''))
        
        race_data = year_data[year_data['round'] == selected_round]
        
        if not race_data.empty:
            st.write(f"**{selected_race_display}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Stops This Race", len(race_data))
                st.metric("Average Duration", f"{race_data['milliseconds'].mean() / 1000:.3f}s")
            
            with col2:
                st.metric("Fastest Stop", f"{race_data['milliseconds'].min() / 1000:.3f}s")
                fastest_team = race_data.loc[race_data['milliseconds'].idxmin(), 'constructor_name'] if 'constructor_name' in race_data.columns else "N/A"
                st.metric("Fastest Team", fastest_team)
            
            # Pit stop timeline
            race_data_sorted = race_data.sort_values('lap')
            race_data_sorted['duration_s'] = race_data_sorted['milliseconds'] / 1000
            
            fig_timeline = px.scatter(
                race_data_sorted,
                x='lap',
                y='duration_s',
                color='constructor_name' if 'constructor_name' in race_data_sorted.columns else 'driver_name',
                size='duration_s',
                title='Pit Stop Timeline',
                labels={'lap': 'Lap Number', 'duration_s': 'Duration (seconds)'},
                hover_data=['driver_name', 'stop']
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Driver pit stop details
            driver_pit_stats = race_data.groupby('driver_name').agg({
                'stop': 'max',
                'milliseconds': 'mean',
                'lap': lambda x: ', '.join(map(str, sorted(x)))
            }).reset_index()
            driver_pit_stats.columns = ['Driver', 'Number of Stops', 'Avg Duration (ms)', 'Pit Stop Laps']
            driver_pit_stats['Avg Duration (s)'] = driver_pit_stats['Avg Duration (ms)'] / 1000
            
            st.dataframe(
                driver_pit_stats[['Driver', 'Number of Stops', 'Avg Duration (s)', 'Pit Stop Laps']],
                use_container_width=True
            )
    
    # Historical trend
    st.subheader("📈 Pit Stop Evolution Over Season")
    
    race_avg = year_data.groupby('round').agg({
        'milliseconds': 'mean',
        'race_name': 'first'
    }).reset_index()
    race_avg['avg_duration_s'] = race_avg['milliseconds'] / 1000
    
    fig_evolution = px.line(
        race_avg,
        x='round',
        y='avg_duration_s',
        title=f'Average Pit Stop Duration Throughout {year} Season',
        labels={'round': 'Race Round', 'avg_duration_s': 'Avg Duration (seconds)'},
        markers=True,
        hover_data=['race_name']
    )
    st.plotly_chart(fig_evolution, use_container_width=True)
    
    # ========================================================================
    # ML ANALYSIS SECTION
    # ========================================================================
    st.markdown("---")
    st.header("🤖 Machine Learning Analysis")
    
    ml_tab1, ml_tab2, ml_tab3 = st.tabs([
        "🎯 Pit Stop Duration Prediction",
        "📊 Strategy Classification",
        "👥 Team Performance Clustering"
    ])
    
    with ml_tab1:
        st.subheader("Pit Stop Duration Prediction")
        
        # Prepare data
        ml_data = year_data[['driver_name', 'race_name', 'lap', 'stop', 'milliseconds']].copy()
        if 'constructor_name' in year_data.columns:
            ml_data['constructor_name'] = year_data['constructor_name']
        ml_data = ml_data.dropna()
        
        if len(ml_data) > 100:
            # Create features
            ml_data['driver_encoded'] = pd.factorize(ml_data['driver_name'])[0]
            ml_data['race_encoded'] = pd.factorize(ml_data['race_name'])[0]
            if 'constructor_name' in ml_data.columns:
                ml_data['team_encoded'] = pd.factorize(ml_data['constructor_name'])[0]
                feature_cols = ['driver_encoded', 'race_encoded', 'team_encoded', 'lap', 'stop']
            else:
                feature_cols = ['driver_encoded', 'race_encoded', 'lap', 'stop']
            
            ml_data['duration_seconds'] = ml_data['milliseconds'] / 1000
            
            X = ml_data[feature_cols]
            y = ml_data['duration_seconds']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            with st.spinner("Training prediction model..."):
                rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Absolute Error", f"{mae:.3f}s")
                st.metric("R² Score", f"{r2:.4f}")
                st.info(f"Trained on {len(X_train):,} pit stops")
            
            with col2:
                # Feature importance
                if 'constructor_name' in ml_data.columns:
                    feature_names = ['Driver', 'Race', 'Team', 'Lap', 'Stop Number']
                else:
                    feature_names = ['Driver', 'Race', 'Lap', 'Stop Number']
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance for Pit Stop Duration',
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Prediction accuracy visualization
            comparison_df = pd.DataFrame({
                'Actual': y_test.values[:100],
                'Predicted': y_pred[:100],
                'Error': np.abs(y_test.values[:100] - y_pred[:100])
            })
            
            fig_pred = px.scatter(
                comparison_df,
                x='Actual',
                y='Predicted',
                color='Error',
                title='Predicted vs Actual Pit Stop Duration',
                labels={'Actual': 'Actual Duration (s)', 'Predicted': 'Predicted Duration (s)', 'Error': 'Abs Error (s)'},
                color_continuous_scale='RdYlGn_r'
            )
            fig_pred.add_trace(go.Scatter(
                x=[comparison_df['Actual'].min(), comparison_df['Actual'].max()],
                y=[comparison_df['Actual'].min(), comparison_df['Actual'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='blue', dash='dash')
            ))
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.warning("Insufficient data for ML modeling (need >100 pit stops)")
    
    with ml_tab2:
        st.subheader("Pit Stop Strategy Classification")
        
        # Aggregate by driver and race to determine strategy
        strategy_data = year_data.groupby(['driver_name', 'race_name']).agg({
            'stop': 'max',
            'milliseconds': 'sum',
            'lap': lambda x: list(x)
        }).reset_index()
        
        strategy_data.columns = ['driver_name', 'race_name', 'total_stops', 'total_duration', 'pit_laps']
        
        if len(strategy_data) > 50:
            # Classify strategy based on number of stops
            def classify_strategy(stops):
                if stops == 0:
                    return 'No Stop'
                elif stops == 1:
                    return '1-Stop'
                elif stops == 2:
                    return '2-Stop'
                elif stops == 3:
                    return '3-Stop'
                else:
                    return '4+ Stop'
            
            strategy_data['strategy'] = strategy_data['total_stops'].apply(classify_strategy)
            
            # Strategy distribution
            strategy_counts = strategy_data['strategy'].value_counts().reset_index()
            strategy_counts.columns = ['Strategy', 'Count']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_strategy_dist = px.pie(
                    strategy_counts,
                    values='Count',
                    names='Strategy',
                    title='Pit Stop Strategy Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_strategy_dist, use_container_width=True)
            
            with col2:
                st.subheader("Strategy Statistics")
                for strategy in strategy_counts['Strategy']:
                    count = strategy_counts[strategy_counts['Strategy'] == strategy]['Count'].values[0]
                    percentage = (count / len(strategy_data)) * 100
                    st.metric(strategy, f"{count} ({percentage:.1f}%)")
            
            # Average pit timing by strategy
            st.subheader("Average Pit Stop Timing by Strategy")
            
            # Calculate average lap for first pit stop
            strategy_data['first_pit_lap'] = strategy_data['pit_laps'].apply(
                lambda x: sorted(x)[0] if len(x) > 0 else None
            )
            
            timing_analysis = strategy_data[strategy_data['first_pit_lap'].notna()].groupby('strategy').agg({
                'first_pit_lap': 'mean',
                'total_duration': 'mean',
                'driver_name': 'count'
            }).reset_index()
            
            timing_analysis.columns = ['Strategy', 'Avg First Pit Lap', 'Avg Total Duration (ms)', 'Sample Size']
            timing_analysis['Avg Total Duration (s)'] = timing_analysis['Avg Total Duration (ms)'] / 1000
            
            fig_timing = px.bar(
                timing_analysis,
                x='Strategy',
                y='Avg First Pit Lap',
                color='Avg Total Duration (s)',
                title='Average First Pit Stop Lap by Strategy',
                labels={'Avg First Pit Lap': 'Average Lap Number'},
                text='Avg First Pit Lap',
                color_continuous_scale='Viridis'
            )
            fig_timing.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig_timing, use_container_width=True)
            
            st.dataframe(timing_analysis, use_container_width=True)
            
            # Train classifier to predict strategy
            if 'constructor_name' in year_data.columns:
                # Add team information
                team_mapping = year_data[['driver_name', 'constructor_name']].drop_duplicates()
                strategy_data = strategy_data.merge(team_mapping, on='driver_name', how='left')
                
                ml_strategy = strategy_data.dropna()
                if len(ml_strategy) > 50:
                    ml_strategy['team_encoded'] = pd.factorize(ml_strategy['constructor_name'])[0]
                    ml_strategy['race_encoded'] = pd.factorize(ml_strategy['race_name'])[0]
                    
                    X_strat = ml_strategy[['team_encoded', 'race_encoded']]
                    y_strat = ml_strategy['strategy']
                    
                    # Filter out rare strategies for better classification
                    strategy_counts_ml = y_strat.value_counts()
                    valid_strategies = strategy_counts_ml[strategy_counts_ml >= 10].index
                    mask = y_strat.isin(valid_strategies)
                    
                    X_strat = X_strat[mask]
                    y_strat = y_strat[mask]
                    
                    if len(X_strat) > 50:
                        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                            X_strat, y_strat, test_size=0.2, random_state=42
                        )
                        
                        with st.spinner("Training strategy classifier..."):
                            rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
                            rf_classifier.fit(X_train_s, y_train_s)
                            y_pred_s = rf_classifier.predict(X_test_s)
                            
                            accuracy = accuracy_score(y_test_s, y_pred_s)
                        
                        st.subheader("🎯 Strategy Prediction Model")
                        st.metric("Classification Accuracy", f"{accuracy:.2%}")
                        
                        # Classification report
                        with st.expander("View Detailed Classification Report"):
                            report = classification_report(y_test_s, y_pred_s, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df, use_container_width=True)
        else:
            st.warning("Insufficient data for strategy classification")
    
    with ml_tab3:
        st.subheader("Team Pit Crew Performance Clustering")
        
        if 'constructor_name' in year_data.columns:
            # Aggregate team performance
            team_perf = year_data.groupby('constructor_name').agg({
                'milliseconds': ['mean', 'std', 'min', 'count'],
                'stop': 'max'
            }).reset_index()
            
            team_perf.columns = ['team', 'avg_duration', 'std_duration', 'best_time', 'total_stops', 'max_stops']
            team_perf = team_perf[team_perf['total_stops'] >= 20]  # Filter teams with enough data
            
            if len(team_perf) >= 3:
                # Prepare features
                features = team_perf[['avg_duration', 'std_duration', 'best_time']].copy()
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                # Determine optimal clusters
                n_clusters = min(3, len(team_perf) // 2)
                n_clusters = max(2, n_clusters)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                team_perf['cluster'] = kmeans.fit_predict(features_scaled)
                
                silhouette = silhouette_score(features_scaled, team_perf['cluster'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Silhouette Score", f"{silhouette:.3f}")
                    st.metric("Number of Clusters", n_clusters)
                
                with col2:
                    st.metric("Teams Analyzed", len(team_perf))
                    st.metric("Total Pit Stops", int(team_perf['total_stops'].sum()))
                
                # Visualize clusters
                fig_team_cluster = px.scatter(
                    team_perf,
                    x='avg_duration',
                    y='std_duration',
                    color='cluster',
                    size='total_stops',
                    hover_data=['team', 'best_time'],
                    title='Team Clustering: Average Duration vs Consistency',
                    labels={
                        'avg_duration': 'Average Duration (ms)',
                        'std_duration': 'Std Dev (ms)',
                        'cluster': 'Performance Cluster'
                    },
                    color_continuous_scale='Plasma'
                )
                st.plotly_chart(fig_team_cluster, use_container_width=True)
                
                # Cluster interpretation
                st.subheader("Cluster Analysis")
                cluster_summary = team_perf.groupby('cluster').agg({
                    'avg_duration': 'mean',
                    'std_duration': 'mean',
                    'best_time': 'mean',
                    'team': 'count'
                }).reset_index()
                cluster_summary.columns = ['Cluster', 'Avg Duration (ms)', 'Consistency (ms)', 'Best Time (ms)', 'Teams']
                
                # Add interpretation
                def interpret_cluster(row):
                    if row['Avg Duration (ms)'] < team_perf['avg_duration'].median():
                        speed = "Fast"
                    else:
                        speed = "Slow"
                    
                    if row['Consistency (ms)'] < team_perf['std_duration'].median():
                        consistency = "Consistent"
                    else:
                        consistency = "Variable"
                    
                    return f"{speed} & {consistency}"
                
                cluster_summary['Interpretation'] = cluster_summary.apply(interpret_cluster, axis=1)
                
                st.dataframe(cluster_summary, use_container_width=True)
                
                # Show teams in each cluster
                for cluster_id in sorted(team_perf['cluster'].unique()):
                    with st.expander(f"Cluster {cluster_id}: {cluster_summary[cluster_summary['Cluster']==cluster_id]['Interpretation'].values[0]}"):
                        cluster_teams = team_perf[team_perf['cluster'] == cluster_id][
                            ['team', 'avg_duration', 'std_duration', 'best_time', 'total_stops']
                        ].sort_values('avg_duration')
                        cluster_teams.columns = ['Team', 'Avg Duration (ms)', 'Std Dev (ms)', 'Best Time (ms)', 'Total Stops']
                        cluster_teams['Avg Duration (s)'] = cluster_teams['Avg Duration (ms)'] / 1000
                        cluster_teams['Best Time (s)'] = cluster_teams['Best Time (ms)'] / 1000
                        st.dataframe(cluster_teams[['Team', 'Avg Duration (s)', 'Std Dev (ms)', 'Best Time (s)', 'Total Stops']], use_container_width=True)
            else:
                st.warning("Insufficient teams for clustering (need at least 3 teams with 20+ pit stops)")
        else:
            st.warning("Team/constructor information not available in dataset")


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
    filtered = df[df["year"] == year]
    
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
    st.dataframe(filtered[available_cols].sort_values(["round", "position"]), use_container_width=True)
    
    st.divider()
    st.header("📊 Advanced Analytics")
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Constructor Championship",
        "📈 Performance Trends",
        "⚔️ Head-to-Head",
        "🎯 Driver Statistics",
        "🗓️ Season Timeline"
    ])
    
    with tab1:
        st.subheader(f"Constructor Championship - {year}")
        
        # Get podium finishes (top 3)
        if len(filtered) > 0:
            podium_counts = filtered[filtered["position"].isin([1, 2, 3])].groupby("surname").size().reset_index()
            podium_counts.columns = ["Driver", "Podium Finishes"]
            podium_counts = podium_counts.sort_values("Podium Finishes", ascending=False).head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_podium = px.bar(
                    podium_counts,
                    x="Driver",
                    y="Podium Finishes",
                    title=f"Top 10 Podium Finishers ({year})",
                    color="Podium Finishes",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_podium, use_container_width=True)
            
            with col2:
                # Average finishing position by driver
                avg_pos = filtered.groupby("surname")["position"].mean().reset_index()
                avg_pos.columns = ["Driver", "Avg Position"]
                avg_pos = avg_pos.sort_values("Avg Position").head(10)
                
                fig_avg = px.bar(
                    avg_pos,
                    x="Driver",
                    y="Avg Position",
                    title=f"Top 10 Average Finishing Position ({year})",
                    color="Avg Position",
                    color_continuous_scale="Reds_r"
                )
                fig_avg.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_avg, use_container_width=True)
    
    with tab2:
        st.subheader(f"Performance Trends - {year}")
        
        # Points progression throughout the season
        if len(filtered) > 0:
            # Get top 10 drivers by total points
            top_drivers = filtered.groupby("surname")["points"].sum().nlargest(10).index.tolist()
            
            # Filter for top drivers only
            trend_data = filtered[filtered["surname"].isin(top_drivers)].copy()
            trend_data = trend_data.sort_values(["surname", "round"])
            
            # Calculate cumulative points
            trend_data["cumulative_points"] = trend_data.groupby("surname")["points"].cumsum()
            
            fig_trend = px.line(
                trend_data,
                x="round",
                y="cumulative_points",
                color="surname",
                title=f"Championship Points Progression ({year})",
                labels={"round": "Race Round", "cumulative_points": "Cumulative Points", "surname": "Driver"},
                markers=True
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Position trends
            st.subheader("Finishing Position Trends")
            selected_drivers = st.multiselect(
                "Select drivers to compare",
                options=top_drivers,
                default=top_drivers[:3] if len(top_drivers) >= 3 else top_drivers
            )
            
            if selected_drivers:
                position_data = filtered[filtered["surname"].isin(selected_drivers)].copy()
                position_data = position_data.sort_values(["surname", "round"])
                
                fig_pos = px.line(
                    position_data,
                    x="round",
                    y="position",
                    color="surname",
                    title="Finishing Positions Throughout Season",
                    labels={"round": "Race Round", "position": "Finishing Position", "surname": "Driver"},
                    markers=True
                )
                fig_pos.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_pos, use_container_width=True)
    
    with tab3:
        st.subheader(f"Head-to-Head Comparison - {year}")
        
        drivers_list = sorted(filtered["surname"].unique())
        
        col1, col2 = st.columns(2)
        with col1:
            driver1 = st.selectbox("Select Driver 1", drivers_list, key="h2h_driver1")
        with col2:
            driver2 = st.selectbox("Select Driver 2", drivers_list, index=min(1, len(drivers_list)-1), key="h2h_driver2")
        
        if driver1 and driver2:
            d1_data = filtered[filtered["surname"] == driver1]
            d2_data = filtered[filtered["surname"] == driver2]
            
            # Create comparison metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"{driver1} - Total Points", f"{d1_data['points'].sum():.0f}")
                st.metric(f"{driver2} - Total Points", f"{d2_data['points'].sum():.0f}")
            
            with col2:
                st.metric(f"{driver1} - Wins", len(d1_data[d1_data["position"] == 1]))
                st.metric(f"{driver2} - Wins", len(d2_data[d2_data["position"] == 1]))
            
            with col3:
                st.metric(f"{driver1} - Podiums", len(d1_data[d1_data["position"] <= 3]))
                st.metric(f"{driver2} - Podiums", len(d2_data[d2_data["position"] <= 3]))
            
            with col4:
                st.metric(f"{driver1} - Avg Pos", f"{d1_data['position'].mean():.1f}")
                st.metric(f"{driver2} - Avg Pos", f"{d2_data['position'].mean():.1f}")
            
            # Points comparison chart
            comparison_df_local = pd.DataFrame({
                "Driver": [driver1, driver2],
                "Total Points": [d1_data['points'].sum(), d2_data['points'].sum()],
                "Wins": [len(d1_data[d1_data["position"] == 1]), len(d2_data[d2_data["position"] == 1])],
                "Podiums": [len(d1_data[d1_data["position"] <= 3]), len(d2_data[d2_data["position"] <= 3])]
            })
            
            fig_comparison = px.bar(
                comparison_df_local.melt(id_vars="Driver", var_name="Metric", value_name="Count"),
                x="Metric",
                y="Count",
                color="Driver",
                barmode="group",
                title=f"{driver1} vs {driver2} - Season Comparison"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tab4:
        st.subheader(f"Driver Statistics - {year}")
        
        # Calculate various statistics
        stats_data = []
        for driver in filtered["surname"].unique():
            driver_df = filtered[filtered["surname"] == driver]
            
            stats_data.append({
                "Driver": driver,
                "Races": len(driver_df),
                "Wins": len(driver_df[driver_df["position"] == 1]),
                "Podiums": len(driver_df[driver_df["position"] <= 3]),
                "Top 5": len(driver_df[driver_df["position"] <= 5]),
                "Top 10": len(driver_df[driver_df["position"] <= 10]),
                "Total Points": driver_df["points"].sum(),
                "Avg Position": driver_df["position"].mean(),
                "Best Finish": driver_df["position"].min(),
                "Worst Finish": driver_df["position"].max()
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df.sort_values("Total Points", ascending=False)
        
        # Add formatting
        st.dataframe(
            stats_df.style.background_gradient(subset=["Total Points"], cmap="Greens")
                         .background_gradient(subset=["Wins"], cmap="Blues")
                         .format({
                             "Total Points": "{:.0f}",
                             "Avg Position": "{:.2f}",
                         }),
            use_container_width=True
        )
        
        # Win rate calculation
        stats_df["Win Rate %"] = (stats_df["Wins"] / stats_df["Races"] * 100).round(1)
        stats_df["Podium Rate %"] = (stats_df["Podiums"] / stats_df["Races"] * 100).round(1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_win_rate = stats_df.nlargest(10, "Win Rate %")
            fig_win_rate = px.bar(
                top_win_rate,
                x="Driver",
                y="Win Rate %",
                title="Top 10 Win Rate %",
                color="Win Rate %",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_win_rate, use_container_width=True)
        
        with col2:
            top_podium_rate = stats_df.nlargest(10, "Podium Rate %")
            fig_podium_rate = px.bar(
                top_podium_rate,
                x="Driver",
                y="Podium Rate %",
                title="Top 10 Podium Rate %",
                color="Podium Rate %",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_podium_rate, use_container_width=True)
    
    with tab5:
        st.subheader(f"Season Timeline - {year}")
        
        # Race winners timeline
        winners_timeline = filtered[filtered["position"] == 1].copy()
        winners_timeline = winners_timeline.sort_values("round")
        
        if len(winners_timeline) > 0:
            fig_timeline = px.scatter(
                winners_timeline,
                x="round",
                y="surname",
                size="points",
                color="surname",
                title=f"Race Winners by Round ({year})",
                labels={"round": "Race Round", "surname": "Winner"},
                hover_data=["name", "points"] if "name" in winners_timeline.columns else None
            )
            fig_timeline.update_traces(marker=dict(size=20, line=dict(width=2, color='DarkSlateGrey')))
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Show race-by-race results
            st.subheader("Race-by-Race Winners")
            winners_cols = ["round", "name", "surname", "points"] if "name" in winners_timeline.columns else ["round", "surname", "points"]
            winners_table = winners_timeline[winners_cols].copy()
            st.dataframe(winners_table, use_container_width=True)
    
    # Load ML prediction data
    st.divider()
    df_preds = load_predictions()
    rf_preds_df = load_rf_predictions()
    xgb_preds_df = load_xgb_predictions()
    
    # Dashboard Predictions - Show Both Models
    st.subheader("🤖 Race Win Predictions - Model Comparison")
    
    if not rf_preds_df.empty and not xgb_preds_df.empty:
        # Filter predictions by season
        rf_filtered = rf_preds_df[rf_preds_df["year"] == year].copy()
        xgb_filtered = xgb_preds_df[xgb_preds_df["year"] == year].copy()
        
        if len(rf_filtered) > 0 and len(xgb_filtered) > 0:
            # Show metrics for both models side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌲 Random Forest")
                rf_y_test = rf_filtered['actual'].values
                rf_preds = rf_filtered['predicted'].values
                rf_correct = (rf_y_test == rf_preds).sum()
                rf_total = len(rf_y_test)
                rf_acc = accuracy_score(rf_y_test, rf_preds)
                
                st.metric("Total Predictions", rf_total)
                st.metric("Correct Predictions", rf_correct)
                st.metric("Accuracy", f"{rf_acc:.1%}")
                
                # Confusion matrix
                rf_cm = confusion_matrix(rf_y_test, rf_preds)
                fig_rf = px.imshow(
                    rf_cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['No Win', 'Win'],
                    y=['No Win', 'Win'],
                    title=f"Random Forest Confusion Matrix ({year})",
                    color_continuous_scale='Greens',
                    text_auto=True
                )
                st.plotly_chart(fig_rf, use_container_width=True)
            
            with col2:
                st.markdown("### 🚀 XGBoost")
                xgb_y_test = xgb_filtered['actual'].values
                xgb_preds = xgb_filtered['predicted'].values
                xgb_correct = (xgb_y_test == xgb_preds).sum()
                xgb_total = len(xgb_y_test)
                xgb_acc = accuracy_score(xgb_y_test, xgb_preds)
                
                st.metric("Total Predictions", xgb_total)
                st.metric("Correct Predictions", xgb_correct)
                st.metric("Accuracy", f"{xgb_acc:.1%}")
                
                # Confusion matrix
                xgb_cm = confusion_matrix(xgb_y_test, xgb_preds)
                fig_xgb = px.imshow(
                    xgb_cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['No Win', 'Win'],
                    y=['No Win', 'Win'],
                    title=f"XGBoost Confusion Matrix ({year})",
                    color_continuous_scale='Reds',
                    text_auto=True
                )
                st.plotly_chart(fig_xgb, use_container_width=True)
            
            # Classification Reports
            st.subheader(f"📊 Detailed Classification Reports - {year}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Random Forest**")
                rf_report = classification_report(rf_y_test, rf_preds, output_dict=True, zero_division=0)
                rf_report_df = pd.DataFrame(rf_report).transpose()
                st.dataframe(rf_report_df, use_container_width=True)
            
            with col2:
                st.markdown("**XGBoost**")
                xgb_report = classification_report(xgb_y_test, xgb_preds, output_dict=True, zero_division=0)
                xgb_report_df = pd.DataFrame(xgb_report).transpose()
                st.dataframe(xgb_report_df, use_container_width=True)
            
            # Sample Predictions Comparison
            st.subheader(f"Sample Predictions Comparison - {year}")
            
            # Merge predictions to show side by side
            comparison_all = rf_filtered.merge(
                xgb_filtered,
                on=['year', 'round', 'win_rate', 'avg_points', 'actual'],
                suffixes=('_rf', '_xgb')
            )
            
            # Show a diverse sample
            wins = comparison_all[comparison_all['actual'] == 1].head(10)
            no_wins = comparison_all[comparison_all['actual'] == 0].head(10)
            comparison_samples = pd.concat([wins, no_wins]).head(20)
            
            comparison_samples_display = comparison_samples[['year', 'round', 'win_rate', 'avg_points', 'actual', 'predicted_rf', 'predicted_xgb']].copy()
            comparison_samples_display.columns = ['Year', 'Round', 'Win Rate', 'Avg Points', 'Actual Win', 'RF Prediction', 'XGB Prediction']
            
            st.dataframe(comparison_samples_display, use_container_width=True)
            
            # Agreement statistics
            agree = (comparison_all['predicted_rf'] == comparison_all['predicted_xgb']).sum()
            total_merged = len(comparison_all)
            st.info(f"**Models Agreement**: Both models agree on {agree}/{total_merged} predictions ({agree/total_merged*100:.1f}%)")
            
        else:
            st.info(f"No predictions available for {year} from both models")
    elif not df_preds.empty:
        # Fallback to legacy predictions
        st.info("Showing legacy predictions. Run train_comparison.py to see both models.")
        pred_filtered = df_preds[df_preds["year"] == year].copy()
        
        if len(pred_filtered) > 0:
            y_test = pred_filtered['actual'].values
            preds = pred_filtered['predicted'].values
            acc = accuracy_score(y_test, preds)
            
            st.metric("Accuracy", f"{acc:.1%}")
            st.dataframe(pred_filtered[['year', 'round', 'win_rate', 'avg_points', 'actual', 'predicted']].head(20), use_container_width=True)
    else:
        st.info("No ML predictions available yet. Run the ML training service first.")


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
            "🔄 Model Comparison",
            "⏱️ Lap Time Analysis",
            "⛽ Pit Stop Strategy"
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
        
        ### ⏱️ Lap Time Analysis
        - Lap-by-lap performance tracking
        - Driver consistency analysis
        - Fastest lap comparisons
        - Race pace visualization
        
        ### ⛽ Pit Stop Strategy
        - Pit crew performance analysis
        - Team strategy comparison
        - Stop duration trends
        - Race-specific pit stop patterns
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
    
    elif page == "⏱️ Lap Time Analysis":
        render_lap_time_analysis()
    
    elif page == "⛽ Pit Stop Strategy":
        render_pit_stop_analysis()


# Always run main() - compatible with both direct execution and imports
main()
