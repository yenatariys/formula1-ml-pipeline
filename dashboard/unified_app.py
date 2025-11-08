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
