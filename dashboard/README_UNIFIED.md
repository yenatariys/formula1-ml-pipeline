# Unified F1 ML Dashboard

A comprehensive dashboard that combines all Formula 1 machine learning pipelines into a single view.

## Features

### 📊 Combined View
- **Classic ML & Race Data**: Traditional machine learning models and race statistics
- **Spark MLlib Models**: Big data Random Forest classifier performance
- **TensorFlow Models**: Deep learning neural network metrics
- **Feature Store**: Engineered features and data distributions
- **Model Comparison**: Side-by-side performance comparison

## Running the Dashboard

### Option 1: Docker (Recommended)

```bash
# Start all services including the unified dashboard
docker compose up -d

# Or start just the unified dashboard
docker compose up -d dashboard_unified
```

The dashboard will be available at: **http://localhost:8503**

### Option 2: Local Development

```bash
# Windows
run_unified_dashboard.bat

# Linux/Mac
./run_unified_dashboard.sh
```

Or manually:

```bash
# Set environment variables (adjust paths as needed)
export F1_FEATURE_STORE_PATH=./artifacts/feature_store
export F1_MLLIB_EVAL_PATH=./artifacts/evaluations
export F1_TF_EVAL_PATH=./artifacts/evaluations

# Run streamlit
streamlit run dashboard/unified_app.py --server.port=8503
```

## Dashboard Sections

### 🏠 Home
- Overview of all pipelines
- Quick stats and status indicators
- Navigation guide

### 🏎️ Classic ML & Race Data
- Race results by season
- Driver performance statistics
- Traditional ML model metrics
- Interactive charts and tables

### ⚡ Spark MLlib Model
- Random Forest classifier performance
- Feature importance visualization
- Big data pipeline status
- Training/test set statistics

### 🧠 TensorFlow Model
- Neural network performance metrics
- Training history over epochs
- Loss and ROC-AUC curves
- Precision, Recall, and F1-Score

### 📦 Feature Store
- Sample of engineered features
- Feature distributions
- Data quality metrics
- Win rate analysis

### 🔄 Model Comparison
- ROC-AUC comparison across all models
- Performance metrics table
- Best model identification

## Dashboard Ports

- **Classic ML Dashboard**: http://localhost:8501
- **Big Data Dashboard**: http://localhost:8502
- **Unified Dashboard**: http://localhost:8503

## Prerequisites

### Python Packages
```bash
pip install streamlit pandas plotly sqlalchemy psycopg2-binary
```

### Required Services
- PostgreSQL (for classic ML data)
- Spark cluster (for big data models)
- Trained models and artifacts

## Architecture

```
dashboard/
├── classic_app.py      # Classic ML dashboard (standalone)
├── bigdata_app.py      # Big data pipeline dashboard (standalone)
├── unified_app.py      # Unified dashboard (combines both)
└── README.md          # This file
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `F1_FEATURE_STORE_PATH` | `artifacts/feature_store` | Path to feature store |
| `F1_MLLIB_EVAL_PATH` | `artifacts/evaluations` | Path to Spark MLlib metrics |
| `F1_TF_EVAL_PATH` | `artifacts/evaluations` | Path to TensorFlow metrics |

## Troubleshooting

### "Database connection not available"
- Ensure PostgreSQL is running
- Check database credentials in the code
- Verify the database contains the `f1_results_transformed` table

### "Artifacts not found"
- Run the training pipelines first:
  ```bash
  # Spark MLlib
  docker compose exec spark-master /opt/spark/bin/spark-submit \
    /app/pipelines/bigdata/spark/train_driver_win_mllib.py
  
  # TensorFlow
  python pipelines/bigdata/tensorflow/train_tensorflow_bigdata.py
  ```

### "Feature store data not available"
- Ensure the Spark pipeline has run and created features
- Check that CSV files exist in `artifacts/feature_store/driver_features_csv/`

## Benefits of Unified Dashboard

1. **Single Source of Truth**: All models and metrics in one place
2. **Easy Comparison**: Side-by-side model performance comparison
3. **Comprehensive View**: From raw data to production models
4. **Better Insights**: Correlate features, training, and results
5. **Streamlined Workflow**: No need to switch between multiple dashboards

## Future Enhancements

- [ ] Real-time model monitoring
- [ ] A/B testing comparison
- [ ] Model deployment status
- [ ] Automated model retraining triggers
- [ ] Custom metric thresholds and alerts
- [ ] Export capabilities for reports
