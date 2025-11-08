# Formula 1 Dashboard

This directory contains the **unified Streamlit dashboard** that combines all ML features in one place.

## Main App

**`unified_app.py`** - Single unified dashboard with all features:
- 🏠 Home - Overview & ETL status
- 🏎️ Classic ML & Race Data - Race results & analytics
- ⚡ Spark MLlib Model - Big Data ML features
- 🧠 TensorFlow Model - Deep learning predictions
- 📦 Feature Store - Feature engineering pipeline
- 🔄 Model Comparison - Compare all models
- ⏱️ Lap Time Analysis - 6 ML models (Classic + Big Data)
- ⛽ Pit Stop Strategy - 6 ML models (Classic + Big Data)

## Running the Dashboard

### Docker (Recommended)
```bash
# Start dashboard with all dependencies
docker-compose up -d dashboard

# Access at: http://localhost:8501
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run unified_app.py

# Or minimal (Streamlit Cloud compatible)
pip install streamlit pandas numpy scikit-learn plotly
streamlit run unified_app.py
```

## Data Sources

Dashboard automatically detects and uses the best available data source:

1. **ETL Export** (Priority 1) - `data/etl_exports/f1_results_latest.csv`
   - Transformed & cleaned data
   - Auto-updated after ETL runs
   - Shows green status banner

2. **PostgreSQL Database** (Priority 2) - `f1_results_transformed` table
   - Transformed data from ETL pipeline
   - Available when docker-compose is running

3. **Raw CSV** (Priority 3) - `data/f1_results_joined.csv`
   - Untransformed data (fallback only)
   - Shows yellow warning banner
   - Includes DNF/null positions

## Deployment Options

### Streamlit Cloud
- Uses raw CSV (no ETL/database needed)
- Supports Lap Time & Pit Stop sections
- Classic ML shows fallback data

### Docker Compose
- Full features including ETL, Database, Big Data ML
- See `QUICKSTART.md` for setup instructions

## Supporting Files

- `Dockerfile` - Dashboard container definition
- `requirements.txt` - Full dependencies (Docker)
- `../requirements.txt` - Minimal dependencies (Streamlit Cloud)
- `README_UNIFIED.md` - Detailed unified dashboard documentation

## Documentation

- **Quick Start:** `../QUICKSTART.md`
- **ETL Auto-Update:** `../docs/ETL_AUTO_UPDATE.md`
- **Runbooks:** `../docs/runbooks/05-dashboards.md`

