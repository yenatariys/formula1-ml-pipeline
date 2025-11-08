# Streamlit Cloud Deployment Guide

## 📋 Prerequisites

1. GitHub repository: `yenatariys/formula1-ml-pipeline`
2. Streamlit Cloud account connected to GitHub
3. Entry point: `streamlit_app.py`

## 🚀 Deployment Steps

### 1. Configure Streamlit Cloud App

**App Settings:**
- Main file path: `streamlit_app.py`
- Python version: 3.11
- Repository: `yenatariys/formula1-ml-pipeline`
- Branch: `main`

### 2. Requirements

The app uses minimal dependencies for cloud compatibility:

```txt
streamlit==1.38.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.3.2
plotly==5.23.0
```

**Why minimal?**
- ❌ Removed `psycopg2-binary` - causes C-extension build errors
- ❌ Removed `sqlalchemy` - not needed without database
- ❌ Removed `pyspark` - too heavy for cloud (3GB+)
- ❌ Removed `tensorflow` - requires GPU/special build
- ✅ Kept only pure Python or well-supported packages

### 3. Available Features in Cloud

**✅ Working Features:**
- 🏠 Home Dashboard (overview)
- ⏱️ Lap Time Analysis (Classic ML mode)
  - Lap time prediction (sklearn RandomForest)
  - Driver clustering (sklearn KMeans)
  - Anomaly detection (pandas Z-score)
- ⛽ Pit Stop Strategy (Classic ML mode)
  - Duration prediction (sklearn RandomForest)
  - Strategy classification (sklearn RF Classifier)
  - Team clustering (sklearn KMeans)
- 📊 All visualizations (Plotly charts)

**⚠️ Limited Features:**
- 🏎️ Classic ML - Database not available (no PostgreSQL on cloud)
- ⚡ Spark MLlib - Big Data ML mode disabled (no PySpark)
- 🧠 TensorFlow - Model viewing only (no training)
- 🔄 Feature Store - Artifacts may not be available

**🔧 Automatic Fallbacks:**
- `PYSPARK_AVAILABLE = False` → Only Classic ML mode shown
- `DB_AVAILABLE = False` → Database features disabled gracefully
- `SQLALCHEMY_AVAILABLE = False` → No database connection attempted

## 🔍 Troubleshooting

### Error: "installer returned a non-zero exit code"

**Cause:** Heavy dependencies or C-extension packages failing to build

**Solution:** 
1. Use minimal `requirements.txt` (already implemented)
2. Remove packages with C-extensions: psycopg2, numpy<1.24, etc.
3. Use exact versions instead of `>=` ranges

### Error: "Module not found: pyspark"

**Cause:** Code trying to import PySpark

**Solution:**
```python
try:
    from pyspark.sql import SparkSession
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
```

### Error: Database connection failed

**Expected behavior** - Database features gracefully disabled:
```python
if not DB_AVAILABLE:
    st.warning("Database not available. Using CSV data only.")
```

## 📊 Performance Optimization

**Memory Usage:**
- Streamlit Cloud limit: 1 GB RAM
- Current usage: ~300-500 MB (well within limits)

**Data Loading:**
- CSV files cached with `@st.cache_data`
- Maximum 500,000 rows loaded per dataset
- Automatic sampling for large files

**Model Training:**
- Classic ML models: < 100 MB memory
- Training time: < 30 seconds per model
- Real-time predictions: < 1 second

## ✅ Deployment Checklist

Before deploying:
- [x] Minimal requirements.txt (no heavy packages)
- [x] Optional imports wrapped in try/except
- [x] Database connection checks
- [x] PySpark graceful fallback
- [x] Streamlit config file (.streamlit/config.toml)
- [x] Entry point: streamlit_app.py exists
- [x] All paths use Path() for cross-platform compatibility

## 🎯 Expected Result

After successful deployment, users will see:

1. **Homepage** with project overview
2. **Lap Time Analysis** with 3 ML models (Classic ML only)
3. **Pit Stop Strategy** with 3 ML models (Classic ML only)
4. All charts and visualizations working
5. No database features (gracefully disabled)
6. No Big Data ML option (PySpark disabled)

## 🔗 Live Demo

Once deployed successfully:
- URL: `https://[your-app-name].streamlit.app`
- Load time: ~10-15 seconds first visit
- Subsequent loads: ~2-3 seconds (cached)

## 📝 Notes

- Cloud deployment is **read-only** (no file writes)
- CSV files must be in repository
- Artifacts loaded from `artifacts/` directory
- No training artifacts needed for cloud (optional)
- Focus on visualization and pre-trained model results

## 🚀 Next Steps After Successful Deployment

1. Test all navigation sections
2. Verify CSV data loads correctly
3. Run sample ML predictions
4. Check all visualizations render
5. Monitor resource usage in Streamlit Cloud dashboard
6. Share app URL with stakeholders
