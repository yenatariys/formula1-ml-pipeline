# 🚀 Formula 1 ML Pipeline - Quick Start Guide

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup & Installation](#setup--installation)
3. [Running the ETL Pipeline](#running-the-etl-pipeline)
4. [Running the Dashboard](#running-the-dashboard)
5. [Development Workflow](#development-workflow)
6. [Deployment Options](#deployment-options)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Docker Desktop** (Windows/Mac) or Docker Engine (Linux)
- **Docker Compose** v2.0+
- **Git**
- **Python 3.11+** (optional, for local development)

### System Requirements
- **RAM:** 8GB minimum, 16GB recommended
- **Disk:** 10GB free space
- **CPU:** 4 cores recommended (for Spark processing)

### Verify Installation
```bash
# Check Docker
docker --version
# Expected: Docker version 20.10+

# Check Docker Compose
docker-compose --version
# Expected: Docker Compose version v2.0+

# Check Git
git --version
# Expected: git version 2.30+
```

---

## Setup & Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/yenatariys/formula1-ml-pipeline.git
cd formula1-ml-pipeline
```

### Step 2: Verify Data Files
```bash
# Check that CSV files exist
ls data/*.csv

# Expected files (15 CSV files):
# - races.csv
# - results.csv
# - drivers.csv
# - constructors.csv
# - circuits.csv
# ... and 10 more
```

### Step 3: Start PostgreSQL Database
```bash
docker-compose up -d postgres

# Wait for database to be ready (30-60 seconds)
docker-compose logs -f postgres
# Look for: "database system is ready to accept connections"
```

### Step 4: Initialize Database Schema
```bash
# Database schema is auto-created by init.sql
# Verify tables exist:
docker exec -it f1_postgres psql -U admin -d f1_data -c "\dt"

# Expected tables:
# - f1_results_transformed
# - (and others from init.sql)
```

---

## Running the ETL Pipeline

### What ETL Does
1. **Extract** - Read CSV files from `data/` folder
2. **Transform** - Clean & process with Spark (filter DNF, cast types)
3. **Load** - Save to PostgreSQL database
4. **Export** - Generate CSV for dashboard (`data/etl_exports/f1_results_latest.csv`)
5. **Track** - Save metadata to `artifacts/etl_metadata/etl_run_info.json`

### Run ETL (Method 1: Docker - Recommended)
```bash
# Start required services
docker-compose up -d postgres spark-master

# Run ETL pipeline
docker-compose run --rm etl python etl/etl_pipeline.py

# Expected output:
# ============================================================
# Starting ETL Pipeline...
# ============================================================
# 
# [1/5] Extracting data from CSV files...
# ✅ Extracted 26,759 records
# 
# [2/5] Transforming data with Spark...
# 
# [3/5] Converting Spark DataFrame to pandas...
# ✅ Transformed to 24,123 records with 8 columns
# 
# [4/5] Loading data to PostgreSQL...
# ✅ Loaded to table: f1_results_transformed
# 
# [5/5] Exporting data for dashboard...
# ✅ Data exported to data/etl_exports/f1_results_latest.csv
# ✅ Metadata saved to artifacts/etl_metadata/etl_run_info.json
# 
# ============================================================
# ✅ ETL Pipeline completed successfully!
# ============================================================
# Duration: 45.23s
# Records: 24,123
```

### Run ETL (Method 2: Local Python)
```bash
# Requires: Python 3.11+, PySpark installed locally
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run ETL
python etl/etl_pipeline.py
```

### Verify ETL Success
```bash
# Check export file exists
ls -lh data/etl_exports/f1_results_latest.csv
# Expected: ~1.2MB file

# Check metadata
cat artifacts/etl_metadata/etl_run_info.json
# Expected: JSON with status: "success"

# Check database
docker exec -it f1_postgres psql -U admin -d f1_data \
  -c "SELECT COUNT(*) FROM f1_results_transformed;"
# Expected: 24,123 rows
```

---

## Running the Dashboard

### Start Dashboard (Docker - Recommended)
```bash
# Start all services (Postgres + Dashboard)
docker-compose up -d postgres dashboard_unified

# Access dashboard
# Open browser: http://localhost:8501

# View logs
docker-compose logs -f dashboard_unified
```

### Start Dashboard (Local - Development)
```bash
# Setup Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install minimal dependencies (for Streamlit Cloud compatibility)
pip install streamlit pandas numpy scikit-learn plotly

# Run dashboard
streamlit run dashboard/unified_app.py

# Or with full dependencies (Big Data ML support)
pip install -r dashboard/requirements.txt
streamlit run dashboard/unified_app.py
```

### Dashboard Features

#### 🏠 Home Page
- ETL status banner with freshness indicator
- Quick metrics from all ML models
- Navigation to all sections

#### 🏎️ Classic ML & Race Data
- **Data Source Priority:**
  1. ✅ ETL Export (transformed, auto-updated)
  2. ✅ PostgreSQL Database (transformed)
  3. ⚠️ Raw CSV (fallback, shows warning)
- Race results visualization
- Wins by driver, points distribution
- Season-by-season analysis

#### ⚡ Spark MLlib Model (Big Data ML)
- PySpark-based driver win prediction
- Feature engineering with Spark
- Model metrics: ROC-AUC, accuracy

#### 🧠 TensorFlow Model
- Deep learning driver win prediction
- Training history visualization
- Performance metrics

#### 📦 Feature Store
- Feature engineering pipeline results
- Driver features (CSV & Parquet)
- Feature quality metrics

#### 🔄 Model Comparison
- Compare all 3 models side-by-side
- Performance benchmarks
- Model selection guidance

#### ⏱️ Lap Time Analysis (CSV-based)
- **6 ML Models:**
  - Classic ML: RandomForest, KMeans, Z-score
  - Big Data ML: Spark RandomForest, Spark KMeans, Spark Window
- Lap time prediction
- Circuit-specific analysis
- Anomaly detection

#### ⛽ Pit Stop Strategy (CSV-based)
- **6 ML Models:**
  - Classic ML: Duration prediction, strategy classification, clustering
  - Big Data ML: Spark regression, Spark classification, Spark clustering
- Pit stop duration prediction
- Optimal strategy recommendations
- Team performance analysis

### Dashboard Data Sources

```
Priority 1: ETL Export ✅ BEST
├─ File: data/etl_exports/f1_results_latest.csv
├─ Status: Auto-updated after ETL runs
├─ Data Type: TRANSFORMED (DNF filtered, types cast)
└─ Cache: 5-minute TTL (auto-refresh)

Priority 2: PostgreSQL Database ✅ GOOD
├─ Table: f1_results_transformed
├─ Status: Available when docker-compose up
└─ Data Type: TRANSFORMED (same as ETL export)

Priority 3: Raw CSV ⚠️ FALLBACK ONLY
├─ File: data/f1_results_joined.csv
├─ Status: Always available (committed to git)
├─ Data Type: UNTRANSFORMED (raw JOIN, includes DNF)
└─ Warning: Dashboard shows yellow banner
```

---

## Development Workflow

### Daily Development Flow

#### 1. Start Services
```bash
# Start all services
docker-compose up -d

# Or start selectively
docker-compose up -d postgres dashboard_unified
```

#### 2. Make Code Changes
```bash
# Edit files in your IDE
# - etl/*.py
# - dashboard/*.py
# - ml/*.py
```

#### 3. Test ETL Changes
```bash
# Rebuild ETL container (if dependencies changed)
docker-compose build etl

# Run ETL
docker-compose run --rm etl python etl/etl_pipeline.py
```

#### 4. Test Dashboard Changes
```bash
# Dashboard auto-reloads on file changes (volume mounted)
# Just refresh browser: http://localhost:8501

# Or restart container
docker-compose restart dashboard_unified
```

#### 5. Commit & Push
```bash
git add .
git commit -m "Your commit message"
git push
```

### Project Structure
```
formula1-ml-pipeline/
├── data/                      # Source CSV files (15 files)
│   ├── *.csv                 # Race data (races, results, drivers, etc.)
│   └── etl_exports/          # ETL output (transformed data)
│       └── f1_results_latest.csv
│
├── etl/                       # ETL pipeline
│   ├── etl_pipeline.py       # Main orchestrator
│   ├── extract_data.py       # CSV extraction
│   ├── transform_data.py     # Spark transformations
│   ├── load_data.py          # PostgreSQL loading
│   └── metadata_tracker.py   # Metadata & export tracking
│
├── dashboard/                 # Streamlit dashboard
│   └── unified_app.py        # SINGLE unified dashboard
│
├── ml/                        # Classic ML models
│   ├── train.py              # Classic ML training
│   └── train_comparison.py   # Model comparison
│
├── pipelines/                 # Big Data ML pipelines
│   ├── bigdata/
│   │   ├── spark/            # Spark MLlib
│   │   └── tensorflow/       # TensorFlow
│   └── classic/              # Classic ML
│
├── analytics/                 # Graph analytics
│   └── graph_analysis.py     # Neo4j integration
│
├── artifacts/                 # Generated artifacts
│   ├── feature_store/        # Feature engineering outputs
│   ├── evaluations/          # Model metrics
│   └── etl_metadata/         # ETL run tracking
│
├── docs/                      # Documentation
│   ├── QUICKSTART.md         # This file
│   ├── ETL_AUTO_UPDATE.md    # ETL auto-update guide
│   └── runbooks/             # Step-by-step guides
│
├── docker-compose.yml         # Service orchestration
├── Dockerfile                 # ETL container definition
└── requirements.txt           # Python dependencies (Streamlit Cloud)
```

---

## Deployment Options

### Option 1: Streamlit Cloud (Lightweight)

**Use Case:** Public demo, no ETL needed

**Setup:**
```bash
# 1. Push to GitHub
git push

# 2. Go to: https://share.streamlit.io/
# 3. Connect your GitHub repo
# 4. Set main file: dashboard/unified_app.py
# 5. Deploy!
```

**Features:**
- ✅ Lap Time Analysis (CSV-based)
- ✅ Pit Stop Strategy (CSV-based)
- ✅ Classic ML (uses raw CSV fallback)
- ❌ ETL Export (not available)
- ❌ Database (not available)
- ❌ Big Data ML (PySpark not supported)

**Data Source:**
- Uses `data/f1_results_joined.csv` (raw CSV)
- Shows yellow warning banner
- Instructions to run ETL locally

### Option 2: Docker Compose (Full Stack)

**Use Case:** Local development, full features

**Setup:**
```bash
# Start all services
docker-compose up -d

# Run ETL
docker-compose run --rm etl python etl/etl_pipeline.py

# Access dashboard: http://localhost:8501
```

**Features:**
- ✅ All dashboard sections
- ✅ ETL Export (transformed data)
- ✅ PostgreSQL Database
- ✅ Big Data ML (Spark, TensorFlow)
- ✅ Neo4j Graph Analytics (optional)

**Services:**
- PostgreSQL (f1_postgres:5432)
- Spark Master (spark-master:7077)
- Spark Worker (spark-worker)
- Dashboard (localhost:8501)
- Neo4j (localhost:7474) - optional

### Option 3: Local Python (Minimal)

**Use Case:** Quick testing, no Docker

**Setup:**
```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn plotly

# Run dashboard
streamlit run dashboard/unified_app.py
```

**Features:**
- ✅ Lap Time Analysis
- ✅ Pit Stop Strategy
- ✅ Classic ML (CSV fallback)
- ❌ ETL, Database, Big Data ML

---

## Troubleshooting

### ETL Issues

#### Problem: "Postgres is not ready"
```bash
# Solution: Wait for Postgres to start
docker-compose up -d postgres
docker-compose logs -f postgres
# Wait for: "database system is ready to accept connections"
```

#### Problem: "Module not found: metadata_tracker"
```bash
# Solution: Rebuild ETL container
docker-compose build etl
docker-compose run --rm etl python etl/etl_pipeline.py
```

#### Problem: "Network error during pip install"
```bash
# Solution: Check internet connection, try again
docker-compose build --no-cache etl
```

### Dashboard Issues

#### Problem: "No race results data available"
```bash
# Solution: Run ETL pipeline to generate data
docker-compose run --rm etl python etl/etl_pipeline.py

# Or check if raw CSV exists
ls data/f1_results_joined.csv
```

#### Problem: "Using Raw CSV Data" warning
```bash
# Solution: This is expected if ETL hasn't run
# Run ETL to get transformed data:
docker-compose run --rm etl python etl/etl_pipeline.py

# Dashboard will auto-detect within 5 minutes
# Or click "Rerun" button in Streamlit
```

#### Problem: Dashboard not auto-refreshing
```bash
# Solution 1: Clear Streamlit cache
# Click "Clear Cache" in dashboard menu (⋮)

# Solution 2: Restart container
docker-compose restart dashboard_unified

# Solution 3: Wait for cache TTL (5 minutes)
```

### Docker Issues

#### Problem: "Port 8501 already in use"
```bash
# Solution: Stop existing Streamlit instance
docker-compose down
docker ps | grep streamlit
docker stop <container_id>
```

#### Problem: "Container exited with code 137"
```bash
# Solution: Increase Docker memory
# Docker Desktop → Settings → Resources → Memory: 8GB+
```

#### Problem: "Permission denied" on Linux
```bash
# Solution: Run with sudo or add user to docker group
sudo usermod -aG docker $USER
# Log out and log back in
```

---

## Advanced Usage

### Run Graph Analytics
```bash
# Set environment variable
export RUN_GRAPH_ANALYTICS=true

# Run ETL with graph analytics
docker-compose run --rm etl python etl/etl_pipeline.py

# Graph data exported to artifacts/
```

### Train ML Models
```bash
# Classic ML
docker-compose run --rm ml python ml/train.py

# Spark MLlib
docker-compose run --rm spark python pipelines/bigdata/spark/train_driver_win_mllib.py

# TensorFlow
docker-compose run --rm ml python pipelines/bigdata/tensorflow/train_tensorflow_bigdata.py
```

### Access PostgreSQL
```bash
# Connect to database
docker exec -it f1_postgres psql -U admin -d f1_data

# Run queries
SELECT COUNT(*) FROM f1_results_transformed;
SELECT * FROM f1_results_transformed LIMIT 10;

# Export data
docker exec -it f1_postgres psql -U admin -d f1_data \
  -c "COPY f1_results_transformed TO STDOUT CSV HEADER" > export.csv
```

### Monitor Services
```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f dashboard_unified

# Check resource usage
docker stats
```

---

## Quick Reference

### Essential Commands
```bash
# Start everything
docker-compose up -d

# Run ETL
docker-compose run --rm etl python etl/etl_pipeline.py

# Open dashboard
open http://localhost:8501  # Mac
start http://localhost:8501  # Windows
xdg-open http://localhost:8501  # Linux

# Stop everything
docker-compose down

# Clean up (WARNING: deletes volumes!)
docker-compose down -v
```

### Key Files
- **Dashboard:** `dashboard/unified_app.py`
- **ETL Pipeline:** `etl/etl_pipeline.py`
- **ETL Export:** `data/etl_exports/f1_results_latest.csv`
- **Metadata:** `artifacts/etl_metadata/etl_run_info.json`
- **Config:** `docker-compose.yml`

### Useful URLs
- **Dashboard:** http://localhost:8501
- **Spark Master:** http://localhost:8080
- **Neo4j Browser:** http://localhost:7474
- **PostgreSQL:** localhost:5432

---

## Getting Help

### Documentation
- **Full docs:** `docs/` folder
- **ETL Guide:** `docs/ETL_AUTO_UPDATE.md`
- **Runbooks:** `docs/runbooks/`

### Common Issues
- Check this guide's Troubleshooting section
- Review container logs: `docker-compose logs`
- Check GitHub Issues

### Support
- GitHub Issues: https://github.com/yenatariys/formula1-ml-pipeline/issues
- Email: [Your email]

---

## Next Steps

1. ✅ Complete this Quick Start
2. 📖 Read `docs/ETL_AUTO_UPDATE.md` for ETL details
3. 🔧 Explore dashboard sections (Lap Time, Pit Stop)
4. 🚀 Train ML models (`ml/train.py`)
5. 📊 Run graph analytics (set `RUN_GRAPH_ANALYTICS=true`)
6. 🌐 Deploy to Streamlit Cloud

**Happy analyzing! 🏎️💨**
