# ETL Auto-Update System

## 🎯 Overview

Dashboard sekarang **otomatis detect dan update** setelah ETL pipeline selesai!

## ⚙️ How It Works

### 1. ETL Pipeline (With Tracking)

```python
# etl/etl_pipeline.py
def main():
    tracker = ETLMetadataTracker()
    start_time = datetime.now()
    
    # Extract, Transform, Load
    df = extract_data(data_dir="data/")
    sdf = transform_data(df)
    df_transformed = sdf.toPandas()
    load_to_postgres(df_transformed, "f1_results_transformed")
    
    # Export for dashboard (NEW!)
    tracker.export_to_csv(df_transformed, "f1_results_latest.csv")
    
    # Save metadata (NEW!)
    tracker.save_etl_metadata(
        table_name="f1_results_transformed",
        record_count=len(df_transformed),
        start_time=start_time,
        end_time=datetime.now(),
        status="success"
    )
```

**What happens:**
1. ✅ Data loaded to PostgreSQL (table: `f1_results_transformed`)
2. ✅ Data exported to CSV (`data/etl_exports/f1_results_latest.csv`)
3. ✅ Metadata saved to JSON (`artifacts/etl_metadata/etl_run_info.json`)

### 2. Dashboard (With Auto-Detection)

```python
# dashboard/unified_app.py
@st.cache_data(ttl=300)  # Auto-refresh every 5 minutes
def load_etl_export():
    """Load latest ETL export if available."""
    export_path = BASE_DIR / "data" / "etl_exports" / "f1_results_latest.csv"
    if export_path.exists():
        return pd.read_csv(export_path)
    return pd.DataFrame()

def load_results():
    """Load with automatic fallback."""
    # Priority 1: ETL export (auto-updated)
    etl_data = load_etl_export()
    if not etl_data.empty:
        return etl_data
    
    # Priority 2: Database (if available)
    if DB_AVAILABLE:
        return pd.read_sql("SELECT * FROM f1_results_transformed", engine)
    
    # Priority 3: Empty
    return pd.DataFrame()
```

**What happens:**
1. ✅ Dashboard checks for ETL export first (fastest)
2. ✅ Falls back to database if export not available
3. ✅ Cache refreshes every 5 minutes
4. ✅ Shows data freshness on homepage

## 📊 Features

### Homepage ETL Status Banner

```
✅ ETL Data Available - Last updated: 2 hours ago (26,759 records)
```

Shows:
- ✅ Success status
- ⏰ How long ago ETL ran
- 📊 Number of records processed

### Classic ML Section Auto-Update

```
✅ Using ETL Export Data - Auto-updated after ETL runs
```

Indicates data source:
- **ETL Export** - Latest data from pipeline
- **PostgreSQL Database** - Direct database connection
- **CSV Files** - Fallback for Lap Time/Pit Stop sections

## 🚀 Usage

### Scenario 1: First Time Setup

```bash
# 1. Run ETL pipeline
docker-compose up -d postgres
python etl/etl_pipeline.py

# Output:
# [1/5] Extracting data...
# [2/5] Transforming data...
# [3/5] Converting to pandas...
# [4/5] Loading to PostgreSQL...
# [5/5] Exporting for dashboard...
# ✅ Data exported to data/etl_exports/f1_results_latest.csv
# ✅ Metadata saved to artifacts/etl_metadata/etl_run_info.json

# 2. Start dashboard
docker-compose up dashboard_unified

# Dashboard automatically shows latest data!
```

### Scenario 2: Update After Data Changes

```bash
# 1. Update source CSV files in data/
# (e.g., add new race results)

# 2. Re-run ETL
python etl/etl_pipeline.py

# ETL automatically:
# - Processes new data
# - Updates export file
# - Updates metadata timestamp

# 3. Dashboard auto-refreshes
# - Within 5 minutes, new data appears
# - Or click "Rerun" in Streamlit to force refresh
```

### Scenario 3: Streamlit Cloud Deployment

```bash
# Dashboard uses CSV files only (no ETL needed)
# But if you run ETL locally and commit exports:

git add data/etl_exports/f1_results_latest.csv
git add artifacts/etl_metadata/etl_run_info.json
git commit -m "Update ETL exports with latest data"
git push

# Streamlit Cloud will:
# - Pull latest code & data
# - Show ETL status banner
# - Use exported CSV data
```

## 📁 File Structure

```
formula1-ml-pipeline/
├── data/
│   ├── *.csv (source CSV files)
│   └── etl_exports/           # NEW!
│       └── f1_results_latest.csv   # Auto-updated by ETL
│
├── artifacts/
│   └── etl_metadata/          # NEW!
│       └── etl_run_info.json  # ETL run info & timestamp
│
├── etl/
│   ├── etl_pipeline.py        # Updated with tracking
│   └── metadata_tracker.py    # NEW! Tracks ETL runs
│
└── dashboard/
    └── unified_app.py         # Updated with auto-detection
```

## 🔄 Auto-Refresh Mechanism

### Cache TTL (Time To Live)

```python
@st.cache_data(ttl=300)  # 5 minutes
def load_etl_export():
    ...
```

- Cache expires every **5 minutes**
- Automatically checks for new data
- No manual refresh needed
- Works for both local and cloud deployment

### Manual Refresh

Users can force refresh by:
1. Click "Rerun" button in Streamlit
2. Press `R` key
3. Refresh browser page

## ✅ Benefits

### 1. **No More Stale Data**
- Dashboard always shows latest ETL results
- Automatic cache invalidation
- Real-time data freshness indicator

### 2. **Works Everywhere**
- ✅ Local development
- ✅ Docker Compose
- ✅ Streamlit Cloud (with committed exports)

### 3. **Multiple Data Sources**
- ✅ ETL export (primary)
- ✅ PostgreSQL (fallback)
- ✅ CSV files (fallback)

### 4. **User-Friendly**
- Shows data source
- Shows last update time
- Shows record count
- Clear status indicators

## 🎯 Example Output

### ETL Pipeline Output
```
============================================================
Starting ETL Pipeline...
============================================================

[1/5] Extracting data from CSV files...
✅ Extracted 26,759 records

[2/5] Transforming data with Spark...

[3/5] Converting Spark DataFrame to pandas...
✅ Transformed to 26,759 records with 15 columns

[4/5] Loading data to PostgreSQL...
✅ Loaded to table: f1_results_transformed

[5/5] Exporting data for dashboard...
✅ Data exported to data/etl_exports/f1_results_latest.csv
   Records: 26,759
   Columns: 15
✅ Metadata saved to artifacts/etl_metadata/etl_run_info.json
   Records: 26,759
   Duration: 45.23s
   Status: success

============================================================
✅ ETL Pipeline completed successfully!
============================================================
Duration: 45.23s
Records: 26,759
Export: data/etl_exports/f1_results_latest.csv
============================================================
```

### Dashboard Status Banner
```
✅ ETL Data Available - Last updated: 2 hours ago (26,759 records)

Welcome to the unified Formula 1 Machine Learning Dashboard!

### 🏎️ Classic ML & Race Data
- View race results and statistics
- Auto-updates after ETL runs!
```

## 🔧 Configuration

### Metadata File Format
```json
{
  "table_name": "f1_results_transformed",
  "record_count": 26759,
  "start_time": "2025-11-08T10:30:00",
  "end_time": "2025-11-08T10:30:45",
  "duration_seconds": 45.23,
  "status": "success",
  "error_message": null,
  "timestamp": "2025-11-08T10:30:45",
  "total_records": 26759,
  "total_columns": 15,
  "year_range": {"min": 2020, "max": 2024},
  "years_count": 5,
  "unique_drivers": 50,
  "unique_races": 100
}
```

## 📝 Notes

- ETL export file is **ignored by git** (optional - can commit if needed)
- Metadata file is **tracked by git** (small JSON file)
- Cache TTL can be adjusted (default: 5 minutes)
- Dashboard works offline with cached data
- No database connection required for exported data
