"""
Manual ETL export script - Export data from PostgreSQL to CSV for dashboard.
Use this when ETL container has old code but database already has transformed data.
"""
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
from datetime import datetime
import json

# Database connection
engine = create_engine("postgresql+psycopg2://admin:admin123@localhost:5432/f1_data")

print("=" * 60)
print("Exporting data from PostgreSQL to dashboard...")
print("=" * 60)

try:
    # Read from database
    print("\n[1/3] Reading from PostgreSQL...")
    df = pd.read_sql("SELECT * FROM f1_results_transformed", engine)
    print(f"✅ Loaded {len(df):,} records from database")
    
    # Export to CSV
    print("\n[2/3] Exporting to CSV...")
    export_dir = Path("data/etl_exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    export_path = export_dir / "f1_results_latest.csv"
    df.to_csv(export_path, index=False)
    print(f"✅ Exported to {export_path}")
    print(f"   Records: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    # Save metadata
    print("\n[3/3] Saving metadata...")
    metadata_dir = Path("artifacts/etl_metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "table_name": "f1_results_transformed",
        "record_count": len(df),
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
        "duration_seconds": 0,
        "status": "success",
        "error_message": None,
        "timestamp": datetime.now().isoformat(),
        "data_type": "transformed",
        "transformations_applied": [
            "Filter null positions (DNF/DNS removed)",
            "Cast year/round/position to int",
            "Cast points to float",
            "Spark-based data quality checks"
        ],
        "total_records": len(df),
        "total_columns": len(df.columns),
        "export_method": "manual_database_export"
    }
    
    metadata_file = metadata_dir / "etl_run_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to {metadata_file}")
    
    print("\n" + "=" * 60)
    print("✅ Export completed successfully!")
    print("=" * 60)
    print(f"Records: {len(df):,}")
    print(f"Export: {export_path}")
    print(f"Dashboard will auto-detect within 5 minutes")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Export failed: {e}")
    raise
