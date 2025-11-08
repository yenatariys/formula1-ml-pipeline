# etl_pipeline.py
import os
import time
from pathlib import Path
from datetime import datetime

import psycopg2

from extract_data import extract_data
from transform_data import transform_data
from load_data import load_to_postgres
from metadata_tracker import ETLMetadataTracker, create_etl_summary

# -----------------------------------------------------------
# Tunggu Postgres siap
# -----------------------------------------------------------
while True:
    try:
        conn = psycopg2.connect(
            dbname="f1_data",
            user="admin",
            password="admin123",
            host="f1_postgres",
            port=5432
        )
        conn.close()
        print("Postgres is ready!")
        break
    except psycopg2.OperationalError:
        print("Waiting for Postgres...")
        time.sleep(2)

# -----------------------------------------------------------
# Main ETL function
# -----------------------------------------------------------
def _str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def run_graph_analytics_if_enabled() -> None:
    if not _str_to_bool(os.getenv("RUN_GRAPH_ANALYTICS", "false")):
        print("Graph analytics step skipped (RUN_GRAPH_ANALYTICS not enabled).")
        return

    from analytics.graph_analysis import run_graph_analysis

    base_path = os.getenv("GRAPH_BASE_PATH", "data")
    export_dir = os.getenv("GRAPH_EXPORT_DIR")
    top_k = int(os.getenv("GRAPH_TOP_K", "10"))

    min_year_env = os.getenv("GRAPH_MIN_YEAR")
    max_year_env = os.getenv("GRAPH_MAX_YEAR")
    min_year = int(min_year_env) if min_year_env else None
    max_year = int(max_year_env) if max_year_env else None

    neo4j_uri = os.getenv("GRAPH_NEO4J_URI")
    neo4j_config = None
    if neo4j_uri:
        neo4j_user = os.getenv("GRAPH_NEO4J_USER")
        neo4j_password = os.getenv("GRAPH_NEO4J_PASSWORD")
        if not neo4j_user or not neo4j_password:
            print("GRAPH_NEO4J_USER and GRAPH_NEO4J_PASSWORD must be set when GRAPH_NEO4J_URI is provided. Skipping Neo4j export.")
        else:
            wipe_flag = _str_to_bool(os.getenv("GRAPH_NEO4J_WIPE", "false"))
            neo4j_config = {
                "uri": neo4j_uri,
                "user": neo4j_user,
                "password": neo4j_password,
                "wipe": wipe_flag,
            }

    try:
        run_graph_analysis(
            base_path=Path(base_path),
            min_year=min_year,
            max_year=max_year,
            top_k=top_k,
            export_dir=Path(export_dir) if export_dir else None,
            neo4j_config=neo4j_config,
            echo=True,
        )
        print("Graph analytics step completed.")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Graph analytics step failed: {exc}")


def main():
    """Main ETL pipeline with metadata tracking."""
    print("=" * 60)
    print("Starting ETL Pipeline...")
    print("=" * 60)
    
    tracker = ETLMetadataTracker()
    start_time = datetime.now()
    
    try:
        # Extract data from CSV
        print("\n[1/5] Extracting data from CSV files...")
        df = extract_data(data_dir="data/")
        print(f"✅ Extracted {len(df):,} records")
        
        # Transform using Spark
        print("\n[2/5] Transforming data with Spark...")
        sdf = transform_data(df)
        
        # Convert Spark DF to pandas DF
        print("\n[3/5] Converting Spark DataFrame to pandas...")
        df_transformed = sdf.toPandas()
        print(f"✅ Transformed to {len(df_transformed):,} records with {len(df_transformed.columns)} columns")
        
        # Load to PostgreSQL
        print("\n[4/5] Loading data to PostgreSQL...")
        load_to_postgres(df_transformed, "f1_results_transformed")
        print(f"✅ Loaded to table: f1_results_transformed")
        
        # Export to CSV for dashboard
        print("\n[5/5] Exporting data for dashboard...")
        export_path = tracker.export_to_csv(df_transformed, "f1_results_latest.csv")
        
        # Save metadata
        end_time = datetime.now()
        summary = create_etl_summary(df_transformed)
        
        tracker.save_etl_metadata(
            table_name="f1_results_transformed",
            record_count=len(df_transformed),
            start_time=start_time,
            end_time=end_time,
            status="success",
            additional_info=summary
        )
        
        print("\n" + "=" * 60)
        print("✅ ETL Pipeline completed successfully!")
        print("=" * 60)
        print(f"Duration: {(end_time - start_time).total_seconds():.2f}s")
        print(f"Records: {len(df_transformed):,}")
        print(f"Export: {export_path}")
        print("=" * 60)
        
        # Run graph analytics if enabled
        run_graph_analytics_if_enabled()
        
    except Exception as e:
        end_time = datetime.now()
        print(f"\n❌ ETL Pipeline failed: {e}")
        
        tracker.save_etl_metadata(
            table_name="f1_results_transformed",
            record_count=0,
            start_time=start_time,
            end_time=end_time,
            status="failed",
            error_message=str(e)
        )
        raise

# -----------------------------------------------------------
if __name__ == "__main__":
    main()