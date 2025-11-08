"""
ETL Metadata Tracker - Track ETL run status and export metadata.

This module provides functions to:
1. Save ETL run metadata after completion
2. Export latest data to CSV for dashboard consumption
3. Track data freshness and update timestamps
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class ETLMetadataTracker:
    """Track ETL execution metadata and data exports."""
    
    def __init__(self, base_dir: Path = None):
        """Initialize metadata tracker.
        
        Args:
            base_dir: Base directory for the project (default: parent of etl/)
        """
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[1]
        
        self.base_dir = base_dir
        self.metadata_dir = base_dir / "artifacts" / "etl_metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.metadata_dir / "etl_run_info.json"
        self.export_dir = base_dir / "data" / "etl_exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def save_etl_metadata(
        self,
        table_name: str,
        record_count: int,
        start_time: datetime,
        end_time: datetime,
        status: str = "success",
        error_message: Optional[str] = None,
        additional_info: Optional[Dict] = None
    ) -> None:
        """Save ETL run metadata to JSON file.
        
        Args:
            table_name: Name of the database table
            record_count: Number of records processed
            start_time: ETL start timestamp
            end_time: ETL end timestamp
            status: ETL status (success/failed)
            error_message: Error message if failed
            additional_info: Additional metadata
        """
        duration_seconds = (end_time - start_time).total_seconds()
        
        metadata = {
            "table_name": table_name,
            "record_count": record_count,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "status": status,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
            "data_type": "transformed",  # NEW: Indicator for transformed data
            "transformations_applied": [
                "Filter null positions (DNF/DNS removed)",
                "Cast year/round/position to int",
                "Cast points to float",
                "Spark-based data quality checks"
            ]
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        # Save to file
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Metadata saved to {self.metadata_file}")
        print(f"   Records: {record_count:,}")
        print(f"   Duration: {duration_seconds:.2f}s")
        print(f"   Status: {status}")
    
    def export_to_csv(
        self,
        df: pd.DataFrame,
        filename: str = "f1_results_latest.csv"
    ) -> Path:
        """Export DataFrame to CSV for dashboard consumption.
        
        Args:
            df: DataFrame to export
            filename: Output filename
            
        Returns:
            Path to exported CSV file
        """
        export_path = self.export_dir / filename
        df.to_csv(export_path, index=False)
        
        print(f"✅ Data exported to {export_path}")
        print(f"   Records: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        
        return export_path
    
    def load_metadata(self) -> Optional[Dict]:
        """Load the latest ETL metadata.
        
        Returns:
            Metadata dictionary or None if not found
        """
        if not self.metadata_file.exists():
            return None
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def get_data_freshness(self) -> Optional[str]:
        """Get human-readable data freshness info.
        
        Returns:
            String like "2 hours ago" or None
        """
        metadata = self.load_metadata()
        if not metadata:
            return None
        
        try:
            end_time = datetime.fromisoformat(metadata['end_time'])
            delta = datetime.now() - end_time
            
            if delta.days > 0:
                return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
            elif delta.seconds >= 3600:
                hours = delta.seconds // 3600
                return f"{hours} hour{'s' if hours > 1 else ''} ago"
            elif delta.seconds >= 60:
                minutes = delta.seconds // 60
                return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
            else:
                return "just now"
        except (ValueError, KeyError):
            return None


def create_etl_summary(df: pd.DataFrame) -> Dict:
    """Create summary statistics for ETL data.
    
    Args:
        df: Transformed DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
    }
    
    # Add column-specific stats if available
    if 'year' in df.columns:
        summary['year_range'] = {
            'min': int(df['year'].min()),
            'max': int(df['year'].max())
        }
        summary['years_count'] = int(df['year'].nunique())
    
    if 'driverId' in df.columns:
        summary['unique_drivers'] = int(df['driverId'].nunique())
    
    if 'raceId' in df.columns:
        summary['unique_races'] = int(df['raceId'].nunique())
    
    return summary
