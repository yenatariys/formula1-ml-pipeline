"""
Main entry point for Streamlit Cloud deployment - Unified Dashboard.

Combines Classic ML and Big Data pipelines with offline artifact viewing.
"""

from pathlib import Path

# Get the path to unified_app.py
root_dir = Path(__file__).parent
unified_app_path = root_dir / "dashboard" / "unified_app.py"

# Execute the unified dashboard directly
with open(unified_app_path, encoding="utf-8") as f:
    code = f.read()
    exec(code)


