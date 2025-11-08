"""
Main entry point for Streamlit Cloud deployment - Unified Dashboard.

Combines Classic ML and Big Data pipelines with offline artifact viewing.
"""

# Execute the unified dashboard directly
with open("dashboard/unified_app.py", encoding="utf-8") as f:
    exec(f.read())


