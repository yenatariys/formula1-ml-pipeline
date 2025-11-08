"""
Main entry point for Streamlit Cloud deployment.

This file automatically loads the unified dashboard which combines:
- Classic ML pipeline (scikit-learn models)
- Big Data pipeline (Spark MLlib + TensorFlow)
- Feature store visualization
- Model comparison

For local development with Docker, use the individual dashboards:
- Classic ML: http://localhost:8501
- Big Data: http://localhost:8502
- Unified: http://localhost:8503
"""

import sys
from pathlib import Path

# Add dashboard directory to Python path
dashboard_dir = Path(__file__).parent / "dashboard"
sys.path.insert(0, str(dashboard_dir))

# Import and run the unified dashboard
from unified_app import main

# Run the unified dashboard
main()
