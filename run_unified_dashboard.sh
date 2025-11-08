#!/bin/bash
# Run the unified F1 ML Dashboard locally

echo "Starting Unified F1 ML Dashboard..."
echo ""
echo "The dashboard will be available at: http://localhost:8503"
echo ""
echo "Make sure you have the required Python packages installed:"
echo "  pip install streamlit pandas plotly sqlalchemy psycopg2-binary"
echo ""

# Set environment variables for local paths
export F1_FEATURE_STORE_PATH="./artifacts/feature_store"
export F1_MLLIB_EVAL_PATH="./artifacts/evaluations"
export F1_TF_EVAL_PATH="./artifacts/evaluations"

# Run streamlit
streamlit run dashboard/unified_app.py --server.port=8503 --server.address=localhost
