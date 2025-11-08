@echo off
REM Run the unified F1 ML Dashboard locally

echo Starting Unified F1 ML Dashboard...
echo.
echo The dashboard will be available at: http://localhost:8503
echo.
echo Make sure you have the required Python packages installed:
echo   pip install streamlit pandas plotly sqlalchemy psycopg2-binary
echo.

REM Set environment variables for local paths
set F1_FEATURE_STORE_PATH=d:\Work\Github\formula1-ml-pipeline\artifacts\feature_store
set F1_MLLIB_EVAL_PATH=d:\Work\Github\formula1-ml-pipeline\artifacts\evaluations
set F1_TF_EVAL_PATH=d:\Work\Github\formula1-ml-pipeline\artifacts\evaluations

REM Run streamlit
streamlit run dashboard/unified_app.py --server.port=8503 --server.address=localhost
