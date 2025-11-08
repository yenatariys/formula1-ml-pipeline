# Dashboards (Step 05)

This directory houses both Streamlit applications used to visualise pipeline outputs.

## Apps
- `classic_app.py` – Postgres-backed dashboard that presents race analytics and classic model evaluations. Served by the `dashboard_classic` service on port 8501.
- `bigdata_app.py` – Artefact-driven dashboard showcasing Spark/TensorFlow feature stores and metrics. Served by the `dashboard_bigdata` service on port 8502.
- `dashboard_app.py` – Backward-compatible shim that redirects to `classic_app.py`.

## Supporting files
- `Dockerfile` – Builds the shared image for both dashboards.
- `requirements.txt` – Shared dependency list for Streamlit, Plotly, and supporting libraries.

Refer to `docs/runbooks/05-dashboards.md` for startup commands and operational tips.
