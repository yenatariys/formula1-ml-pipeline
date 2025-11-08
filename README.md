# Formula 1 ML Pipeline

This repository orchestrates an end-to-end Formula 1 analytics workflow covering ETL, classic machine-learning models, big data pipelines (Spark + TensorFlow), and dual dashboards. The folder layout and numbered runbooks are designed so newcomers can follow the execution order without hunting through the codebase.


## Project layout (execution order)

| Step | Folder | Purpose |
| --- | --- | --- |
| 01 | `docs/runbooks/` | Numbered guides for environment setup, ETL, ML pipelines, and dashboards. |
| 02 | `etl/` | ETL scripts for transforming and loading race, driver, and results data. |
| 03 | `pipelines/classic/` | Scikit-learn trainers for classic ML workflows. |
| 04 | `pipelines/bigdata/` | Spark feature engineering and TensorFlow training for big data ML. |
| 05 | `dashboard/` | Streamlit apps: `classic_app.py`, `bigdata_app.py`, and `unified_app.py` (Unified Dashboard). |
| — | `ml/` | Legacy ML entrypoints (delegating to classic pipeline code). |
| — | `docs/reference/` | Documentation and design notes. |
| — | `artifacts/` | Output directory for feature stores, models, and evaluation metrics. |
| — | `archive/unused/` | Legacy scripts and guides for reference. |

## Quick start

1. Follow `docs/runbooks/01-environment-setup.md` to build images and start infrastructure.
2. Run `docs/runbooks/02-data-ingest-etl.md` to populate Postgres.
3. Choose between the classic (`docs/runbooks/03-classic-ml-workflow.md`) or big data (`docs/runbooks/04-bigdata-ml-workflow.md`) pipelines—or execute both.
4. Launch dashboards using `docs/runbooks/05-dashboards.md` to explore results.

## Architecture diagram

```mermaid
flowchart TB
    subgraph Input ["📥 Data Sources"]
        Raw[("Raw CSV Files<br/>data/*.csv")]

    ## Architecture diagram

    ```mermaid
    flowchart TD
        CSV[Raw CSV Files] --> ETL[ETL Pipeline]
        ETL --> POSTGRES[(PostgreSQL)]
        ETL --> SPARK[Spark Cluster]
        ETL --> NEO4J[(Neo4j)]

        %% Feature Engineering
        POSTGRES --> FE_CLASSIC[Feature Engineering (Classic ML)]
        SPARK --> FE_BIGDATA[Feature Engineering (Big Data)]
        FE_CLASSIC --> FEATURE_STORE[(Feature Store)]
        FE_BIGDATA --> FEATURE_STORE

        %% ML Training
        FEATURE_STORE --> ML_CLASSIC[Classic ML]
        FEATURE_STORE --> ML_SPARK[Spark MLlib]
        FEATURE_STORE --> ML_TF[TensorFlow]

        %% Dashboards group
        subgraph DASHBOARDS[Dashboards]
            DASH_C[Classic Dashboard]
            DASH_B[BigData Dashboard]
            DASH_U[Unified Dashboard]
        end
        FEATURE_STORE --> DASHBOARDS
        ML_CLASSIC --> DASHBOARDS
        ML_SPARK --> DASHBOARDS
        ML_TF --> DASHBOARDS
        NEO4J --> DASHBOARDS

        %% Admin UIs
        POSTGRES --> ADMIN[pgAdmin]
        NEO4J --> NEO4J_UI[Neo4j Browser]

        %% Users
        USER[Users] -.-> DASHBOARDS
        USER -.-> ADMIN
        USER -.-> NEO4J_UI

        classDef storage fill:#E1F5FE,stroke:#0277BD,stroke-width:2px
        classDef processing fill:#FFF9C4,stroke:#F57F17,stroke-width:2px
        classDef feature fill:#FFFDE7,stroke:#FBC02D,stroke-width:2px
        classDef ml fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
        classDef dashboards fill:#FCE4EC,stroke:#C2185B,stroke-width:2px
        classDef admin fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px
        classDef users fill:#EEEEEE,stroke:#616161,stroke-width:2px

        class CSV,POSTGRES,FEATURE_STORE,NEO4J storage
        class ETL,SPARK processing
        class FE_CLASSIC,FE_BIGDATA feature
        class ML_CLASSIC,ML_SPARK,ML_TF ml
        class DASHBOARDS dashboards
        class ADMIN,NEO4J_UI admin
        class USER users
    ```

| Service | Description |
| --- | --- |
| `postgres` | Primary datastore for ETL outputs and classic ML predictions. |
| `etl` | One-shot container that executes the ETL pipeline. |
| `ml_train` | Runs the classic ML trainers. |
| `spark-master`, `spark-worker-*` | Spark cluster backing the big data feature engineering job. |
| `dashboard_classic` | Streamlit app on port 8501 reading Postgres tables. |
| `dashboard_bigdata` | Streamlit app on port 8502 visualising artefacts in `artifacts/`. |
| `pgadmin` | Optional Postgres UI for manual inspection. |
| `neo4j` | Optional knowledge graph store backing the graph analytics export. |

## Architecture Documentation

Comprehensive architecture diagrams and documentation are available in the `docs/` folder:

- **[Star Schema Design](docs/star_schema.md)** - Dimensional data model with fact/dimension tables optimized for analytics and ML feature engineering
- **[IT Infrastructure Architecture](docs/infrastructure_architecture.md)** - Docker Compose service architecture, network topology, data flows, and deployment guide
- **[Architecture Overview](docs/README.md)** - Quick reference and integration guide for both diagrams

These documents include:
- Mermaid diagrams (auto-rendered on GitHub)
- Detailed component descriptions
- SQL query patterns for ML features
- Scaling and security recommendations
- Monitoring and disaster recovery strategies

## Contribution tips

- Keep new documentation in the appropriate numbered runbook or reference folder so the execution order remains obvious.
- Whenever you introduce a new pipeline stage, describe its inputs/outputs in the relevant README and update the root table above.
- Use the `artifacts/` folder (or override via env vars) for outputs so dashboards and collaborators can locate results automatically.
- Update architecture diagrams in `docs/` when modifying database schema or infrastructure services.
