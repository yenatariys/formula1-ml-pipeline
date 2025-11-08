# Formula 1 ML Pipeline - Architecture Documentation

## Overview
This directory contains comprehensive architecture documentation for the Formula 1 Machine Learning Pipeline project, including data models and IT infrastructure diagrams.

## Documentation Index

### 📊 [Star Schema Design](./star_schema.md)
**Purpose**: Minimal dimensional model built from the CSVs currently used by the ETL (`races.csv`, `results.csv`, `drivers.csv`)

**Contents**:
- Compact star schema ERD (1 fact table, 2 dimensions)
- Column breakdown for each table
- Example analytical queries
- Guidance for extending the model as more sources are added

**Key Highlights**:
- Mirrors the joins in `etl/extract_data.py`
- Easy to implement in PostgreSQL or a warehouse
- Provides a clean base for dashboards and ML feature engineering
- Designed to expand when additional CSVs enter the pipeline

**Use Cases**:
- Database schema design and migration
- Feature engineering for ML pipelines
- Dashboard query optimization
- Data quality validation

---

### 🏗️ [Infrastructure Architecture](./infrastructure_architecture.md)
**Purpose**: Containerized IT infrastructure and service orchestration

**Contents**:
- Docker Compose service architecture
- Component details (databases, Spark cluster, ML services, dashboards)
- Network topology and port mappings
- Data flow pipelines
- Deployment and operations guide

**Key Highlights**:
- 11 Docker services across 4 layers (storage, processing, ML, visualization)
- Spark cluster with master + 2 workers (4 cores, 6GB RAM)
- PostgreSQL + Neo4j dual database architecture
- Streamlit dashboards for classic ML and big data workflows
- Shared volume mount for artifact storage

**Use Cases**:
- DevOps setup and deployment
- Service monitoring and troubleshooting
- Scaling strategies for production
- Security hardening checklist

---

## Quick Reference

### Star Schema Tables
| Type | Table Name | Description |
|------|-----------|-------------|
| **Fact** | FACT_RACE_RESULTS | One row per driver per race with position, grid slot, and points |
| **Dim** | DIM_RACE | Race metadata (season year, round, race name) |
| **Dim** | DIM_DRIVER | Driver attributes (name, nationality, code, number) |

### Infrastructure Services
| Service | Container | Ports | Purpose |
|---------|-----------|-------|---------|
| **PostgreSQL** | f1_postgres | 5432 | Relational database |
| **Neo4j** | f1_neo4j | 7474, 7687 | Graph database |
| **Spark Master** | spark-master | 7077, 8080 | Cluster manager + UI |
| **Spark Worker 1** | spark-worker-1 | 8081 | Processing node |
| **Spark Worker 2** | spark-worker-2 | 8082 | Processing node |
| **ETL** | etl_service | 8888 | Data pipeline |
| **ML Training** | ml_train | - | Scikit-learn models |
| **Classic Dashboard** | dashboard_classic | 8501 | Streamlit UI |
| **Big Data Dashboard** | dashboard_bigdata | 8502 | Streamlit UI |
| **pgAdmin** | f1_pgadmin | 5050 | DB admin tool |

## Architecture Diagrams Preview

### Star Schema Entity Relationship Diagram
The star schema consists of a **single fact table** (`FACT_RACE_RESULTS`) joined to two dimensions (`DIM_RACE`, `DIM_DRIVER`). It captures exactly the data produced by the current extract step and keeps joins easy to reason about.

*See [star_schema.md](./star_schema.md) for the full Mermaid ERD*

### IT Infrastructure Flow Diagram
The infrastructure spans 4 layers:
1. **Data Storage**: PostgreSQL, Neo4j, shared volumes
2. **ETL & Processing**: Spark cluster (1 master + 2 workers), ETL service
3. **ML Training**: Classic ML, Spark MLlib, TensorFlow
4. **Visualization**: Streamlit dashboards, pgAdmin, Neo4j Browser

*See [infrastructure_architecture.md](./infrastructure_architecture.md) for the full Mermaid diagram*

## Integration Between Diagrams

### Data Flow: Star Schema → Infrastructure
```
CSV Files (data/)
    ↓
ETL Service (extract, transform)
    ↓
PostgreSQL (star schema tables)
    ↓
Spark MLlib (feature engineering from dimensions)
    ↓
Feature Store (artifacts/feature_store/)
    ↓
TensorFlow Training (ML models)
    ↓
Dashboards (visualization)
```

### Use Case: Driver Win Prediction ML Pipeline

**Star Schema Perspective**:
1. Query `FACT_RACE_RESULTS` joined with `DIM_DRIVER` and `DIM_RACE`
2. Aggregate metrics such as total points or average finishing position
3. Publish the results as views or materialized tables for downstream ML jobs

**Infrastructure Perspective**:
1. **ETL Service** loads CSV → PostgreSQL star schema
2. **Spark Master** reads `f1_results_joined.csv` (alternative to DB)
3. **Spark Workers** compute rolling window features in parallel
4. **Spark MLlib** trains Random Forest on feature dataframe
5. **Artifact Store** receives Parquet/CSV feature shards
6. **TensorFlow** reads feature store, trains neural network
7. **Dashboard** displays model metrics from artifact JSON files

## Future Enhancements

### Star Schema
- [ ] Implement slowly changing dimensions (SCD Type 2) for driver/constructor history
- [ ] Add junk dimensions for categorical attributes
- [ ] Create aggregate fact tables for common dashboard queries
- [ ] Build bridge tables for many-to-many relationships (e.g., driver contracts)

### Infrastructure
- [ ] Add Redis for caching frequently accessed data
- [ ] Implement Kafka for real-time race data streaming
- [ ] Deploy Airflow for workflow orchestration
- [ ] Set up MLflow for experiment tracking
- [ ] Add Prometheus + Grafana for monitoring
- [ ] Migrate to Kubernetes for production scale

## Related Documentation

- [ETL Pipeline README](../etl/README.md) - Data ingestion and transformation
- [ML Pipeline README](../ml/README.md) - Model training workflows
- [Dashboard README](../dashboard/README.md) - Visualization guides
- [Runbook: Big Data ML Workflow](./runbooks/04-bigdata-ml-workflow.md) - Step-by-step execution

## Viewing Diagrams

### Mermaid Rendering
Both diagrams use [Mermaid](https://mermaid.js.org/) syntax for rendering:

- **GitHub**: Automatically rendered in markdown preview
- **VS Code**: Install [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) extension
- **Web**: Use [Mermaid Live Editor](https://mermaid.live/)
- **Documentation Sites**: Works natively in GitBook, MkDocs, Docusaurus

### Exporting Diagrams
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Export star schema diagram
mmdc -i docs/star_schema.md -o docs/figures/star_schema.png

# Export infrastructure diagram
mmdc -i docs/infrastructure_architecture.md -o docs/figures/infrastructure.png
```

## Contributing

When updating architecture documentation:

1. **Star Schema Changes**: Update `star_schema.md` when modifying:
   - Database table definitions
   - Fact/dimension relationships
   - ETL transformation logic
   - Feature engineering queries

2. **Infrastructure Changes**: Update `infrastructure_architecture.md` when modifying:
   - docker-compose.yml service definitions
   - Port mappings or network configuration
   - Volume mounts or persistent storage
   - Service dependencies or startup order

3. **Diagram Updates**: Ensure Mermaid syntax is valid:
   ```bash
   # Validate diagram syntax
   mmdc -i docs/star_schema.md -o /dev/null
   ```

4. **Version Control**: Commit diagram changes with descriptive messages:
   ```bash
   git add docs/star_schema.md docs/infrastructure_architecture.md
   git commit -m "docs: update star schema to include sprint_results fact table"
   ```

---

**Last Updated**: 2025-11-04  
**Maintained By**: Formula 1 ML Pipeline Team  
**Documentation Standard**: Mermaid Diagrams + Markdown
