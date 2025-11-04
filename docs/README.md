# Formula 1 ML Pipeline - Architecture Documentation

## Overview
This directory contains comprehensive architecture documentation for the Formula 1 Machine Learning Pipeline project, including data models and IT infrastructure diagrams.

## Documentation Index

### 📊 [Star Schema Design](./star_schema.md)
**Purpose**: Dimensional data model for analytics and ML feature engineering

**Contents**:
- Star schema ERD with fact and dimension tables
- Table definitions and relationships
- SQL query patterns for ML features
- Migration path from current flat schema
- Performance optimization strategies

**Key Highlights**:
- 7 fact tables (race results, qualifying, lap times, pit stops, sprint results, standings)
- 7 dimension tables (drivers, constructors, circuits, races, seasons, status, date)
- Optimized for Spark feature engineering and TensorFlow training
- Supports temporal analysis and graph analytics integration

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
| **Fact** | FACT_RACE_RESULTS | Core race outcomes and performance metrics |
| **Fact** | FACT_QUALIFYING | Qualifying session times (Q1, Q2, Q3) |
| **Fact** | FACT_LAP_TIMES | Lap-by-lap performance data |
| **Fact** | FACT_PIT_STOPS | Pit stop events and durations |
| **Fact** | FACT_SPRINT_RESULTS | Sprint race outcomes |
| **Fact** | FACT_CONSTRUCTOR_STANDINGS | Team championship standings |
| **Fact** | FACT_DRIVER_STANDINGS | Driver championship standings |
| **Dim** | DIM_DRIVERS | Driver master data |
| **Dim** | DIM_CONSTRUCTORS | Constructor/team information |
| **Dim** | DIM_CIRCUITS | Circuit details with coordinates |
| **Dim** | DIM_RACES | Race event metadata |
| **Dim** | DIM_SEASONS | Season/year dimension |
| **Dim** | DIM_STATUS | Race finish status codes |
| **Dim** | DIM_DATE | Date dimension for time-based analysis |

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
The star schema consists of:
- **Central Fact Tables**: Race results, qualifying, lap times, pit stops, sprint results, standings
- **Surrounding Dimensions**: Drivers, constructors, circuits, races, seasons, status, date
- **Snowflake Elements**: Races dimension connects to circuits and seasons
- **Optimized Joins**: Minimized join complexity for analytical queries

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
1. Query `FACT_RACE_RESULTS` joined with `DIM_DRIVERS`, `DIM_RACES`
2. Aggregate rolling statistics (win rate, avg points, races completed)
3. Create `FACT_DRIVER_PERFORMANCE` materialized view

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
