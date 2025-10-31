# Pipelines Overview (Steps 03 & 04)

The `pipelines` folder splits into two execution tracks:

1. **Classic (`pipelines/classic/`)** – lightweight scikit-learn and XGBoost trainers that read/write Postgres tables.
2. **Big Data (`pipelines/bigdata/`)** – Spark feature engineering and TensorFlow training jobs that operate on distributed artefacts.

Consult the corresponding READMEs in each subfolder for prerequisites, commands, and output locations. Ordered runbooks live under `docs/runbooks/`:
- `03-classic-ml-workflow.md`
- `04-bigdata-ml-workflow.md`
