# ML Container Entrypoints (Compatibility Layer)

The `ml` directory provides Docker-friendly entrypoints retained for backward compatibility. The actual training logic now lives in `pipelines/classic/`.

## Files
- `Dockerfile` – builds the `ml_train` image used by `docker-compose`.
- `train.py`, `train_comparison.py`, `train_with_best_params.py` – legacy scripts that import and delegate to the updated pipelines.

## Recommended workflow
Run the modern trainers directly from `pipelines/classic/` or through the `ml_train` service as described in `docs/runbooks/03-classic-ml-workflow.md`. Keep these wrappers untouched unless you need to modify container startup behaviour.
