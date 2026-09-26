# mlops/

MLOps runtime state and orchestration for this project, kept out of
the application code under `rag/`.

| Path | What it is |
|---|---|
| `mlflow.db` | MLflow's local SQLite tracking store — params/metrics/tags for every run of `helper_scripts/run_experiment.py` (`make experiment` / `make experiments-all`). Gitignored; rebuilt by running experiments. |
| `mlruns/` | MLflow run artifacts (resolved config, per-question retrieval results) referenced by `mlflow.db`. Gitignored. |
| `airflow/dags/` | Airflow DAG definitions for scheduling recurring pipeline jobs (e.g. hyperparameter sweeps). Tracked in git — these are source, not runtime output. |

View tracked experiments with `make mlflow-ui` (equivalent to
`mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db`).
