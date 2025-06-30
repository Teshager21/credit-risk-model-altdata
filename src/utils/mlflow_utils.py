import mlflow
import os


def setup_mlflow(tracking_dir="../../mlruns", experiment_name="credit-risk-modeling"):
    if not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir)
    mlflow.set_tracking_uri(f"file:{tracking_dir}")
    mlflow.set_experiment(experiment_name)


def log_model_mlflow(
    model, model_name, params, metrics, threshold=None, run_name=None, flavor="sklearn"
):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if threshold is not None:
            mlflow.log_param("optimal_threshold", threshold)

        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, model_name)
        elif flavor == "xgboost":
            mlflow.xgboost.log_model(model, model_name)
        else:
            raise ValueError("Unsupported MLflow flavor")

        return run.info.run_id
