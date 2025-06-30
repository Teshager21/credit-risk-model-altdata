import mlflow
import os

# Find absolute path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
mlruns_path = os.path.join(project_root, "mlruns")


def setup_mlflow(tracking_dir=mlruns_path, experiment_name="credit-risk-modeling"):
    if not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir)
    mlflow.set_tracking_uri(f"file:{tracking_dir}")
    mlflow.set_experiment(experiment_name)


def log_model_mlflow(
    model,
    model_name,
    params,
    metrics,
    threshold=None,
    run_name=None,
    flavor="sklearn",
    register=False,
    stage=None,
):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if threshold is not None:
            mlflow.log_param("optimal_threshold", threshold)

        # Save under the run
        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, model_name)
        elif flavor == "xgboost":
            mlflow.xgboost.log_model(model, model_name)
        else:
            raise ValueError("Unsupported MLflow flavor")

        model_uri = f"runs:/{run.info.run_id}/{model_name}"

        # Register in Model Registry if desired
        if register:
            mv = mlflow.register_model(model_uri=model_uri, name=model_name)
            print(f"Registered model '{model_name}' as version {mv.version}")

            if stage:
                client = mlflow.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage=stage
                )
                print(f"Moved model version {mv.version} to stage '{stage}'")

        return run.info.run_id
