"""
mlflow/mlflow_tracking.py
MLflow experiment tracking for Laptop Price Predictor
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "laptop-price-prediction"


def setup_mlflow(tracking_uri: str = TRACKING_URI):
    """Initialize MLflow with tracking server URI"""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"✅ MLflow tracking URI: {tracking_uri}")


def log_data_info(df: pd.DataFrame):
    """Log dataset metadata"""
    mlflow.log_params({
        "data_source": "laptop_data.xlsx",
        "num_rows": len(df),
        "num_features": df.shape[1] - 1,
        "target": "Price (log-transformed)"
    })
    logger.info(f"📊 Logged data info: {len(df)} rows")


def log_preprocessing_steps():
    """Log feature engineering steps applied"""
    mlflow.log_params({
        "ram_cleaning": "extracted GB from string",
        "weight_cleaning": "extracted float from kg string",
        "cpu_encoding": "LabelEncoder (6 classes)",
        "gpu_encoding": "LabelEncoder (4 classes)",
        "ppi_calculated": True,
        "touchscreen_detected": True,
        "ips_detected": True,
        "memory_parsed": "HDD + SSD separately",
        "target_transform": "log(Price)"
    })


def log_model_run(
    model_name: str,
    model,
    X_train, y_train,
    X_test, y_test,
    extra_params: dict = None
):
    """
    Train, evaluate, and log a single model to MLflow.

    Args:
        model_name: Display name for the run
        model: Scikit-learn / XGBoost model
        X_train, y_train: Training data
        X_test, y_test: Test data
        extra_params: Additional hyperparameters to log

    Returns:
        dict: Evaluation metrics
    """
    with mlflow.start_run(run_name=model_name):

        # Log data info once per experiment (tags)
        mlflow.set_tags({
            "model_type": model_name,
            "framework": "xgboost" if "XGBoost" in model_name else "sklearn",
            "dataset": "laptop_data.xlsx"
        })

        # Log hyperparameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            # Truncate to avoid MLflow param-count limits
            mlflow.log_params({k: v for k, v in list(params.items())[:15]})
        if extra_params:
            mlflow.log_params(extra_params)

        # Train
        logger.info(f"🚀 Training {model_name}...")
        model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        metrics = {
            "train_r2":  r2_score(y_train, y_train_pred),
            "test_r2":   r2_score(y_test, y_test_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "test_mae":  mean_absolute_error(y_test, y_test_pred),
        }
        mlflow.log_metrics(metrics)

        # Log model artifact
        if "XGBoost" in model_name:
            mlflow.xgboost.log_model(model, artifact_path=model_name)
        else:
            mlflow.sklearn.log_model(model, artifact_path=model_name)

        logger.info(
            f"✅ {model_name}: "
            f"R²={metrics['test_r2']:.4f} | "
            f"RMSE={metrics['test_rmse']:.4f} | "
            f"MAE={metrics['test_mae']:.4f}"
        )

        return metrics


def log_best_model(best_name: str, best_model, metrics: dict):
    """Register the best model in MLflow Model Registry"""
    with mlflow.start_run(run_name=f"BEST_{best_name}"):
        mlflow.set_tags({
            "best_model": True,
            "model_name": best_name
        })
        mlflow.log_metrics(metrics)

        if "XGBoost" in best_name:
            result = mlflow.xgboost.log_model(
                best_model,
                artifact_path="best_model",
                registered_model_name="LaptopPricePredictor"
            )
        else:
            result = mlflow.sklearn.log_model(
                best_model,
                artifact_path="best_model",
                registered_model_name="LaptopPricePredictor"
            )

        logger.info(f"🏆 Best model '{best_name}' registered in MLflow Model Registry.")
        logger.info(f"   Model URI: {result.model_uri}")

        return result


def get_best_run(metric: str = "test_r2") -> dict:
    """Fetch the best run from MLflow experiment by metric"""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        logger.warning("⚠️ Experiment not found in MLflow.")
        return {}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )

    if not runs:
        return {}

    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "model_name": best_run.data.tags.get("model_type", "Unknown"),
        "test_r2": best_run.data.metrics.get("test_r2"),
        "test_rmse": best_run.data.metrics.get("test_rmse"),
        "test_mae": best_run.data.metrics.get("test_mae"),
    }


def compare_all_runs() -> pd.DataFrame:
    """Return a DataFrame comparing all MLflow runs in the experiment"""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_r2 DESC"]
    )

    records = []
    for run in runs:
        records.append({
            "Model":      run.data.tags.get("model_type", run.info.run_name),
            "train_r2":   run.data.metrics.get("train_r2"),
            "test_r2":    run.data.metrics.get("test_r2"),
            "test_rmse":  run.data.metrics.get("test_rmse"),
            "test_mae":   run.data.metrics.get("test_mae"),
        })

    df = pd.DataFrame(records).dropna()
    return df.sort_values("test_r2", ascending=False).reset_index(drop=True)