"""
mlflow_helper.py
================
All MLflow utility functions for the Hand Gesture Classification project.
Import this module in the notebook to manage experiment tracking.

Usage in notebook:
    from mlflow_helper import *
    setup_experiment()
    ...
    with start_run("RandomForest_n200_pca35") as run:
        log_dataset_info(df)
        log_preprocessing_params(...)
        log_model_params(name, model)
        train_and_log_run(name, model, X_train_pca, X_test_pca, y_train, y_test, le)
        register_best_model(best_name, best_run_id)
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import time
import os
import tempfile

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "HaGRID_Hand_Gesture_Classification"
REGISTRY_NAME   = "HaGRID_BestGestureClassifier"


# ──────────────────────────────────────────────────────────────
# 1. Experiment setup
# ──────────────────────────────────────────────────────────────
def setup_experiment(tracking_uri: str = "mlruns") -> str:
    """
    Point MLflow at a local mlruns folder and create (or retrieve)
    the experiment.  Returns the experiment_id.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"[MLflow] Experiment  : {EXPERIMENT_NAME}")
    print(f"[MLflow] Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"[MLflow] Experiment ID: {exp.experiment_id}")
    return exp.experiment_id


# ──────────────────────────────────────────────────────────────
# 2. Start a named run
# ──────────────────────────────────────────────────────────────
def start_run(run_name: str):
    """
    Context-manager wrapper — use with `with start_run(...) as run:`.
    run_name should be descriptive, e.g. 'SVM_C0.24_linear_pca35'
    """
    return mlflow.start_run(run_name=run_name)


# ──────────────────────────────────────────────────────────────
# 3. Log dataset information
# ──────────────────────────────────────────────────────────────
def log_dataset_info(df: pd.DataFrame, csv_path: str = "hand_landmarks_data.csv") -> None:
    """
    Log key dataset statistics as MLflow params/tags.
    Also logs the CSV as a dataset artifact if the file exists.
    """
    mlflow.set_tag("dataset.name",    "HaGRID Hand Gesture Dataset")
    mlflow.set_tag("dataset.source",  csv_path)
    mlflow.log_param("dataset.n_samples",  len(df))
    mlflow.log_param("dataset.n_classes",  df["label"].nunique())
    mlflow.log_param("dataset.raw_features", len(df.columns) - 1)  # excluding label
    mlflow.log_param("dataset.missing_values", int(df.isnull().sum().sum()))

    # Log class distribution as artifact
    class_dist = df["label"].value_counts().reset_index()
    class_dist.columns = ["gesture", "count"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "class_distribution.csv")
        class_dist.to_csv(path, index=False)
        mlflow.log_artifact(path, artifact_path="dataset")

    # Log the CSV itself if it exists locally
    if os.path.exists(csv_path):
        mlflow.log_artifact(csv_path, artifact_path="dataset")

    print(f"[MLflow] Dataset info logged  ({len(df)} samples, {df['label'].nunique()} classes)")


# ──────────────────────────────────────────────────────────────
# 4. Log preprocessing parameters
# ──────────────────────────────────────────────────────────────
def log_preprocessing_params(
    test_size: float = 0.20,
    random_state: int = 42,
    n_pca_components: int = 35,
    engineered_features: int = 231,
    feature_description: str = "210 pairwise_distances + 21 y_direction_signs"
) -> None:
    """Log preprocessing / pipeline hyperparameters."""
    mlflow.log_param("preprocessing.test_size",            test_size)
    mlflow.log_param("preprocessing.random_state",         random_state)
    mlflow.log_param("preprocessing.scaler",               "StandardScaler")
    mlflow.log_param("preprocessing.pca_components",       n_pca_components)
    mlflow.log_param("preprocessing.engineered_features",  engineered_features)
    mlflow.log_param("preprocessing.feature_description",  feature_description)
    mlflow.set_tag("pipeline.stratified_split", "True")
    print("[MLflow] Preprocessing params logged")


# ──────────────────────────────────────────────────────────────
# 5. Log model hyperparameters
# ──────────────────────────────────────────────────────────────
def log_model_params(model_name: str, model) -> None:
    """
    Log model-specific hyperparameters based on model type.
    model_name: e.g. 'Random Forest', 'SVM', 'KNN', 'XGBoost'
    """
    mlflow.set_tag("model.algorithm", model_name)

    params = model.get_params()
    # Only log a curated subset to keep the UI clean
    key_params = {
        "Random Forest": ["n_estimators", "max_depth", "min_samples_split",
                          "min_samples_leaf", "random_state"],
        "SVM":           ["C", "kernel", "probability", "random_state"],
        "KNN":           ["n_neighbors", "metric", "weights"],
        "XGBoost":       ["n_estimators", "max_depth", "learning_rate",
                          "subsample", "colsample_bytree", "random_state"],
    }
    selected = key_params.get(model_name, list(params.keys()))
    for k in selected:
        if k in params and params[k] is not None:
            mlflow.log_param(f"model.{k}", params[k])

    print(f"[MLflow] Model params logged for {model_name}")


# ──────────────────────────────────────────────────────────────
# 6. Log metrics
# ──────────────────────────────────────────────────────────────
def log_metrics(y_true, y_pred, elapsed_time: float) -> dict:
    """
    Compute and log accuracy, precision, recall, F1, and training time.
    Returns a dict of metric values.
    """
    metrics = {
        "accuracy":           accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted":    recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted"),
        "training_time_sec":  elapsed_time,
    }
    mlflow.log_metrics(metrics)
    print(f"[MLflow] Metrics logged  →  Accuracy={metrics['accuracy']*100:.2f}%  "
          f"F1={metrics['f1_weighted']*100:.2f}%  Time={elapsed_time:.1f}s")
    return metrics


# ──────────────────────────────────────────────────────────────
# 7. Log confusion matrix as artifact
# ──────────────────────────────────────────────────────────────
def log_confusion_matrix(y_true, y_pred, class_names, model_name: str) -> None:
    """Save confusion matrix heatmap and log it as an artifact."""
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(cm_pct, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.4, annot_kws={"size": 7}, ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}  (% of true class)", fontsize=13)
    ax.set_xlabel("Predicted Gesture")
    ax.set_ylabel("True Gesture")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
        plt.savefig(path, dpi=120)
        mlflow.log_artifact(path, artifact_path="confusion_matrices")
    plt.close()
    print(f"[MLflow] Confusion matrix artifact logged for {model_name}")


# ──────────────────────────────────────────────────────────────
# 8. Log classification report as artifact
# ──────────────────────────────────────────────────────────────
def log_classification_report(y_true, y_pred, class_names, model_name: str) -> None:
    """Save and log the per-class classification report as a text artifact."""
    report = classification_report(y_true, y_pred, target_names=class_names)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"classification_report_{model_name.replace(' ', '_')}.txt")
        with open(path, "w") as f:
            f.write(f"Classification Report — {model_name}\n\n")
            f.write(report)
        mlflow.log_artifact(path, artifact_path="reports")
    print(f"[MLflow] Classification report artifact logged for {model_name}")


# ──────────────────────────────────────────────────────────────
# 9. Log the trained model
# ──────────────────────────────────────────────────────────────
def log_model(model, model_name: str, X_sample) -> None:
    """
    Log the serialized model to the run.
    Handles sklearn and xgboost separately.
    """
    signature = infer_signature(X_sample, model.predict(X_sample))
    artifact_path = f"model_{model_name.replace(' ', '_')}"

    if model_name == "XGBoost":
        mlflow.xgboost.log_model(
            model,
            artifact_path=artifact_path,
            signature=signature
        )
    else:
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            signature=signature
        )
    print(f"[MLflow] Model artifact logged at '{artifact_path}'")


# ──────────────────────────────────────────────────────────────
# 10. Full train-and-log workflow (call inside a `with start_run(...)`)
# ──────────────────────────────────────────────────────────────
def train_and_log_run(
    model_name: str,
    model,
    X_train_pca, X_test_pca,
    y_train, y_test,
    le,          # LabelEncoder with .classes_
    log_artifact_model: bool = True
) -> dict:
    """
    Train `model`, then log params, metrics, artifacts, and optionally
    the serialized model — all inside the *active* MLflow run.

    Returns a dict with y_pred + all metrics.
    """
    # ── Train
    print(f"[MLflow] Training {model_name}...")
    start   = time.time()
    model.fit(X_train_pca, y_train)
    elapsed = time.time() - start
    y_pred  = model.predict(X_test_pca)

    # ── Log
    log_model_params(model_name, model)
    metrics = log_metrics(y_test, y_pred, elapsed)
    log_confusion_matrix(y_test, y_pred, le.classes_, model_name)
    log_classification_report(y_test, y_pred, le.classes_, model_name)

    if log_artifact_model:
        log_model(model, model_name, X_test_pca[:5])

    return {**metrics, "y_pred": y_pred, "model": model}


# ──────────────────────────────────────────────────────────────
# 11. Comparison chart — log after all runs
# ──────────────────────────────────────────────────────────────
def log_comparison_chart(results: dict) -> None:
    """
    Create and log a grouped bar chart comparing all four models
    across Accuracy, Precision, Recall, and F1.
    `results` is the dict produced by the notebook:
        { model_name: { 'accuracy':..., 'precision':..., ... } }
    """
    models_  = list(results.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    x      = np.arange(len(models_))
    width  = 0.18
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        vals = [results[m].get(key, results[m].get(f"{key}_weighted", 0)) * 100
                for m in models_]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models_, fontsize=11)
    ax.set_ylim(90, 102)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Comparison — HaGRID Hand Gesture Classification", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model_comparison_chart.png")
        plt.savefig(path, dpi=150)
        mlflow.log_artifact(path, artifact_path="comparison")
    plt.show()
    plt.close()
    print("[MLflow] Comparison chart artifact logged")


# ──────────────────────────────────────────────────────────────
# 12. Register best model in the Model Registry
# ──────────────────────────────────────────────────────────────
def register_best_model(best_model_name: str, run_id: str) -> None:
    """
    Register the best model's artifact in the MLflow Model Registry
    under REGISTRY_NAME and add a description + alias.
    """
    artifact_path = f"model_{best_model_name.replace(' ', '_')}"
    model_uri     = f"runs:/{run_id}/{artifact_path}"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTRY_NAME
    )

    client = MlflowClient()
    client.update_model_version(
        name=REGISTRY_NAME,
        version=result.version,
        description=(
            f"Best model: {best_model_name}. "
            "Trained on HaGRID dataset (14 421 samples, 18 gesture classes). "
            "Feature engineering: 231 geometric features (pairwise distances + Y-direction signs). "
            "Preprocessing: StandardScaler → PCA(35 components)."
        )
    )
    client.set_registered_model_alias(
        name=REGISTRY_NAME,
        alias="champion",
        version=result.version
    )

    print(f"[MLflow] Model registered as '{REGISTRY_NAME}' v{result.version} "
          f"(alias: 'champion')  →  {model_uri}")


# ──────────────────────────────────────────────────────────────
# 13. Convenience: get run_id of the best run in the experiment
# ──────────────────────────────────────────────────────────────
def get_best_run_id(metric: str = "f1_weighted") -> tuple[str, str]:
    """
    Query the experiment and return (run_id, model_name_tag) of the
    run with the highest value of `metric`.
    """
    client = MlflowClient()
    exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs   = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    if not runs:
        raise RuntimeError("No runs found in experiment.")
    best = runs[0]
    model_tag = best.data.tags.get("model.algorithm", "Unknown")
    print(f"[MLflow] Best run: {best.info.run_name}  "
          f"({metric}={best.data.metrics[metric]*100:.2f}%,  id={best.info.run_id})")
    return best.info.run_id, model_tag
