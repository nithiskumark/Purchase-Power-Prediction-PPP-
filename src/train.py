"""
Purchasing Power Prediction (PPP)

train.py — End-to-end MLflow + Logistic Regression training script.

Pipeline:
  1. Load & split data 
  2. Hyper-parameter search (GridSearchCV)
  3. Log params / metrics / artefacts / model to MLflow
  4. Register best model in the MLflow Model Registry
"""

import os
import json
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# CONFIG
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
EXPERIMENT_NAME     = "Purchase-power-pred"
MODEL_NAME          = "logreg-purchase-power"
RANDOM_STATE        = 42
TEST_SIZE           = 0.20
CV_FOLDS            = 5

PARAM_GRID = {
    "clf__C":           [0.001, 0.01, 0.1, 1, 10, 100],
    "clf__penalty":     ["l1", "l2"],
    "clf__solver":      ["liblinear", "saga"],
    "clf__max_iter":    [500],
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "purchase_power_c.csv"

# LOAD DATA
def load_data():
    df = pd.read_csv(DATA_PATH)
    drop_cols = ["purchase_power", "purchasing_power", "purchase_class"]
    target = "purchase_class"
    X = df.drop(columns=drop_cols)
    y = df[target]
    log.info("Dataset: %s rows, %s features, target=%s", X.shape[0], X.shape[1], target)
    return X, y

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
    }

def save_confusion_matrix(y_true, y_pred, labels, path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_roc_curve(estimator, X_test, y_test, path="roc_curve.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def register_model(run_id: str, model_uri: str, model_name: str) -> str:
    """Register model and transition best version to Staging."""
    client = MlflowClient()

    # Create registered model if it doesn't exist
    try:
        client.create_registered_model(model_name)
        log.info("Created registered model '%s'", model_name)
    except mlflow.exceptions.MlflowException:
        log.info("Registered model '%s' already exists", model_name)

    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
    )
    log.info("Created model version %s for '%s'", mv.version, model_name)

    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=True,   # archive older Staging versions
    )
    log.info("Transitioned version %s to Staging", mv.version)
    return mv.version

def main():
    # 1. Tracking setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

        # 3. Build pipeline skeleton
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(random_state=RANDOM_STATE)),
    ])

    # 4. Grid search with cross-validation
    log.info("Starting GridSearchCV (%d folds, %d combos)…",
             CV_FOLDS,
             len(PARAM_GRID["clf__C"]) * len(PARAM_GRID["clf__penalty"]) * len(PARAM_GRID["clf__solver"]))

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        return_train_score=True,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_

    # 5. Evaluate
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    log.info("Best params : %s", grid.best_params_)
    log.info("CV ROC-AUC  : %.4f", grid.best_score_)
    log.info("Test metrics: %s", metrics)

    # 6. MLflow run
    with mlflow.start_run(run_name="logreg-gridsearch") as run:
        run_id = run.info.run_id
        log.info("MLflow Run ID: %s", run_id)

        # ── Tags ──────────────────────────────────
        mlflow.set_tags({
            "model_family":  "linear",
            "dataset":       "Purchase power",
            "developer":     os.getenv("USER", "ds-team"),
            "cv_strategy":   f"StratifiedKFold-{CV_FOLDS}",
        })

        # ── Parameters ────────────────────────────
        # Log search space
        mlflow.log_params({
            "test_size":    TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_folds":     CV_FOLDS,
            "param_grid":   json.dumps(PARAM_GRID),
        })
        # Log best params (flatten prefix)
        best_params_clean = {
            k.replace("clf__", ""): v
            for k, v in grid.best_params_.items()
        }
        mlflow.log_params(best_params_clean)

        # ── Metrics ────────────────────────────────
        mlflow.log_metric("cv_roc_auc", grid.best_score_)
        for name, val in metrics.items():
            mlflow.log_metric(f"test_{name}", val)

        # ── Artifacts ─────────────────────────────
        # Classification report as text

        os.makedirs("artifacts", exist_ok=True)

        report_path = "artifacts/classification_report.txt"
        cm_path = "artifacts/confusion_matrix.png"
        roc_path = "artifacts/roc_curve.png"
        cv_csv_path = "artifacts/cv_results.csv"
        feat_path = "artifacts/feature_importance.csv"

        report = classification_report(
            y_test,
            y_pred,
            target_names=["Low Purchase Power", "High Purchase Power"]
        )

        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, "reports")

        # Confusion matrix plot
        cm_path = save_confusion_matrix(
            y_test,
            y_pred,
            ["Low Purchase Power", "High Purchase Power"],
            cm_path,
        )

        mlflow.log_artifact(cm_path, "plots")

        # ROC curve plot
        roc_path = save_roc_curve(best_pipeline, X_test, y_test, roc_path)
        mlflow.log_artifact(roc_path, "plots")

        # CV results as CSV
        cv_df = pd.DataFrame(grid.cv_results_)
        cv_df.to_csv(cv_csv_path, index=False)
        mlflow.log_artifact(cv_csv_path, "reports")

        # Feature importance (coefficients)
        coef = best_pipeline.named_steps["clf"].coef_[0]
        feat_imp = pd.DataFrame({
            "feature":    X.columns,
            "coefficient": coef,
            "abs_coef":   np.abs(coef),
        }).sort_values("abs_coef", ascending=False)
        feat_imp.to_csv(feat_path, index=False)
        mlflow.log_artifact(feat_path, "reports")

        # ── Log model with signature & input example ──
        signature    = infer_signature(X_train, best_pipeline.predict(X_train))
        input_example = X_train.iloc[:5]

        model_info = mlflow.sklearn.log_model(
            sk_model       = best_pipeline,
            artifact_path  = "model",
            signature      = signature,
            input_example  = input_example,
            registered_model_name = None,   # we register manually below
        )
        log.info("Model logged at: %s", model_info.model_uri)

        # ── Register ──────────────────────────────
        version = register_model(run_id, model_info.model_uri, MODEL_NAME)

    print("\n" + "="*60)
    print(f"  Experiment : {EXPERIMENT_NAME}")
    print(f"  Run ID     : {run_id}")
    print(f"  Model name : {MODEL_NAME}  (version {version}, stage=Staging)")
    print(f"  Test metrics:")
    for k, v in metrics.items():
        print(f"    {k:12s}: {v}")
    print("="*60)
    print(f"\n  View UI:  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print(f"  Serve  :  See serve.py or docker/README.md\n")


if __name__ == "__main__":
    main()




































