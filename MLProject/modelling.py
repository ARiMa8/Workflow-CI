import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = "Credit Scoring Workflow CI - Andi Rizki Mahesa"


def parse_max_depth(value):
    if value is None:
        return None

    value = str(value).strip().lower()
    if value in {"", "none", "null"}:
        return None

    return int(value)


def load_processed_dataset(data_dir):
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")["credit_risk"]
    y_test = pd.read_csv(data_dir / "y_test.csv")["credit_risk"]

    with open(data_dir / "metadata.json", "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    return X_train, X_test, y_train, y_test, metadata


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def save_training_outputs(model, X_train, X_test, y_test, metrics, metadata, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    y_pred = model.predict(X_test)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["bad", "good"],
        output_dict=True,
        zero_division=0,
    )
    with open(reports_dir / "classification_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    confusion = confusion_matrix(y_test, y_pred)
    pd.DataFrame(
        confusion,
        index=["actual_bad", "actual_good"],
        columns=["predicted_bad", "predicted_good"],
    ).to_csv(reports_dir / "confusion_matrix.csv")

    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(reports_dir / "feature_importance.csv", index=False)

    input_example = X_train.head(5)
    signature = infer_signature(input_example, model.predict(input_example))
    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(output_dir / "model"),
        signature=signature,
        input_example=input_example,
    )


def train(args):
    data_dir = (BASE_DIR / args.data_dir).resolve()
    output_dir = (BASE_DIR / args.output_dir).resolve()
    X_train, X_test, y_train, y_test, metadata = load_processed_dataset(data_dir)

    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(str(BASE_DIR / "mlruns"))

    mlflow_project_run_id = os.getenv("MLFLOW_RUN_ID")
    if not mlflow_project_run_id:
        mlflow.set_experiment(EXPERIMENT_NAME)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=parse_max_depth(args.max_depth),
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )

    start_run_kwargs = (
        {"run_id": mlflow_project_run_id}
        if mlflow_project_run_id
        else {"run_name": "credit_scoring_workflow_ci"}
    )

    with mlflow.start_run(**start_run_kwargs) as run:
        mlflow.set_tag("mlflow.runName", "credit_scoring_workflow_ci")
        mlflow.set_tag("student_name", "Andi Rizki Mahesa")
        mlflow.set_tag("dicoding_username", "arima88")
        mlflow.set_tag("workflow", "github_actions")
        mlflow.set_tag("dataset", metadata["dataset_name"])

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("model_n_estimators", args.n_estimators)
        mlflow.log_param("model_max_depth", parse_max_depth(args.max_depth))
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("model_random_state", args.random_state)
        mlflow.log_param("target_column", metadata["target_column"])
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])
        mlflow.log_param("feature_count", X_train.shape[1])

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        save_training_outputs(model, X_train, X_test, y_test, metrics, metadata, output_dir)
        mlflow.log_artifacts(str(output_dir), artifact_path="model_output")

        print(f"Run ID: {run.info.run_id}")
        print(f"Output directory: {output_dir}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="credit_scoring_preprocessing")
    parser.add_argument("--output-dir", default="model_output")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", default="none")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
