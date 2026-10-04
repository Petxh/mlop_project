import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ACCURACY_THRESHOLD = 0.95
ROC_AUC_THRESHOLD = 0.98
MODEL_NAME = "cancer-classifier-prod"


def train_evaluate_register(preprocessing_run_id, C=1.0):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_registry_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Breast Cancer - Model Training")

    with mlflow.start_run(run_name=f"logistic_regression_C_{C}"):
        mlflow.set_tag("ml.step", "model_training_evaluation")

        mlflow.log_params(
            {
                "preprocessing_run_id": preprocessing_run_id,
                "C": C,
                "accuracy_threshold": ACCURACY_THRESHOLD,
                "roc_auc_threshold": ROC_AUC_THRESHOLD,
            }
        )

        print(f"Starting training run with C={C}...")

        # ดึง CSV จาก run ที่เลือกจริง
        # ไม่โหลดและแบ่งข้อมูลใหม่ในขั้นเทรน
        local_dir = Path(
            download_artifacts(
                run_id=preprocessing_run_id,
                artifact_path="processed_data",
            )
        )

        train_df = pd.read_csv(local_dir / "train.csv")
        test_df = pd.read_csv(local_dir / "test.csv")

        X_train = train_df.drop("target", axis=1)
        y_train = train_df["target"]

        X_test = test_df.drop("target", axis=1)
        y_test = test_df["target"]

        print(
            f"Loaded artifacts: {len(X_train)} train rows, "
            f"{len(X_test)} test rows"
        )

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=C,
                        random_state=42,
                        max_iter=10000,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        # classes_ เป็น [0, 1]; คอลัมน์ 1 คือ P(target=1) หรือ P(benign)
        # ROC-AUC ต้องใช้คะแนน/ความน่าจะเป็น ไม่ใช่ y_pred ที่เป็น 0/1
        y_score = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_score)

        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        mlflow.log_metrics(
            {
                "accuracy": acc,
                "roc_auc": roc_auc,
            }
        )

        # เก็บทั้ง scaler และ classifier พร้อม signature จาก input_example
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="cancer_classifier_pipeline",
            input_example=X_train.head(5),
            serialization_format="cloudpickle",
        )

        gate_passed = (
            acc >= ACCURACY_THRESHOLD
            and roc_auc >= ROC_AUC_THRESHOLD
        )

        mlflow.set_tag(
            "gate.status",
            "Passed" if gate_passed else "Failed",
        )

        if gate_passed:
            registered_model = mlflow.register_model(
                model_info.model_uri,
                MODEL_NAME,
            )

            MlflowClient().set_registered_model_alias(
                name=MODEL_NAME,
                alias="staging",
                version=registered_model.version,
            )

            print(
                f"Model registered as '{MODEL_NAME}' "
                f"version {registered_model.version}"
            )
            print(
                f"Set alias '@staging' -> {MODEL_NAME} "
                f"version {registered_model.version}"
            )
        else:
            print(
                "Model failed the quality gate. "
                "Not registering; alias is unchanged."
            )

        print("Training run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and register a cancer classifier."
    )

    parser.add_argument(
        "preprocessing_run_id",
        help="Run ID printed by script 02.",
    )

    parser.add_argument(
        "C",
        type=float,
        nargs="?",
        default=1.0,
    )

    args = parser.parse_args()

    if args.C <= 0:
        parser.error("C must be greater than zero")

    train_evaluate_register(
        args.preprocessing_run_id,
        args.C,
    )