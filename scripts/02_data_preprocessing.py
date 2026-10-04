from pathlib import Path

import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def preprocess_data(test_size=0.25, random_state=42):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_registry_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Breast Cancer - Data Preprocessing")

    with mlflow.start_run() as run:
        mlflow.set_tag("ml.step", "data_preprocessing")

        df = load_breast_cancer(as_frame=True).frame

        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        # ยังไม่ scale: StandardScaler จะ fit เฉพาะ train ใน Pipeline ของขั้น 03
        processed_data_dir = Path("processed_data")
        processed_data_dir.mkdir(exist_ok=True)

        pd.concat([X_train, y_train], axis=1).to_csv(
            processed_data_dir / "train.csv",
            index=False,
        )

        pd.concat([X_test, y_test], axis=1).to_csv(
            processed_data_dir / "test.csv",
            index=False,
        )

        mlflow.log_params(
            {
                "test_size": test_size,
                "random_state": random_state,
            }
        )

        mlflow.log_metrics(
            {
                "training_set_rows": len(X_train),
                "test_set_rows": len(X_test),
            }
        )

        mlflow.log_artifacts(
            str(processed_data_dir),
            artifact_path="processed_data",
        )

        print(f"training_set_rows = {len(X_train)}")
        print(f"test_set_rows = {len(X_test)}")
        print("Saved and logged train.csv and test.csv as artifacts.")
        print(f"Preprocessing Run ID: {run.info.run_id}")


if __name__ == "__main__":
    preprocess_data()