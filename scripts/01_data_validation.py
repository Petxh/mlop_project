import argparse

import mlflow
from sklearn.datasets import load_breast_cancer

MIN_CLASS_BALANCE = 0.20
TRACKING_URI = "sqlite:///mlflow.db"


def validate_data(min_class_balance=MIN_CLASS_BALANCE):
    # ทุกสคริปต์และ UI ต้องใช้ฐานข้อมูลเดียวกัน และรันจาก root ของโปรเจกต์
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment("Breast Cancer - Data Validation")

    with mlflow.start_run():
        mlflow.set_tag("ml.step", "data_validation")

        df = load_breast_cancer(as_frame=True).frame

        num_rows, num_cols = df.shape
        num_classes = df["target"].nunique()
        missing_values = int(df.isnull().sum().sum())

        # ใช้สัดส่วนของคลาสที่มีแถวน้อยที่สุด ไม่ใช่จำนวนแถวของคลาสนั้น
        class_balance = float(
            df["target"].value_counts(normalize=True).min()
        )

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")
        print(f"Class balance: {class_balance:.4f}")
        print(f"Class balance threshold: {min_class_balance:.4f}")

        mlflow.log_metrics(
            {
                "num_rows": num_rows,
                "num_cols": num_cols,
                "missing_values": missing_values,
                "class_balance": class_balance,
            }
        )

        mlflow.log_params(
            {
                "num_classes": num_classes,
                "min_class_balance": min_class_balance,
            }
        )

        # ข้อมูลต้องมี 2 คลาสพอดี และ class balance ต้องไม่น้อยกว่า threshold
        failed = (
            missing_values > 0
            or num_classes != 2
            or class_balance < min_class_balance
        )

        validation_status = "Failed" if failed else "Success"

        mlflow.log_param("validation_status", validation_status)

        print(f"Validation status: {validation_status}")

        if failed:
            # การ print อย่างเดียวไม่ทำให้ GitHub Actions ขึ้นแดง
            raise SystemExit("Data validation failed: stopping the pipeline.")

        print("Data validation run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the Breast Cancer dataset."
    )

    parser.add_argument(
        "--min-class-balance",
        type=float,
        default=MIN_CLASS_BALANCE,
        help="Minimum class fraction, default 0.20; use 0.45 for Capture 2.2.",
    )

    args = parser.parse_args()

    if not 0 <= args.min_class_balance <= 1:
        parser.error("--min-class-balance must be between 0 and 1")

    validate_data(args.min_class_balance)