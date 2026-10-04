import mlflow
from sklearn.datasets import load_breast_cancer, load_iris


def load_and_predict():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_registry_uri("sqlite:///mlflow.db")

    model_uri = "models:/cancer-classifier-prod@staging"

    print(f"Loading model: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    data = load_breast_cancer(as_frame=True)

    X = data.data
    y = data.target

    # index 0 เป็น malignant และ index 19 เป็น benign ในข้อมูลชุดนี้
    sample_indices = [
        y.index[y.eq(label)][0]
        for label in (0, 1)
    ]

    sample_data = X.loc[sample_indices]

    predictions = model.predict(sample_data)

    print("Sample index | Actual | Predicted | Correct")

    for sample_index, prediction in zip(
        sample_indices,
        predictions,
        strict=True,
    ):
        actual_label = int(y.loc[sample_index])
        predicted_label = int(prediction)

        actual_name = data.target_names[actual_label]
        predicted_name = data.target_names[predicted_label]

        correct = actual_label == predicted_label

        print(
            f"{sample_index:12} | "
            f"{actual_name:9} | "
            f"{predicted_name:9} | "
            f"{correct}"
        )


def test_no_missing():
    assert df.isnull().sum().sum() == 0


def test_schema():
    assert df.shape == (150, 5)
    assert "target" in df.columns
    assert "sepal length (cm)" in df.columns


def test_three_classes():
    assert set(df["target"].unique()) == {0, 1, 2}


def test_class_balance():
    assert df["target"].value_counts(normalize=True).min() >= 0.20


# Iris example only (3 classes)
df = load_iris(as_frame=True).frame


if __name__ == "__main__":
    load_and_predict()