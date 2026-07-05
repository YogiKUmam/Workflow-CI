from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "heart_disease_preprocessing" / "heart_preprocessed.csv"
OUTPUT_MODEL = ROOT / "outputs" / "model"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    x = df.drop(columns=["target"])
    y = df["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    signature = infer_signature(x_test, predictions)

    mlflow.set_experiment("heart-disease-ci")
    with mlflow.start_run(run_name="ci-random-forest"):
        mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
        mlflow.log_metric("precision", precision_score(y_test, predictions))
        mlflow.log_metric("recall", recall_score(y_test, predictions))
        mlflow.log_metric("f1", f1_score(y_test, predictions))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=x_test.head(5),
        )

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(
        model,
        path=str(OUTPUT_MODEL),
        signature=signature,
        input_example=x_test.head(5),
    )
    print(f"Saved model to {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
