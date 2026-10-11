import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"C:\Users\cd_61\OneDrive\Documentos\TEC DE MONTERREY\5059\dataset\data.csv")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none"
    )
    parser.add_argument("--random-state", type=int, default=42)
    args, _ = parser.parse_known_args()
    return args


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df = df.drop(
        columns=["id", "Unnamed: 32"],
        errors="ignore"
    )
    df = df.drop_duplicates()
    df = df.dropna(subset=["diagnosis"])

    y = df["diagnosis"].astype(str).str.strip().str.upper()
    y = y.map({"M": 1, "B": 0})

    X = df.drop(columns=["diagnosis"])
    X = X.apply(pd.to_numeric, errors="coerce")

    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid].astype(int)


def build_model(C, max_iter, class_weight, seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=(
                None if class_weight == "none"
                else "balanced"
            ),
            random_state=seed,
            solver="liblinear"
        ))
    ])


def main():
    args = parse_args()
    X, y = load_data(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=args.random_state,
        stratify=y
    )

    mlflow.set_experiment("Breast Cancer Classification")
    model = build_model(
        args.C, args.max_iter,
        args.class_weight, args.random_state
    )

    with mlflow.start_run():
        mlflow.log_params({
            "C": args.C,
            "max_iter": args.max_iter,
            "class_weight": args.class_weight,
            "random_state": args.random_state,
            "test_size": 0.20,
            "solver": "liblinear"
        })

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            )
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        Path("outputs").mkdir(exist_ok=True)
        metrics_path = Path("outputs/metrics.json")
        metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8"
        )
        mlflow.log_artifact(str(metrics_path))

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benigno", "Maligno"]
        )
        display.plot()
        plt.tight_layout()
        plt.savefig("outputs/confusion_matrix.png", dpi=150)
        plt.close()
        mlflow.log_artifact("outputs/confusion_matrix.png")

        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Métricas:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
