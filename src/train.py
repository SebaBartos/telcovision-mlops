import argparse
import json
import os
import pandas as pd
import yaml
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 🔹 NUEVO: para loggear métricas a DagsHub / MLflow
import mlflow


def load_params(params_path: str):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def main(params_path: str):
    # Cargar parámetros
    params = load_params(params_path)

    train_path = params["paths"]["train_data"]
    test_path = params["paths"]["test_data"]
    model_path = params["paths"]["model_path"]
    metrics_path = params["paths"]["metrics_path"]

    model_C = params["model"]["C"]
    model_max_iter = params["model"]["max_iter"]
    target_col = "churn"

    # Leer datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Entrenar modelo
    clf = LogisticRegression(C=model_C, max_iter=model_max_iter)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calcular métricas
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }

    # Guardar modelo y métricas
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Modelo guardado en {model_path}")
    print(f"✅ Métricas: {metrics}")

    # =====================================================
    # Log en DagsHub / MLflow (opcional si hay credenciales)
    # =====================================================
    tracking_uri = "https://dagshub.com/SebaBartos/telcovision-mlops.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    mlflow_user = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_pass = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not mlflow_user or not mlflow_pass:
        print("⚠️ No hay credenciales MLflow en el entorno. Salteo el log en DagsHub.")
        return

    mlflow.set_experiment("telcovision-mlops")

    with mlflow.start_run():
        # parámetros
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", model_C)
        mlflow.log_param("max_iter", model_max_iter)

        # métricas
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1", metrics["f1"])

        # artefactos
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(metrics_path)
    # =====================================================


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    args = ap.parse_args()
    main(args.params)
