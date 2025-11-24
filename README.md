# Proyecto MLOps – TelcoVision  
Predicción de churn con pipeline reproducible (DVC + DAGSHUB + MLflow + GitHub Actions)
Este proyecto construye un pipeline MLOps completo para predecir churn de clientes utilizando DVC para versionado de datos/modelos, MLflow para experimentación y GitHub Actions para automatización CI/CD.

---

## 📁 Estructura del repositorio

```text
TELCOVISION-MLOPS
├── .dvc/               ← metadata DVC
├── .github/workflows/  ← CI/CD
│   └── ci.yaml
│
├── data/
│   ├── raw/            ← dataset original
│   │   └── telco_churn.csv
│   └── processed/      ← train/test versionados por DVC
│       ├── train.csv
│       └── test.csv
│
├── models/
│   └── telco_churn.pkl ← modelo entrenado
│
├── src/
│   ├── data_prep.py    ← limpieza + features
│   ├── make_data.py
│   └── train.py        ← entrenamiento + MLflow
│
├── dvc.yaml            ← definición del pipeline
├── dvc.lock            ← hashes para reproducibilidad
├── params.yaml         ← hiperparámetros
├── metrics.json        ← métricas del modelo
├── requirements.txt
└── README.md
