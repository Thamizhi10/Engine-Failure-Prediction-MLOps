# Engine-Failure-Prediction—MLOps Pipeline

An end-to-end MLOps pipeline that predicts engine failure from real-time sensor data (RPM, oil pressure, fuel pressure, coolant pressure/temperature), automating the full workflow — data registration, preprocessing, model training, and deployment using GitHub Actions for CI/CD. Capstone project for the PG Program in AI & Machine Learning, UT Austin McCombs School of Business.

## Problem Statement

Vehicle and engine breakdowns cause costly repairs, downtime, and safety risks. This project builds a predictive maintenance model that classifies engine condition (normal vs. faulty) from sensor readings, enabling proactive maintenance instead of reactive repairs.

## Approach

1. **EDA** — analyzed distributions and relationships across 6 sensor features (~20,000 records); identified lubrication oil pressure/temperature as the strongest discriminators between healthy and faulty engines
2. **Feature Engineering** — derived temperature and pressure differential features to capture thermal imbalance and pressure loss dynamics
3. **Data Preparation** — outlier capping (99th percentile), domain-informed value correction for invalid sensor readings, StandardScaler normalization, SMOTE for class imbalance (64/36 split)
4. **Model Building** — trained and compared Random Forest, Gradient Boosting, and XGBoost with `GridSearchCV` (3-fold CV), tracked via **MLflow**; also tested a stacking ensemble, which did not outperform the best individual model and was correctly not adopted
5. **Model Selection** — **Random Forest** performed best (67.2% test accuracy, balanced precision/recall), serialized with `joblib` and registered to Hugging Face Model Hub
6. **Deployment** — Streamlit app on Hugging Face Spaces, containerized with Docker, loading the model directly from the Hugging Face Model Hub
7. **CI/CD Automation** — GitHub Actions pipeline (`.github/workflows/pipeline.yml`) automatically registers data, retrains, evaluates, and redeploys on every push to `main` — a fully automated retrain-to-deploy pipeline

## Repository Structure

engine_project/
├── data/
│ └── engine_data.csv
├── model_building/
│ ├── data_register.py
│ ├── prep.py
│ └── train.py
├── deployment/
│ ├── app.py
│ ├── Dockerfile
│ └── requirements.txt
└── hosting/
└── hosting.py

.github/workflows/
└── pipeline.yml

notebook/
└── PredictiveMaintenance_Notebook.ipynb

## Results

| Model | Test Accuracy |
|---|---|
| **Random Forest (final)** | **67.2%** |
| XGBoost | 66.9% |
| Gradient Boosting | 66.4% |
| Stacking Ensemble | ~67% (no meaningful uplift over best individual model) |

**Top predictive features:** Engine RPM, Fuel Pressure, Lubrication Oil Temperature

## Skills Demonstrated

- EDA, feature engineering, outlier treatment, class imbalance handling (SMOTE)
- Model comparison & hyperparameter tuning (GridSearchCV)
- Ensemble methods, including stacking
- Experiment tracking (MLflow)
- CI/CD pipeline automation (GitHub Actions)
- Dataset & model versioning (Hugging Face Hub)
- Containerization (Docker) and cloud deployment (Hugging Face Spaces)

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `imbalanced-learn` (SMOTE) · `MLflow` · `Docker` · `Streamlit` · `GitHub Actions` · `Hugging Face Hub`

## Live Deployment

- Streamlit App: https://huggingface.co/spaces/tam3222/PredictiveMaintenance
- Model Hub: https://huggingface.co/tam3222/Engine_Model

## Dataset

Engine sensor dataset (~20,000 records), provided as part of the PGP-AIML capstone.
