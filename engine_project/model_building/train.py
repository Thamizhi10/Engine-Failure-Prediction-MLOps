import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, login
from pyngrok import ngrok
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from xgboost import XGBClassifier
import os
import joblib
from huggingface_hub import upload_file
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# Login to Hugging Face
login(os.getenv("HF_TOKEN"))
api = HfApi(token=os.getenv("HF_TOKEN"))

# ========================
# 1. Fetch data from HF Hub
# ========================
DATA_REPO_ID = "tam3222/engine_data"   

XTRAIN_FILE = "Xtrain.csv"
XTEST_FILE  = "Xtest.csv"
YTRAIN_FILE = "ytrain.csv"
YTEST_FILE  = "ytest.csv"

# Download files (dataset repo)
xtrain_path = hf_hub_download(
    repo_id=DATA_REPO_ID, 
    filename=XTRAIN_FILE, 
    repo_type="dataset"
)
xtest_path  = hf_hub_download(
    repo_id=DATA_REPO_ID, 
    filename=XTEST_FILE, 
    repo_type="dataset"
)
ytrain_path = hf_hub_download(
    repo_id=DATA_REPO_ID, 
    filename=YTRAIN_FILE, 
    repo_type="dataset"
)
ytest_path  = hf_hub_download(
    repo_id=DATA_REPO_ID, 
    filename=YTEST_FILE, 
    repo_type="dataset"
)

# Read into DataFrames
X_train = pd.read_csv(xtrain_path)
X_test  = pd.read_csv(xtest_path)

# y usually needs to be a 1D array/Series
y_train = pd.read_csv(ytrain_path).iloc[:, 0]
y_test  = pd.read_csv(ytest_path).iloc[:, 0]

print("Shapes:")
print("X_train:", X_train.shape, "| y_train:", y_train.shape)
print("X_test:", X_test.shape, "| y_test:", y_test.shape)

# ========================
# 2. Define Models & Params
# ========================
models = {
    "RandomForest": (
        RandomForestClassifier(),
        {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
        }
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(),
        {
            "n_estimators": [100, 150],
            "learning_rate": [0.05, 0.1],
        }
    ),
    "XGBoost": (
        XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        }
    ),
}

best_model = None
best_accuracy = 0
best_model_name = ""

# ========================
# 3. Train & Evaluate Models
# ========================
for model_name, (model, params) in models.items():
    print(f"\n🔍 Training model: {model_name}")
    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=3,
        scoring="accuracy",
        verbose=1
    )
    grid.fit(X_train, y_train)

    preds = grid.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"{model_name} Best Params: {grid.best_params_}")
    print(f"{model_name} Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    # Track best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = grid.best_estimator_
        best_model_name = model_name

print("Best model:", best_model_name)
print(f"Best accuracy: {best_accuracy:.4f}")

# ========================
# 4. Save & Upload Best Model
# ========================
best_model_path = f"best_{best_model_name}.pkl"
joblib.dump(best_model, best_model_path)
print(f"Saved best model to {best_model_path}")

# (Re)login & upload to HF model repo
login(os.getenv("HF_TOKEN"))
api = HfApi(token=os.getenv("HF_TOKEN"))

upload_file(
    path_or_fileobj=best_model_path,
    path_in_repo=f"models/{best_model_name}.pkl",
    repo_id="tam3222/Engine_Model",
    repo_type="model"
)

print("Best model uploaded to Hugging Face Hub.")
