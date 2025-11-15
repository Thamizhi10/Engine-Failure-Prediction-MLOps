import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, login
import mlflow
import mlflow.sklearn
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

login(os.getenv("HF_TOKEN"))
api = HfApi(token=os.getenv("HF_TOKEN"))

#fetch data
DATA_REPO_ID = "tam3222/engine_data"   

XTRAIN_FILE = "Xtrain.csv"
XTEST_FILE  = "Xtest.csv"
YTRAIN_FILE = "ytrain.csv"
YTEST_FILE  = "ytest.csv"

# Download files
xtrain_path = hf_hub_download(repo_id=DATA_REPO_ID, filename=XTRAIN_FILE)
xtest_path  = hf_hub_download(repo_id=DATA_REPO_ID, filename=XTEST_FILE)
ytrain_path = hf_hub_download(repo_id=DATA_REPO_ID, filename=YTRAIN_FILE)
ytest_path  = hf_hub_download(repo_id=DATA_REPO_ID, filename=YTEST_FILE)

# Read into DataFrames
X_train = pd.read_csv(xtrain_path)
X_test  = pd.read_csv(xtest_path)

# y usually needs to be a 1D array/Series
y_train = pd.read_csv(ytrain_path).iloc[:, 0]
y_test  = pd.read_csv(ytest_path).iloc[:, 0]

#Define Models and Parameters
models = {
    "RandomForest": (RandomForestClassifier(), {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
    }),
    "GradientBoosting": (GradientBoostingClassifier(), {
        "n_estimators": [100, 150],
        "learning_rate": [0.05, 0.1],
    }),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric="logloss"), {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    })
}

best_model = None
best_accuracy = 0
best_model_name = ""



#Experimentation & Tracking
for model_name, (model, params) in models.items():
    with mlflow.start_run(run_name=model_name):
        grid = GridSearchCV(model, params, cv=3, scoring="accuracy", verbose=1)
        grid.fit(X_train, y_train)

        preds = grid.best_estimator_.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        # Log to MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(grid.best_estimator_, model_name)

        print(f"✅ {model_name} Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = grid.best_estimator_
            best_model_name = model_name

#Save & Register the Best Model
best_model_path = f"best_{best_model_name}.pkl"
joblib.dump(best_model, best_model_path)
login(os.getenv("HF_TOKEN"))
api = HfApi(token=os.getenv("HF_TOKEN"))
upload_file(
    path_or_fileobj=best_model_path,
    path_in_repo=f"models/{best_model_name}.pkl",
    repo_id="tam3222/Engine_Model",
    repo_type="model"
)
