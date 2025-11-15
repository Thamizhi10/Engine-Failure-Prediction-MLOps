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

dataset = load_dataset("tam3222/engine_data")
df = dataset['train'].to_pandas()

# Replace any inconsistent column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

#Cap outliers at 99th percentile for 'Engine rpm' and 'Fuel pressure'
for col in ['engine_rpm', 'fuel_pressure']:
    upper_limit = df[col].quantile(0.99)
    df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

#Handle extreme outliers in 'Coolant temp'
# Let's assume valid temperature range is around 60°C–110°C
# Replace extreme low or high temperatures outside this range with the median
coolant_median = df['coolant_temp'].median()
df['coolant_temp'] = np.where((df['coolant_temp'] < 60) | (df['coolant_temp'] > 110),
                              coolant_median, df['coolant_temp'])

#Feature Scaling

# Define features and target
X = df.drop('engine_condition', axis=1)
y = df['engine_condition']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Combine scaled features with target
df_scaled = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

#Handle Class Imbalance (SMOTE)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Create final balanced dataframe
df_balanced = pd.concat([X_resampled, y_resampled], axis=1)

# Check class distribution after balancing
y_resampled.value_counts()

#Train-Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Convert into DataFrames
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

#print(train_data.shape, test_data.shape)

X_train.to_csv("Xtrain.csv",index=False)
X_test.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)

# Upload files to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

from huggingface_hub import HfApi, login
login(os.getenv("HF_TOKEN"))

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="tam3222/engine_data",
        repo_type="dataset",
    )
