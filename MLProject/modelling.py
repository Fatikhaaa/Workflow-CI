import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# =============== MLflow Tracking (SQLite) ===============
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Membuat eksperimen jika belum ada
experiment_name = "insurance-experiment"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Parse parameters for MLflow Project
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
args = parser.parse_args()

# Dataset path
train_path = "MLProject/insurance_preprocessing/insurance_train_preprocessed.csv"
test_path = "MLProject/insurance_preprocessing/insurance_test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # log model ke MLflow, artifact_path = "model"
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Training success. MSE:", mse, "R2:", r2)
