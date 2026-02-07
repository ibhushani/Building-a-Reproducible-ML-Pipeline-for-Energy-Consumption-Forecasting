import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

mlflow.set_tracking_uri("http://127.0.0.1:5000") 

def train():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    df = pd.read_csv("data/processed/features.csv")

    # reduce size for speed
    if len(df) > 300000:
        df = df.sample(300000, random_state=42)

    X = df.drop("Global_active_power", axis=1)
    y = df["Global_active_power"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"]
    )

    with mlflow.start_run():

        model = RandomForestRegressor(
            n_estimators=params["model"]["n_estimators"],
            max_depth=params["model"]["max_depth"],
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # ✅ FIXED RMSE
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        joblib.dump(model, "model.pkl")

        mlflow.log_params(params["model"])
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train()
