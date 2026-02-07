import pandas as pd
import joblib
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def evaluate():
    df = pd.read_csv("data/processed/features.csv")
    model = joblib.load("model.pkl")

    X = df.drop("Global_active_power", axis=1)
    y = df["Global_active_power"]

    preds = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    with mlflow.start_run():
        mlflow.log_metric("Eval_RMSE", rmse)
        mlflow.log_metric("Eval_R2", r2)

if __name__ == "__main__":
    evaluate()
