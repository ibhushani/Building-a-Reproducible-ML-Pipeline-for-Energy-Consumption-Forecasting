import pandas as pd

def feature_engineering():
    df = pd.read_csv("data/processed/cleaned.csv")

    df["datetime"] = pd.to_datetime(df["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    df.drop("datetime", axis=1, inplace=True)

    df.to_csv("data/processed/features.csv", index=False)

if __name__ == "__main__":
    feature_engineering()
