import pandas as pd

def clean_data():
    df = pd.read_csv("data/processed/data.csv")

    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # numeric conversion
    cols = df.columns.drop("datetime")
    df[cols] = df[cols].astype(float)

    df.to_csv("data/processed/cleaned.csv", index=False)

if __name__ == "__main__":
    clean_data()
