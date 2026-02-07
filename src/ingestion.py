import pandas as pd
import os

def ingest_data():
    df = pd.read_csv(
        "data/raw/energy.csv",
        sep=",",  # kyunki tumhari file CSV hai
        parse_dates={'datetime': ['Date', 'Time']},
        low_memory=False
    )

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/data.csv", index=False)

if __name__ == "__main__":
    ingest_data()
