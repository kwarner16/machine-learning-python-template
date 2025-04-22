from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import os

# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine

def load_or_download_data(path="data/raw/AB_NYC_2019.csv",
                          url="https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv"):
    """
    Loads the dataset from local path if available,
    otherwise downloads it from the provided URL and saves it locally.
    """
    if not os.path.exists(path):
        print(f"File not found at path: {path}. Downloading from {url}...")
        df = pd.read_csv(url)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Downloaded and saved to {path}")
    else:
        print(f"Found file at {path}. Loading from disk...")

    return pd.read_csv(path)