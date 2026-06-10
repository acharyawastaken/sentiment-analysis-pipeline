# general helper utilities for data loading and model serialization

import logging
import os
import pickle
import sys

import pandas as pd

# find the project root directory and set up main folder paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

SEED = 42
DEFAULT_LIMIT = 100000

# standard column names from the sentiment140 dataset
COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
LABEL_MAP = {0: 0, 4: 1}

def get_logger():
    """Returns a simple logger for terminal monitoring."""
    logger = logging.getLogger("sentiment")
    # check if logger already has handlers so we don't duplicate logs
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        # format log output with custom timestamp format
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    return logger

def init_workspace():
    # loop through directories and create them if they are missing
    for d in [RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

def fetch_nltk_packages():
    # download specific nltk packages required for the processing scripts
    import nltk
    pkgs = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]
    # look for each resource and download silently if missing
    for loc, name in pkgs:
        try:
            nltk.data.find(loc)
        except LookupError:
            nltk.download(name, quiet=True)

def load_raw_csv(path=None, limit=DEFAULT_LIMIT):
    """Loads raw dataset, maps target target column and balances classes."""
    # default path if none was passed in
    if path is None:
        path = os.path.join(RAW_DATA_PATH, "sentiment140.csv")

    # verify raw csv file is present
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}. Download from Kaggle first!")

    logger = get_logger()
    logger.info(f"Reading CSV from {path}...")
    
    # Read the dataset using ISO-8859-1 encoding (standard for Sentiment140)
    df = pd.read_csv(path, encoding="ISO-8859-1", header=None, names=COLUMNS)
    logger.info(f"Raw rows: {len(df):,}")

    # Normalize labels to 0 (Negative) and 1 (Positive)
    df["target"] = df["target"].map(LABEL_MAP)
    # drop any rows where target mapping failed (e.g., neutral reviews)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    # Downsample and balance classes if limit is specified
    if 0 < limit < len(df):
        sz = limit // 2
        # select equal number of positive and negative reviews to prevent bias
        p_df = df[df["target"] == 1].sample(n=sz, random_state=SEED)
        n_df = df[df["target"] == 0].sample(n=sz, random_state=SEED)
        # shuffle the dataset so classes are mixed
        df = pd.concat([p_df, n_df]).sample(frac=1, random_state=SEED).reset_index(drop=True)
        logger.info(f"Sampled down to {len(df):,} balanced rows")

    return df

def save_csv(df, name):
    # build processed file path and export to csv
    target = os.path.join(PROCESSED_DATA_PATH, name)
    df.to_csv(target, index=False)
    get_logger().info(f"Exported df: {target}")
    return target

def dump_obj(obj, name):
    # serialize python objects like classifiers using pickle
    target = os.path.join(MODELS_DIR, name)
    with open(target, "wb") as f:
        pickle.dump(obj, f)
    get_logger().info(f"Saved: {target}")
    return target

def load_obj(name):
    # deserialize the saved pickle model
    target = os.path.join(MODELS_DIR, name)
    if not os.path.exists(target):
        raise FileNotFoundError(f"Missing model file: {target}")
    with open(target, "rb") as f:
        res = pickle.load(f)
    get_logger().info(f"Loaded: {target}")
    return res

if __name__ == "__main__":
    init_workspace()
    fetch_nltk_packages()
    print("Project directory structure verified.")
