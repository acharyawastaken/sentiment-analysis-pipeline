# feature engineering script - tfidf and lstm tokenization

import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.utils import (
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    SEED,
    get_logger,
    init_workspace,
)

logger = get_logger()

# define hyperparams for vectorizers
MAX_WORDS_TFIDF = 10000
NGRAMS = (1, 2)
MIN_DOC_FREQ = 5
MAX_DOC_FREQ = 0.95

VOCAB_CAP_LSTM = 20000
MAX_LEN_LSTM = 100
SPLIT_RATIO = 0.2

def load_preprocessed_df(name="cleaned_tweets.csv"):
    # build full path for processed data
    path = os.path.join(PROCESSED_DATA_PATH, name)
    # throw error if file is missing
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cleaned file missing: {path}. Run preprocessing script first.")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset: {len(df):,} records")
    return df

def get_tfidf_features(texts, fit=True, vectorizer=None):
    # if we are fitting, train a new vectorizer on the texts
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=MAX_WORDS_TFIDF,
            ngram_range=NGRAMS,
            min_df=MIN_DOC_FREQ,
            max_df=MAX_DOC_FREQ,
            sublinear_tf=True
        )
        features = vectorizer.fit_transform(texts)
        logger.info(f"TFIDF vocabulary size: {features.shape[1]}")
    # otherwise just transform using the pre-fit vectorizer
    else:
        features = vectorizer.transform(texts)
    return features, vectorizer

def get_lstm_sequences(texts, fit=True, tokenizer=None):
    # local imports to speed up startup
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # fit tokenizer on texts if fit=True
    if fit:
        tokenizer = Tokenizer(num_words=VOCAB_CAP_LSTM, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        logger.info(f"Keras vocabulary size: {len(tokenizer.word_index)}")

    # convert texts to lists of integers
    seqs = tokenizer.texts_to_sequences(texts)
    # pad sequence length to max_len_lstm
    padded = pad_sequences(seqs, maxlen=MAX_LEN_LSTM, padding="post", truncating="post")
    logger.info(f"Seq shape: {padded.shape}")
    return padded, tokenizer

def split_dataset(df, text_field="processed_text", target_field="target"):
    # fill missing values in text field with empty string
    X = df[text_field].fillna("")
    y = df[target_field]
    # split using stratified partition so both classes are balanced in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SPLIT_RATIO, random_state=SEED, stratify=y
    )
    logger.info(f"Dataset split: Train={len(X_train)} / Test={len(X_test)}")
    return X_train, X_test, y_train, y_test

# simple serialize/deserialize functions for savings models later
def save_vectorizer(vec, name="tfidf_vectorizer.pkl"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(vec, f)
    return path

def load_vectorizer(name="tfidf_vectorizer.pkl"):
    path = os.path.join(MODELS_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)

def save_tokenizer(tok, name="lstm_tokenizer.pkl"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(tok, f)
    return path

def load_tokenizer(name="lstm_tokenizer.pkl"):
    path = os.path.join(MODELS_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)

def execute_feature_pipeline():
    init_workspace()
    logger.info("*" * 50)
    logger.info("RUNNING FEATURE ENGINEERING")
    logger.info("*" * 50)

    # load the preprocessed dataframe first
    df = load_preprocessed_df()
    # split the dataframe into train/test sets
    X_train, X_test, y_train, y_test = split_dataset(df)

    # fit and save tfidf features
    logger.info("\nExtracting TF-IDF Features...")
    X_train_tfidf, tfidf_vec = get_tfidf_features(X_train, fit=True)
    X_test_tfidf, _ = get_tfidf_features(X_test, fit=False, vectorizer=tfidf_vec)
    save_vectorizer(tfidf_vec)

    # fit and save lstm token sequences
    logger.info("\nExtracting LSTM Sequences...")
    X_train_seq, keras_tok = get_lstm_sequences(X_train, fit=True)
    X_test_seq, _ = get_lstm_sequences(X_test, fit=False, tokenizer=keras_tok)
    save_tokenizer(keras_tok)

    logger.info("FEATURE ENGINEERING PIPELINE DONE")
    logger.info("*" * 50)

    return {
        "X_train_tfidf": X_train_tfidf, "X_test_tfidf": X_test_tfidf,
        "X_train_seq": X_train_seq, "X_test_seq": X_test_seq,
        "y_train": y_train, "y_test": y_test,
        "tfidf_vectorizer": tfidf_vec, "keras_tokenizer": keras_tok
    }

if __name__ == "__main__":
    execute_feature_pipeline()
