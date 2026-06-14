# training scripts for ML models (Naive Bayes, Logistic Regression, LSTM)

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB

from src.feature_engineering import execute_feature_pipeline
from src.utils import MODELS_DIR, REPORTS_DIR, SEED, dump_obj, get_logger

logger = get_logger()

def train_nb(X_train, y_train, X_test, y_test):
    logger.info("--- Training Naive Bayes Baseline ---")
    # initialize multinomial naive bayes classifier with default smoothing
    model = MultinomialNB(alpha=1.0)
    # fit model on tfidf vectors
    model.fit(X_train, y_train)

    # predict labels and probabilities for validation set
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    # evaluate performance metrics and save model to models/ directory
    metrics = evaluate_metrics("Naïve Bayes", y_test, preds, probas)
    dump_obj(model, "naive_bayes.pkl")
    return {"model": model, "metrics": metrics, "y_pred": preds, "y_proba": probas}

def train_lr(X_train, y_train, X_test, y_test):
    logger.info("--- Training Logistic Regression ---")
    # initialize logistic regression model with L2 regularization
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    # train the logistic regression classifier
    model.fit(X_train, y_train)

    # run predictions on test set
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_metrics("Logistic Regression", y_test, preds, probas)
    dump_obj(model, "logistic_regression.pkl")
    return {"model": model, "metrics": metrics, "y_pred": preds, "y_proba": probas}

def build_lstm(vocab_sz, embed_dim=64, max_len=100):
    # import Keras layers locally to speed up execution of non-LSTM models
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, SpatialDropout1D, LSTM, Dense, Dropout,
        Bidirectional, GlobalMaxPooling1D, BatchNormalization
    )
    from tensorflow.keras.optimizers import Adam

    # define a sequential keras model with embedding and bi-lstm
    model = Sequential([
        Embedding(vocab_sz, embed_dim, input_length=max_len),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    # compile the model using adam optimizer with learning rate 0.002
    model.compile(
        optimizer=Adam(learning_rate=0.002),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary(print_fn=logger.info)
    return model

def train_lstm_network(X_train_seq, y_train, X_test_seq, y_test, vocab_sz):
    from tensorflow.keras.callbacks import EarlyStopping

    logger.info("--- Training Bidirectional LSTM ---")
    model = build_lstm(vocab_sz)

    # get early stopping callback ready so we don't overfit
    stopper = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # fit the network using 10% validation split and 15 epochs
    hist = model.fit(
        X_train_seq, y_train.values,
        validation_split=0.1,
        epochs=15, batch_size=128,
        callbacks=[stopper],
        verbose=1
    )

    # run predictions and format probabilities to binary classes (0 or 1)
    probas = model.predict(X_test_seq).flatten()
    preds = (probas >= 0.5).astype(int)

    metrics = evaluate_metrics("LSTM", y_test, preds, probas)

    # create target models path and save trained network weights
    path = os.path.join(MODELS_DIR, "lstm_model.keras")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(path)
    logger.info(f"Saved LSTM weights: {path}")

    return {
        "model": model, "metrics": metrics, "history": hist.history,
        "y_pred": preds, "y_proba": probas
    }

def evaluate_metrics(name, y_true, y_pred, y_proba=None):
    # calculate accuracy, weighted precision, recall, and f1 score
    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted")
    }

    # compute area under roc curve if probabilities are provided
    if y_proba is not None:
        res["auc_roc"] = roc_auc_score(y_true, y_proba)

    # calculate confusion matrix values
    res["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    # log the metrics comparison to stdout
    logger.info("-" * 40)
    logger.info(f" {name} Metrics:")
    logger.info("-" * 40)
    logger.info(f" Accuracy:  {res['accuracy']:.4f}")
    logger.info(f" Precision: {res['precision']:.4f}")
    logger.info(f" Recall:    {res['recall']:.4f}")
    logger.info(f" F1 Score:  {res['f1_score']:.4f}")
    if "auc_roc" in res:
        logger.info(f" AUC-ROC:   {res['auc_roc']:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])}")

    return res

def run_model_training():
    logger.info("=" * 60)
    logger.info("STARTING MODELS TRAINING PIPELINE")
    logger.info("=" * 60)

    # run the feature pipeline first to get clean splits
    feats = execute_feature_pipeline()

    runs = {}

    # train traditional machine learning models first
    runs["naive_bayes"] = train_nb(
        feats["X_train_tfidf"], feats["y_train"],
        feats["X_test_tfidf"], feats["y_test"]
    )
    runs["logistic_regression"] = train_lr(
        feats["X_train_tfidf"], feats["y_train"],
        feats["X_test_tfidf"], feats["y_test"]
    )

    # define vocab size based on actual unique tokens, capped at 20k
    vocab_sz = min(len(feats["keras_tokenizer"].word_index) + 1, 20000)
    # train the bidirectional lstm recurrent network
    runs["lstm"] = train_lstm_network(
        feats["X_train_seq"], feats["y_train"],
        feats["X_test_seq"], feats["y_test"],
        vocab_sz
    )

    # extract accuracy/precision/recall/f1-score and save to reports/ directory
    comparison = {}
    for k, v in runs.items():
        comparison[k] = {metric: val for metric, val in v["metrics"].items() if metric != "confusion_matrix"}
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    # save the comparisons json file
    summary_path = os.path.join(REPORTS_DIR, "model_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Exported metrics comparison table: {summary_path}")

    # print a quick summary of model performance at the end
    logger.info("\n" + "=" * 60)
    logger.info("MODEL TRAINING PIPELINE DONE")
    logger.info("=" * 60)
    comp_df = pd.DataFrame(comparison).T
    logger.info(f"\n{comp_df.to_string()}")

    return runs

if __name__ == "__main__":
    run_model_training()
