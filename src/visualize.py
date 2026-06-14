# visualization helper functions to generate charts for our reports

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

matplotlib.use("Agg")

from src.utils import FIGURES_DIR, get_logger

logger = get_logger()

# use seaborn darkgrid theme and gray color palette
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("gray")
DEFAULT_SIZE = (10, 6)
IMG_DPI = 150

def make_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def draw_distribution(labels, title="Sentiment Distribution"):
    make_dirs()
    # build dual subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # first subplot is a pie chart of sentiment classes
    lbl_counts = labels.value_counts()
    names = ["Negative" if x == 0 else "Positive" for x in lbl_counts.index]
    color_palette = ["#111827", "#718096"]
    axes[0].pie(lbl_counts, labels=names, autopct="%1.1f%%", colors=color_palette,
                startangle=90, explode=[0.02]*len(lbl_counts), shadow=True,
                textprops={"fontsize": 11, "fontweight": "bold"})
    axes[0].set_title("Sentiment distribution (Pie)", fontsize=13, fontweight="bold")

    # second subplot is a simple bar chart of counts
    sns.barplot(x=names, y=lbl_counts.values, palette=color_palette, ax=axes[1])
    axes[1].set_title("Sentiment distribution (Bar)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Quantity")
    
    # add numbers on top of the bars
    for index, val in enumerate(lbl_counts.values):
        axes[1].text(index, val + (val * 0.01), f"{val:,}", ha="center", fontweight="bold")

    plt.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    # save the distribution chart to reports folder
    target = os.path.join(FIGURES_DIR, "sentiment_distribution.png")
    plt.savefig(target, dpi=IMG_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated chart: {target}")
    return target

def draw_single_cm(matrix, name):
    make_dirs()
    fig, ax = plt.subplots(figsize=(8, 6))
    # convert matrix to numpy array for seaborn heatmap
    arr = np.array(matrix)

    # draw the confusion matrix using Greys colormap
    sns.heatmap(arr, annot=True, fmt="d", cmap="Greys",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax, annot_kws={"size": 13})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {name}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    normalized_name = name.lower().replace(" ", "_")
    target = os.path.join(FIGURES_DIR, f"confusion_matrix_{normalized_name}.png")
    plt.savefig(target, dpi=IMG_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated chart: {target}")
    return target

def draw_all_cms(results):
    make_dirs()
    # get the list of models we trained
    keys = list(results.keys())
    cnt = len(keys)
    # plot confusion matrices side-by-side in one figure
    fig, axes = plt.subplots(1, cnt, figsize=(6 * cnt, 5))
    if cnt == 1:
        axes = [axes]

    # generate heatmap for each model in results
    for ax, name in zip(axes, keys):
        cm = np.array(results[name]["metrics"]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys",
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
                    ax=ax, annot_kws={"size": 12})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        label = name.replace("_", " ").title()
        ax.set_title(label, fontsize=12, fontweight="bold")

    plt.suptitle("Model Confusion Matrices", fontsize=15, fontweight="bold")
    plt.tight_layout()
    target = os.path.join(FIGURES_DIR, "confusion_matrices_all.png")
    plt.savefig(target, dpi=IMG_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated chart: {target}")
    return target

def draw_model_comparison(results):
    make_dirs()
    key_metrics = ["accuracy", "precision", "recall", "f1_score"]
    # pivot results dict to a flat list of rows for pandas
    rows = []
    for k, v in results.items():
        for metric in key_metrics:
            rows.append({
                "Model": k.replace("_", " ").title(),
                "Metric": metric.replace("_", " ").title(),
                "Score": v["metrics"][metric]
            })

    # convert list of dicts to dataframe
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    # draw grouped bar plot using seaborn
    sns.barplot(data=df, x="Metric", y="Score", hue="Model", palette=["#111827", "#4b5563", "#9ca3af"], ax=ax)

    ax.set_ylim(0, 1.0)
    ax.set_title("Performance Metrics Comparison", fontsize=15, fontweight="bold")
    ax.set_ylabel("Value")
    ax.legend(loc="lower right")

    # add labels on top of the comparison bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9)

    plt.tight_layout()
    target = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(target, dpi=IMG_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated chart: {target}")
    return target

def draw_roc_curves(results, y_test):
    make_dirs()
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#111827", "#4b5563", "#9ca3af"]

    # compute and plot roc curve for each model that has probability estimates
    for (name, data), col in zip(results.items(), colors):
        if "y_proba" in data:
            fpr, tpr, _ = roc_curve(y_test, data["y_proba"])
            score = auc(fpr, tpr)
            label = name.replace("_", " ").title()
            ax.plot(fpr, tpr, color=col, lw=2, label=f"{label} (AUC = {score:.4f})")

    # plot diagonal random guess line
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Guess (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    target = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(target, dpi=IMG_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated chart: {target}")
    return target

def draw_wordclouds(df, text_col="processed_text", target_col="target"):
    make_dirs()
    # set up subplots for positive and negative sentiments wordclouds
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    cfgs = {
        0: ("Negative Sentiments", "#718096", "gray"),
        1: ("Positive Sentiments", "#111827", "binary")
    }

    # loop through negative (0) and positive (1) classes
    for ax, (lbl, (title, color, colormap)) in zip(axes, cfgs.items()):
        # generate wordcloud text content
        raw_words = " ".join(df[df[target_col] == lbl][text_col].dropna().values)
        if not raw_words.strip():
            ax.text(0.5, 0.5, "No tweets to show", ha="center", va="center")
            ax.set_title(title)
            continue
        wc = WordCloud(
            width=800, height=450, max_words=100, background_color="white",
            colormap=colormap, contour_width=1, contour_color=color
        ).generate(raw_words)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", color=color)

    plt.suptitle("Social Media Word Clouds", fontsize=16, fontweight="bold")
    plt.tight_layout()
    target = os.path.join(FIGURES_DIR, "wordclouds.png")
    plt.savefig(target, dpi=IMG_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated chart: {target}")
    return target

def draw_lstm_history(history):
    make_dirs()
    # draw learning curve plots (accuracy and loss) side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(history["accuracy"], label="Train Accuracy", lw=2)
    axes[0].plot(history["val_accuracy"], label="Val Accuracy", lw=2)
    axes[0].set_title("LSTM Learning Curve (Accuracy)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history["loss"], label="Train Loss", lw=2, color="#111827")
    axes[1].plot(history["val_loss"], label="Val Loss", lw=2, color="#718096")
    axes[1].set_title("LSTM Learning Curve (Loss)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.suptitle("Deep Learning Training History", fontsize=15, fontweight="bold")
    plt.tight_layout()
    target = os.path.join(FIGURES_DIR, "lstm_training_history.png")
    plt.savefig(target, dpi=IMG_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated chart: {target}")
    return target

def draw_top_features(vectorizer, model, n_top=15):
    make_dirs()
    words = vectorizer.get_feature_names_out()

    # check if model has coef_ attribute (logistic regression)
    if not hasattr(model, "coef_"):
        logger.info("Provided model does not contain coefficients, skipping importance chart.")
        return None

    weights = model.coef_[0]
    # get indices of top 15 positive and negative coefficients
    high_pos = np.argsort(weights)[-n_top:]
    high_neg = np.argsort(weights)[:n_top]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].barh(range(n_top), weights[high_pos], color="#111827")
    axes[0].set_yticks(range(n_top))
    axes[0].set_yticklabels(words[high_pos])
    axes[0].set_title("Top Positive Word Predictors", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Weight Coefficient")

    axes[1].barh(range(n_top), weights[high_neg], color="#718096")
    axes[1].set_yticks(range(n_top))
    axes[1].set_yticklabels(words[high_neg])
    axes[1].set_title("Top Negative Word Predictors", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Weight Coefficient")

    plt.suptitle("TF-IDF Feature Coefficients (Logistic Regression)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    target = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(target, dpi=IMG_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated chart: {target}")
    return target

def generate_all_visualizations(results, df, y_test, tfidf_vec=None):
    logger.info("=" * 60)
    logger.info("GENERATING REPORT GRAPHICS")
    logger.info("=" * 60)

    # generate and save all standard report plots
    paths = [
        draw_distribution(df["target"]),
        draw_all_cms(results),
        draw_model_comparison(results),
        draw_roc_curves(results, y_test),
        draw_wordclouds(df)
    ]

    # generate extra learning curve plot if lstm was run
    if "lstm" in results and "history" in results["lstm"]:
        paths.append(draw_lstm_history(results["lstm"]["history"]))

    if tfidf_vec and "logistic_regression" in results:
        paths.append(draw_top_features(tfidf_vec, results["logistic_regression"]["model"]))

    # generate standalone confusion matrices for each individual model
    for name, data in results.items():
        lbl = name.replace("_", " ").title()
        paths.append(draw_single_cm(data["metrics"]["confusion_matrix"], lbl))

    logger.info("REPORT GRAPHICS DUMPED SUCCESSFULLY")
    return [p for p in paths if p]
