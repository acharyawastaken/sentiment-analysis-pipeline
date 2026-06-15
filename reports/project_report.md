# Project Report: NLP Sentiment Analysis on Social Media Data

**Prepared by:** Samee  
**Project Choice:** NLP – Sentiment Analysis on Social Media Data  
**Internship Program:** Project Acharya Final Submission  
**Date:** June 15, 2026  

---

## 1. Problem Statement

In the modern digital era, social media platforms like Twitter generate vast amounts of unstructured text data every second. Understanding public opinion, customer feedback, and brand perception from this data is crucial for organizations, political campaigns, and businesses. Manual analysis of millions of social media posts is impossible. 

This project aims to solve this problem by building an automated, end-to-end Sentiment Analysis pipeline. The system ingests public social media posts (specifically tweets), preprocesses the unstructured text, extracts numerical feature representations, trains multiple machine learning and deep learning models, and evaluates their performance. Finally, it deploys a real-time interactive dashboard to visualize sentiment distribution, feature importance, and model comparisons.

---

## 2. Dataset Description

### 2.1 Dataset Source & Details
- **Dataset Name:** Sentiment140
- **Source:** Kaggle ([Sentiment140 Dataset Link](https://www.kaggle.com/datasets/kazanova/sentiment140)) (originally compiled by Alec Go, Richa Bhayani, and Lei Huang of Stanford University)
- **Size:** 1.6 million tweets
- **Class Balance:** Perfectly balanced (800,000 positive tweets, 800,000 negative tweets)
- **Target Label Coding:** `0` = Negative, `2` = Neutral, `4` = Positive (The dataset contains only binary labels: 0 and 4).

### 2.2 Features & Columns
The raw dataset contains six columns:
1. `target`: the polarity of the tweet (0 = negative, 4 = positive)
2. `ids`: The id of the tweet (e.g., 2087)
3. `date`: the date of the tweet (e.g., Sat May 16 23:58:44 UTC 2009)
4. `flag`: The query (lyx). If there is no query, then this value is NO_QUERY.
5. `user`: the user that tweeted (e.g., robotickilldozr)
6. `text`: the text of the tweet (e.g., "Lyx is cool")

### 2.3 Working Subset
To optimize local training speed and maintain model robustness, a balanced subset of **100,000 tweets** (50,000 Positive, 50,000 Negative) was sampled. After filtering empty posts during text preprocessing, the final training dataset size was **99,484 rows**.

---

## 3. Methods & Pipeline

A structured engineering approach was taken to develop the sentiment analysis system. The pipeline is divided into three key phases:

```
[Raw Tweets] ──▶ [Text Preprocessing] ──▶ [Feature Extraction] ──▶ [Model Training]
```

### 3.1 Text Preprocessing (NLP Cleaning)
Social media text is highly noisy, containing slang, abbreviations, hashtags, mentions, and URLs. To extract meaningful patterns, a multi-step preprocessing pipeline was implemented using the Natural Language Toolkit (NLTK):
1. **Lowercasing:** Converting all text to lowercase to ensure consistency.
2. **Contraction Expansion:** Expanding short forms (e.g., "don't" $\rightarrow$ "do not", "can't" $\rightarrow$ "cannot").
3. **URL & Hyperlink Removal:** Strip out web links using regular expressions (`http\S+` and `www\.\S+`).
4. **Mention & User Handling:** Remove handles (e.g., `@user`) as they do not carry semantic sentiment.
5. **Hashtag Cleaning:** Strip the `#` symbol but keep the keyword text (e.g., `#happy` $\rightarrow$ `happy`).
6. **Retweet (RT) Marker Removal:** Strip out the retweet prefixes.
7. **Special Characters & Numbers Removal:** Retain only alphabetic characters, stripping special punctuation and numeric digits.
8. **Tokenization:** Split the text into individual word tokens.
9. **Stopwords Removal:** Exclude common grammatical filler words (e.g., "is", "the", "and", "a") that lack sentiment indicator value.
10. **Lemmatization:** Apply NLTK's `WordNetLemmatizer` to reduce words to their base dictionary form (e.g., "running", "ran" $\rightarrow$ "run").

### 3.2 Feature Representation
Two different text representation methodologies were utilized:
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Used for traditional Machine Learning. It represents documents as sparse vectors of uni-grams and bi-grams (`ngram_range=(1,2)`) with a maximum vocabulary size of `10,000`. This captures local word patterns and prioritizes words that are highly descriptive of specific sentiment classes.
- **Word Token Sequences & Padding:** Used for Deep Learning. Raw text was tokenized into sequence integer IDs mapped to a maximum vocabulary size of `20,000`. Sequences were padded/truncated to a fixed size of `100` tokens to construct dense 2D matrices suitable for recurrent neural networks.

### 3.3 Algorithms & Model Implementations
Three models representing different computational paradigms were trained:

1. **Multinomial Naïve Bayes (NB):** A fast, probabilistic baseline classifier that uses Bayes' Theorem. It assumes independence between features.
2. **Logistic Regression (LR):** A standard linear classification model. L2 regularization was applied to prevent overfitting, using an iterative L-BFGS solver.
3. **Deep Learning - Bidirectional LSTM (Long Short-Term Memory):** A Recurrent Neural Network (RNN) designed for sequential data. It processes words in both forward and backward directions to capture long-range contextual relationships.
   - **Architecture:** 
     - *Embedding Layer:* Projects vocabulary IDs into a 64-dimensional continuous space.
     - *Spatial Dropout (20%):* Regularizes feature maps.
     - *Bidirectional LSTM (64 units):* Processes context in both directions.
     - *Global MaxPooling1D:* Extracts the most prominent features from sequence vectors.
     - *Batch Normalization:* Stabilizes backpropagation.
     - *Dense Layer (64 units, ReLU activation) + Dropout (30%).*
     - *Dense Output Layer (1 unit, Sigmoid activation) for binary classification.*

---

## 4. Results & Evaluation

The models were evaluated on an 80/20 stratified train/test split. The performance metrics are summarized below:

### 4.1 Model Metrics Summary

| Model | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naïve Bayes** | 74.94% | 74.95% | 74.94% | 74.94% | 0.8316 |
| **Logistic Regression** | **76.17%** | **76.20%** | **76.17%** | **76.16%** | **0.8418** |
| **Bidirectional LSTM** | 75.33% | 75.37% | 75.33% | 75.33% | 0.8351 |

### 4.2 Key Insights
- **Logistic Regression performed the best** overall, achieving an accuracy of **76.17%** and an AUC-ROC of **0.8418**. This suggests that when using n-gram TF-IDF features, a regularized linear model is highly effective and computationally efficient.
- **Bidirectional LSTM achieved 75.33% accuracy** and 0.8351 AUC-ROC. While deep learning models often outperform traditional machine learning, LSTMs require substantially more parameter tuning and epochs to converge. The current subset size (100K) and sequence length (100) indicate that a linear decision boundary with word combinations (bi-grams) remains a very strong competitor.
- **Feature Importance analysis** (extracted from Logistic Regression coefficients) showed that the top positive words were: *"thank", "welcome", "good", "great", "love", "awesome"*, while the top negative words were: *"sad", "miss", "sorry", "hate", "wish", "sick", "bad"*. This confirms that the models successfully learned emotional markers.

---

## 5. Limitations & Future Improvements

### 5.1 Limitations
- **Binary Classification:** The system only distinguishes between positive and negative sentiment, failing to handle neutral tweets or complex mixed sentiments.
- **Sarcasm and Irony:** Literal cleaning and bag-of-words models struggles to recognize sarcasm (e.g., "Oh great, another delay!").
- **Language Constraint:** The preprocessing pipeline is hardcoded to English grammar (stopwords list, contractions, lemmatization).
- **Out of Vocabulary (OOV) Words:** Newly emerging internet slang, emojis, or misspellings might not map well onto pre-trained vocabularies.

### 5.2 Future Improvements
- **Transformer-based Models:** Fine-tune pre-trained models like BERT or DistilBERT, which utilize attention mechanisms to capture much richer syntactic and semantic context.
- **Multi-Class Sentiment:** Re-train models to handle a third "Neutral" class using advanced datasets.
- **Aspect-Based Sentiment Analysis (ABSA):** Go beyond overall sentiment to identify what specific entity or aspect (e.g., price, delivery, quality) the sentiment is directed towards.
- **Emojis and Punctuation Preservation:** Emojis (such as positive or negative symbols) and punctuation (e.g., "!!!", "?!?") carry high emotional weight and should be encoded using custom tokenizers rather than being stripped out.
- **Model Deployment Expansion:** Host the Streamlit app on Streamlit Community Cloud and integrate a live Twitter API listener for real-time brand monitoring.
