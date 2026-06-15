# NLP Sentiment Analysis Pipeline

This repository contains an end-to-end NLP sentiment analysis system developed for the final capstone project. The project cleans raw social media posts, performs feature representation, trains multiple classification models, and provides an interactive web dashboard for real-time sentiment predictions and model evaluations.

## Project Structure

- **src/**: Core source modules.
  - `preprocessing.py`: NLP text cleaning (url stripping, lemmatization, stopword removal).
  - `feature_engineering.py`: TF-IDF vectorization and Keras sequence tokenization.
  - `models.py`: Classifier training for Naive Bayes, Logistic Regression, and Bidirectional LSTM.
  - `visualize.py`: Evaluator plot generator (ROC curves, confusion matrices, word clouds).
  - `utils.py`: Path setup and helper utilities.
- **dashboard/**:
  - `app.py`: Streamlit web dashboard application and interactive sentiment chatbot.
- **notebooks/**:
  - `sentiment_analysis.ipynb`: Exploratory prototype notebook.
- **reports/**:
  - `project_report.md`: Markdown summary of project goals, metrics, and conclusions.
  - `project_report.docx`: Formatted Word document version of the report.
  - **figures/**: Generated performance plots.
- **models/**: Serialized model checkpoints and vectorizers.
- **data/**: Storage path for raw and preprocessed data.

## Pipeline Workflow

1. **Preprocessing**: Cleans the raw data (stripping user mentions, hashtags, URLs, and stopwords) and applies lemmatization.
2. **Feature Engineering**: Generates TF-IDF representations (for machine learning models) and padded word token sequences (for the LSTM).
3. **Training & Evaluation**: Trains and logs evaluation metrics (Accuracy, Precision, Recall, F1, AUC-ROC) for:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Bidirectional LSTM
4. **Visualization**: Saves diagnostic plots to the reports folder.
5. **Dashboard Deployment**: Loads the saved models and runs an interactive dashboard, including a conversational chatbot interface to test sentiment analysis in real time.

## Getting Started

### Installation
Install the project dependencies:
```bash
pip install -r requirements.txt
```

### Running the Dashboard
Start the local Streamlit application:
```bash
python -m streamlit run dashboard/app.py
```

### Running the Training Pipeline
To retrain the models and regenerate plots:
```bash
python -m src.models
```
