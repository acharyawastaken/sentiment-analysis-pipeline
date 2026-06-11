# simple text preprocessing script for project acharya

import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.utils import (
    DEFAULT_LIMIT,
    fetch_nltk_packages,
    get_logger,
    init_workspace,
    load_raw_csv,
    save_csv,
)

# download nltk stuff if missing
fetch_nltk_packages()

# get logger and setup lemmatizer/stop words
logger = get_logger()
lemmatizer = WordNetLemmatizer()
stop_words_set = set(stopwords.words("english"))

# simple dict to map contractions to full words
CONTRACTION_RULES = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am",
}

def clean_tweet_text(raw_text):
    # check if input is a valid string
    if not isinstance(raw_text, str):
        return ""
        
    # lowercase everything first
    val = raw_text.lower()
    
    # expand contractions like won't to will not
    for shortcut, expanded in CONTRACTION_RULES.items():
        val = val.replace(shortcut, expanded)
        
    # remove urls, user handles, hashtags, rt symbols, and all punctuation
    val = re.sub(r"http\S+|www\.\S+", "", val)
    val = re.sub(r"@\w+", "", val)
    val = re.sub(r"#", "", val)
    val = re.sub(r"\brt\b", "", val)
    val = re.sub(r"[^a-zA-Z\s]", "", val)
    
    # merge duplicate spaces and strip whitespace from ends
    val = " ".join(val.split()).strip()
    return val

def process_tokens(cleaned_text):
    # handle empty text case
    if not cleaned_text:
        return ""
    # tokenize the text using nltk word tokenizer
    tokens = word_tokenize(cleaned_text)
    
    # remove stopwords and words with length <= 1, then lemmatize to base form
    filtered = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words_set and len(t) > 1
    ]
    return " ".join(filtered)

def clean_single_text(text):
    """Full text cleanup helper."""
    return process_tokens(clean_tweet_text(text))

def clean_dataframe(df, text_col="text"):
    logger.info(f"Cleaning dataframe with {len(df):,} items...")
    
    # apply clean_tweet_text first then process_tokens to get the clean output columns
    df["cleaned_text"] = df[text_col].apply(clean_tweet_text)
    df["processed_text"] = df["cleaned_text"].apply(process_tokens)
    
    # drop rows that ended up empty after we cleaned them
    before_drop = len(df)
    df = df[df["processed_text"].str.len() > 0].reset_index(drop=True)
    after_drop = len(df)
    
    logger.info(f"Removed {before_drop - after_drop:,} empty rows. Kept {after_drop:,} rows.")
    return df

def run_preprocessing(limit=DEFAULT_LIMIT, out_file="cleaned_tweets.csv"):
    # initialize folders if they don't exist yet
    init_workspace()
    
    logger.info("*" * 50)
    logger.info("RUNNING PREPROCESSING PIPELINE")
    logger.info("*" * 50)
    
    # load raw csv file from data folder
    df = load_raw_csv(limit=limit)
    
    # run the clean dataframe function
    df = clean_dataframe(df)
    
    # count how many words we have on average after processing
    word_counts = df["processed_text"].apply(lambda s: len(s.split()))
    logger.info(f"Average words per tweet: {word_counts.mean():.1f}")
    
    # save the processed dataframe columns we need
    save_csv(df[["target", "text", "cleaned_text", "processed_text"]], out_file)
    logger.info("PREPROCESSING PIPELINE DONE")
    logger.info("*" * 50)
    return df

if __name__ == "__main__":
    run_preprocessing()
