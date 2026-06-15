# streamlit dashboard for interactive testing of trained models

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure the parent directory is added to system path for local package imports
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.preprocessing import clean_single_text
from src.utils import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_PATH, REPORTS_DIR

st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS to styling the dashboard page layout and card grids
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #374151;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .main-header h1 {
        color: #f9fafb;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: #9ca3af;
        font-size: 1.0rem;
        margin: 0.4rem 0 0 0;
    }
    
    .stat-card {
        background: #f9fafb;
        padding: 1.3rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }
    
    .pos-card {
        background: #f3f4f6;
        border: 1px solid #d1d5db;
    }
    
    .neg-card {
        background: #e5e7eb;
        border: 1px solid #9ca3af;
    }
    
    .neu-card {
        background: #f3f4f6;
        border: 1px solid #d1d5db;
    }
    
    .metric-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #111827;
    }
    
    .metric-tag {
        font-size: 0.85rem;
        color: #4b5563;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .label-pos {
        color: #111827;
        font-size: 1.6rem;
        font-weight: 700;
    }
    
    .label-neg {
        color: #374151;
        font-size: 1.6rem;
        font-weight: 700;
    }
    
    div[data-testid="stSidebar"] {
        background: #111827;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_models():
    # load vectorizer and models from target directory
    models = {}
    
    # load the saved tfidf vectorizer first
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(vec_path):
        with open(vec_path, "rb") as f:
            models["tfidf"] = pickle.load(f)
            
    # load naive bayes classifier
    nb_path = os.path.join(MODELS_DIR, "naive_bayes.pkl")
    if os.path.exists(nb_path):
        with open(nb_path, "rb") as f:
            models["naive_bayes"] = pickle.load(f)
            
    # load logistic regression model
    lr_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
    if os.path.exists(lr_path):
        with open(lr_path, "rb") as f:
            models["logistic_regression"] = pickle.load(f)
            
    # load bidirectional lstm and its word tokenizer
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    tok_path = os.path.join(MODELS_DIR, "lstm_tokenizer.pkl")
    if os.path.exists(lstm_path) and os.path.exists(tok_path):
        from tensorflow.keras.models import load_model
        models["lstm"] = load_model(lstm_path)
        with open(tok_path, "rb") as f:
            models["lstm_tokenizer"] = pickle.load(f)
            
    return models

@st.cache_data
def get_metrics():
    path = os.path.join(REPORTS_DIR, "model_comparison.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def get_df():
    path = os.path.join(PROCESSED_DATA_PATH, "cleaned_tweets.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def analyze_tweet(text, model_name, models):
    # Apply raw text clean helper (URLs removal, lowercasing, etc.)
    cleaned = clean_single_text(text)
    
    # check type and route to correct model predictions
    if model_name in ["naive_bayes", "logistic_regression"]:
        if "tfidf" not in models or model_name not in models:
            return None, None
        # get predictions from traditional machine learning models
        feats = models["tfidf"].transform([cleaned])
        clf = models[model_name]
        pred = clf.predict(feats)[0]
        probs = clf.predict_proba(feats)[0]
        return int(pred), float(probs[pred])
        
    elif model_name == "lstm":
        if "lstm" not in models or "lstm_tokenizer" not in models:
            return None, None
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seqs = models["lstm_tokenizer"].texts_to_sequences([cleaned])
        # pad sequence token strings to length 100 for keras recurrent layers
        padded = pad_sequences(seqs, maxlen=100, padding="post", truncating="post")
        prob = models["lstm"].predict(padded, verbose=0)[0][0]
        pred = 1 if prob >= 0.5 else 0
        conf = prob if pred == 1 else (1.0 - prob)
        return int(pred), float(conf)
        
    return None, None

def main():
    st.markdown("""
    <div class="main-header">
        <h1>Sentiment Analyzer</h1>
        <p>Real-time prediction and model exploration tool (Project Acharya Submission)</p>
    </div>
    """, unsafe_allow_html=True)

    # load cached objects and load dataset
    models = load_all_models()
    metrics = get_metrics()
    df = get_df()

    # draw the selection box in streamlit sidebar widget
    with st.sidebar:
        st.markdown("<h3 style='color:#111827;'>Settings</h3>", unsafe_allow_html=True)
        choices = []
        if "naive_bayes" in models: choices.append("naive_bayes")
        if "logistic_regression" in models: choices.append("logistic_regression")
        if "lstm" in models: choices.append("lstm")
        
        # check if any models actually exists
        if not choices:
            st.error("No saved models found in /models directory. Run training first!")
            return
            
        mapping = {
            "naive_bayes": "Naive Bayes (Baseline)",
            "logistic_regression": "Logistic Regression (Linear)",
            "lstm": "Bidirectional LSTM (Deep Learning)"
        }
        
        selected = st.selectbox("Choose Model", choices, format_func=lambda s: mapping.get(s, s))
        
        st.markdown("---")
        st.markdown("""
        <div style='color:#111827; font-size:1.1rem; font-weight:600;'>sameera acharya</div>
        <div style='color:#4b5563; font-size:0.95rem; font-weight:500; margin-bottom:0.2rem;'>ML intern</div>
        <div style='color:#718096; font-size:0.85rem;'>project Sentiment Analysis pipeline</div>
        """, unsafe_allow_html=True)

    # Define primary application sections via tabs
    tab_predict, tab_chat, tab_compare, tab_plots, tab_data = st.tabs([
        "Prediction", "Sentiment Chatbot", "Compare Models", "Performance Plots", "Dataset View"
    ])

    with tab_predict:
        st.markdown("### Enter text to classify:")
        
        # Cache text state across widget adjustments
        if "tweet_input" not in st.session_state:
            st.session_state.tweet_input = ""
            
        # Synchronize text box value when selecting presets
        def on_preset_change():
            if st.session_state.chosen_preset:
                st.session_state.tweet_input = st.session_state.chosen_preset
            else:
                st.session_state.tweet_input = ""
                
        text_input = st.text_area(
            "Tweet / Review text:",
            key="tweet_input",
            placeholder="Type your review here...",
            height=120
        )
        
        c_btn, c_sample = st.columns([1, 3])
        with c_btn:
            submit = st.button("Analyze Text", type="primary", use_container_width=True)
        with c_sample:
            # show preset examples to pick from
            presets = [
                "This is an absolutely fantastic product! The quality is amazing.",
                "Worst experience ever. The product arrived late and broke on day one.",
                "It works fine I guess, nothing special but gets the job done.",
                "Really sad that it did not work out. Had high expectations.",
                "Highly recommended, super quick shipping and excellent customer care!",
                "I love the new design. It is sleek, comfortable, and very modern.",
                "Completely useless. Do not waste your money on this garbage.",
                "The customer support was extremely rude and unhelpful.",
                "Exceeded all my expectations. Absolutely brilliant service!",
                "Average quality. It is okay for the price, but could be better.",
                "I am so happy with this purchase! Worth every single penny.",
                "Terrible. It doesn't work as advertised and the setup was a nightmare.",
                "Works perfectly out of the box. Simple, clean, and fast.",
                "I will never buy from this brand again. Terrible quality control.",
                "A solid product with great battery life and clear display.",
                "Highly disappointed. The item arrived damaged and scratched.",
                "The best customer experience I have had in a long time.",
                "It is fine, but definitely overpriced for what it offers.",
                "Super fast shipping, neat packaging, and the product is top notch!",
                "Extremely poor performance. It constantly freezes and crashes."
            ]
            chosen_preset = st.selectbox(
                "Choose a sample text:", 
                [""] + presets, 
                key="chosen_preset",
                on_change=on_preset_change
            )

        # perform class prediction using chosen settings
        if submit and text_input:
            with st.spinner("Classifying sentiment..."):
                pred, conf = analyze_tweet(text_input, selected, models)
                
            if pred is not None:
                st.markdown("---")
                # draw result indicators with custom styling
                col_res, col_conf, col_details = st.columns(3)
                
                with col_res:
                    status = "Positive" if pred == 1 else "Negative"
                    style = "pos" if pred == 1 else "neg"
                    st.markdown(f"""
                     <div class="stat-card {style}-card">
                        <div class="metric-tag">Result</div>
                        <div class="label-{style}">{status}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_conf:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="metric-tag">Confidence Score</div>
                        <div class="metric-title">{conf*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_details:
                    model_title = mapping.get(selected, selected)
                    st.markdown(f"""
                    <div class="stat-card neu-card">
                        <div class="metric-tag">Classifier Type</div>
                        <div style="font-size:1.2rem; font-weight:600; color:#111827; margin-top:0.4rem;">
                            {model_title}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf * 100,
                    title={"text": "Confidence Level", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#111827"},
                        "steps": [
                            {"range": [0, 50], "color": "#f3f4f6"},
                            {"range": [50, 75], "color": "#e5e7eb"},
                            {"range": [75, 100], "color": "#d1d5db"}
                        ]
                    }
                ))
                fig.update_layout(height=280)
                st.plotly_chart(fig, use_container_width=True)
                
                # compare performance against other saved classifiers
                st.markdown("### Comparison Across Models")
                c_cols = st.columns(len(choices))
                for c_col, ch in zip(c_cols, choices):
                    p_val, c_val = analyze_tweet(text_input, ch, models)
                    with c_col:
                        lbl = "Positive" if p_val == 1 else "Negative"
                        st.metric(
                            mapping.get(ch, ch),
                            lbl,
                            f"Confidence: {c_val*100:.1f}%"
                        )

    with tab_chat:
        st.markdown("### Sentiment Chatbot")
        st.markdown("Interact with the selected classifier model through a conversational interface.")
        
        # Initialize chat message history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! Send me a message and I will analyze its sentiment."}
            ]
            
        # Display chat messages from history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # React to user input
        if prompt := st.chat_input("Say something..."):
            # Display user message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Predict using selected classifier
            with st.spinner("Analyzing..."):
                pred, conf = analyze_tweet(prompt, selected, models)
                
            if pred is not None:
                lbl = "Positive" if pred == 1 else "Negative"
                reply = f"I think that text is **{lbl}** (Confidence: {conf*100:.1f}%)"
            else:
                reply = "Oops! I encountered an error running the classifier model."
                
            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    with tab_compare:
        # display results of metrics comparison
        st.markdown("### Model Comparison Metrics")
        if metrics:
            comp_df = pd.DataFrame(metrics).T
            comp_df.index = [n.replace("_", " ").title() for n in comp_df.index]
            
            st.dataframe(
                comp_df.style.format("{:.4f}").highlight_max(axis=0, color="#e5e7eb"),
                use_container_width=True
            )
            
            metrics_keys = ["accuracy", "precision", "recall", "f1_score"]
            fig = go.Figure()
            color_scheme = ["#111827", "#4b5563", "#9ca3af"]
            
            for index, (m_name, row) in enumerate(comp_df.iterrows()):
                fig.add_trace(go.Bar(
                    name=m_name,
                    x=[m.replace("_", " ").title() for m in metrics_keys],
                    y=[row.get(m, 0) for m in metrics_keys],
                    marker_color=color_scheme[index % len(color_scheme)]
                ))
                
            fig.update_layout(
                barmode="group",
                title="Comparative Graph",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                template="plotly_white",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Metrics not found. Run model training first.")

    with tab_plots:
        # list and display the static figures generated by visualize.py
        st.markdown("### Generated Static Plots")
        if os.path.exists(FIGURES_DIR):
            plots = [p for p in os.listdir(FIGURES_DIR) if p.endswith(".png")]
            if plots:
                st_cols = st.columns(2)
                for index, plot_file in enumerate(plots):
                    col_target = st_cols[index % 2]
                    with col_target:
                        caption = plot_file.replace(".png", "").replace("_", " ").title()
                        st.markdown(f"**{caption}**")
                        st.image(os.path.join(FIGURES_DIR, plot_file), use_container_width=True)
            else:
                st.info("No generated plots found in figures directory.")
        else:
            st.info("Figures directory missing. Train models first.")

    with tab_data:
        # show the dataset preview data explorer
        st.markdown("### Data Explorer")
        if df is not None:
            col_a, col_b, col_c = st.columns(3)
            with col_a: st.metric("Total Records", f"{len(df):,}")
            with col_b: st.metric("Positives", f"{(df['target']==1).sum():,}")
            with col_c: st.metric("Negatives", f"{(df['target']==0).sum():,}")
            
            filter_val = st.selectbox("Filter rows:", ["All", "Positive", "Negative"])
            if filter_val == "Positive":
                target_df = df[df["target"] == 1]
            elif filter_val == "Negative":
                target_df = df[df["target"] == 0]
            else:
                target_df = df
                
            st.dataframe(target_df.head(50), use_container_width=True, height=350)
            
            if "processed_text" in df.columns:
                df["token_len"] = df["processed_text"].apply(lambda s: len(str(s).split()))
                fig = px.histogram(
                    df, x="token_len", color=df["target"].map({0: "Negative", 1: "Positive"}),
                    title="Length of Preprocessed Tweets",
                    labels={"token_len": "Words count", "color": "Sentiment Class"},
                    color_discrete_sequence=["#111827", "#9ca3af"],
                    nbins=40, barmode="overlay", opacity=0.75
                )
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cleaned CSV not found in dataset folder.")

if __name__ == "__main__":
    main()
