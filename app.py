import streamlit as st
import joblib
import os
import requests
import numpy as np

# Function to download model from Hugging Face
@st.cache_data
def download_model():
    url = "https://huggingface.co/karmanizafar/sentiments/resolve/main/voting_model.pkl"
    filename = "voting_model.pkl"
    if not os.path.exists(filename):
        with st.spinner("Downloading model..."):
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
    return filename

# Download and load model
model_path = download_model()
model = joblib.load(model_path)

# Load small tf-idf vectorizer from local
if not os.path.exists("tfidf_vectorizer.pkl"):
    st.error("TF-IDF vectorizer file not found. Please upload it.")
    st.stop()

vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("Sentiment Analysis App")
text_input = st.text_area("Enter text to analyze sentiment:")

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        X_input = vectorizer.transform([text_input])
        prediction = model.predict(X_input)[0]
        st.success(f"Predicted Sentiment: **{prediction}**")
