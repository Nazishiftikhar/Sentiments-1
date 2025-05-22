import streamlit as st
import joblib
import os
import requests
import numpy as np

# Function to download large model from Hugging Face
@st.cache_data
def download_model():
    url = "https://huggingface.co/karmanizafar/sentiments/blob/main/voting_model.pkl"
    filename = "voting_model.pkl"
    if not os.path.exists(filename):
        with st.spinner("Downloading model..."):
            response = requests.get(url)
            with open(filename, "wb") as f:
                f.write(response.content)
    return filename

# Download and load model
model_path = download_model()
model = joblib.load(model_path)

# Load small tf-idf vectorizer from local
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
        st.success(f"Predicted Sentiment: {prediction}")
