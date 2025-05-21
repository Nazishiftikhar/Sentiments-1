import streamlit as st
import joblib
import numpy as np

# Load vectorizer and model
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("voting_model.pkl")

st.title("Sentiment Analysis App")
text_input = st.text_area("Enter text to analyze sentiment:")

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        X_input = vectorizer.transform([text_input])
        prediction = model.predict(X_input)[0]
        st.success(f"Predicted Sentiment: {prediction}")