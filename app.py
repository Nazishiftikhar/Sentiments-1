import streamlit as st
import joblib
import os
import requests
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ----------------- Download model -------------------
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

# ----------------- Email Function -------------------
def send_email_alert(subject, body, to_emails):
    try:
        sender_email = st.secrets["EMAIL_ADDRESS"]
        sender_password = st.secrets["EMAIL_PASSWORD"]
    except Exception:
        st.error("Email credentials not found. Set EMAIL_ADDRESS and EMAIL_PASSWORD in Streamlit secrets.")
        return

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(to_emails)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_emails, msg.as_string())
        st.info("🚨 Alert email sent successfully.")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# ----------------- Load model & vectorizer -------------------
model_path = download_model()
model = joblib.load(model_path)

if not os.path.exists("tfidf_vectorizer.pkl"):
    st.error("TF-IDF vectorizer file not found. Please upload it.")
    st.stop()
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ----------------- Top Menu -------------------
st.markdown("<h1 style='text-align: center;'>AI Sentiment Detection App</h1>", unsafe_allow_html=True)
menu = st.selectbox("📂 Select Page", ["Home", "Sentiment Analysis", "About"], index=0)

# ----------------- Home Page -------------------
if menu == "Home":
    st.subheader("🏠 Welcome to the Sentiment Detection System")
    st.write("""
    This web application helps analyze the **sentiment** of a given piece of text using a trained ensemble model.

    - 🧠 Built with machine learning
    - 🚨 Sends alerts on detecting **suicidal intent**
    - 📊 Accurate predictions based on TF-IDF & Voting Classifier
    
    🔍 Click below to start analyzing sentiments!
    """)
    if st.button("Go to Sentiment Analyzer"):
        st.experimental_set_query_params(menu="Sentiment Analysis")

# ----------------- Sentiment Analysis Page -------------------
elif menu == "Sentiment Analysis":
    st.subheader("🧠 Sentiment Analysis Tool")
    text_input = st.text_area("Enter text to analyze sentiment:")

    if st.button("Predict"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            X_input = vectorizer.transform([text_input])
            prediction = model.predict(X_input)[0]
            st.success(f"Predicted Sentiment: **{prediction}**")

            if prediction.lower() == "suicide":
                st.error("⚠️ Suicide sentiment detected. Alert triggered.")
                send_email_alert(
                    subject="🚨 Suicide Sentiment Detected",
                    body=f"The following message indicates suicidal intent:\n\n{text_input}",
                    to_emails=["example1@example.com", "example2@example.com"]
                )
    if st.button("Back to Home"):
        st.experimental_set_query_params(menu="Home")

# ----------------- About Page -------------------
elif menu == "About":
    st.subheader("ℹ️ About This App")
    st.write("""
    This app performs sentiment classification using a **Voting Classifier** trained on labeled text data.
    
    It uses **TF-IDF vectorization** and alerts authorities if the predicted sentiment is 'suicide'.

    **Developed by**: Nazish 
    **Model hosted on**: [Hugging Face](https://huggingface.co/karmanizafar/sentiments)

    ---
    📧 For questions or support, please contact: `karmanizafar@gmail.com`
    """)
    if st.button("Go to Sentiment Analyzer"):
        st.experimental_set_query_params(menu="Sentiment Analysis")
