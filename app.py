import streamlit as st
import joblib
import os
import requests
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

# Email Alert Function
def send_email_alert(subject, body, to_emails):
    try:
        sender_email = st.secrets["EMAIL_ADDRESS"]
        sender_password = st.secrets["EMAIL_PASSWORD"]
    except Exception:
        st.error("Email credentials not found. Please set them in Streamlit secrets.")
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

# Download and load model
model_path = download_model()
model = joblib.load(model_path)

# Load vectorizer
if not os.path.exists("tfidf_vectorizer.pkl"):
    st.error("TF-IDF vectorizer file not found. Please upload it.")
    st.stop()
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Sidebar Menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Sentiment Analyzer", "About"])

if menu == "Sentiment Analyzer":
    st.title("🧠 Sentiment Analysis App")
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
                # Send email
                send_email_alert(
                    subject="🚨 Suicide Sentiment Detected",
                    body=f"The following message indicates suicidal intent:\n\n{text_input}",
                    to_emails=["example1@example.com", "example2@example.com"]  # Replace with real addresses
                )

elif menu == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This sentiment analysis app is powered by a Voting Classifier model and TF-IDF vectorization.
    
    It can detect potentially dangerous sentiments like **suicide** and trigger alerts to specified contacts.

    **Developed by:** Nazish  
    **Model hosted on:** [Hugging Face](https://huggingface.co/naziii/sentiments)
    """)

