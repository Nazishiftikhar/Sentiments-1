import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import os
import requests
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
from streamlit_option_menu import option_menu
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

# ----------------- Top Navigation Menu -------------------
selected = option_menu(
    menu_title=None,  # No title
    options=["Home", "Sentiment Analysis", "About"],
    icons=["house", "emoji-smile", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)
# ----------------- Home Page -------------------
if selected == "Home":
    # Main title with style
    st.markdown("""
        <h1 style='text-align: center; color: #6c63ff; font-size: 40px;'>
            🏠 Welcome to the Suicidal Thought Detection System
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style='text-align: center; font-size: 18px; color: #555;'>
            Your mental health matters. This AI-powered system analyzes your text to detect early signs of suicidal thoughts and provide timely alerts.
        </p>
        <br>
    """, unsafe_allow_html=True)

    # Use two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💡 How It Works")
        st.markdown("""
        - 🧠 Uses a trained NLP model with real-world data  
        - 🚨 Sends email alerts when high-risk content is detected  
        - 🤖 Fast, simple & secure  
        - 🔍 Analyze thoughts in real-time  
        """)
        st.markdown("")

        # Optional button to move to next section
        if st.button("🚀 Start Analyzing"):
            st.experimental_set_query_params(tab="Sentiment Analysis")  # optional if you're using query params

    with col2:
        # You can add an image or animation here
        st.image("https://media.giphy.com/media/26ufnwz3wDUli7GU0/giphy.gif", caption="Stay strong. You're not alone.", use_column_width=True)

    

# ----------------- Sentiment Analysis Page -------------------
elif selected == "Sentiment Analysis":
    st.title("🧠 Suicidal Ideation Detection")
    text_input = st.text_area("Enter text to analyze sentiment:")

    if st.button("Predict"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            X_input = vectorizer.transform([text_input])
            prediction = model.predict(X_input)[0]
            st.success(f"Predicted Sentiment: **{prediction}**")

            if prediction == "Suicidal":
                st.error("⚠️ Suicide sentiment detected. Alert triggered.")
                send_email_alert(
                    subject="🚨 Suicide Sentiment Detected",
                    body=f"The following message indicates suicidal intent:\n\n{text_input}",
                    to_emails=["nazishiftikhar112@gmail.com"]
                )

# ----------------- About Page -------------------
elif selected == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This web application analyzes the **sentiment** of user-provided text.

    - 🧠 Trained with real-world data  
    - 🛡️ Sends email alerts if **suicidal intent** is detected  
    - 🚀 Simple and fast sentiment classification  

    👉 Navigate to the **Sentiment Analysis** tab above to begin.

    **Developer**: Nazish Iftikhar  
    **Model Hosted On**: [Hugging Face](https://huggingface.co/naziiiii/Sentiments/blob/main/voting_model.pkl)  
    📫 **Contact**: `nazivirk113@gmail.com`
    """)

