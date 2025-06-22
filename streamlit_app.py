import streamlit as st
import joblib
import os
import re

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_PATH, "fixed_spam_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_PATH, "fixed_tfidf_vectorizer.pkl"))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()


st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("📧 Spam Email Detector")
st.markdown("Type a message or email content below:")

message = st.text_area("Enter your message:")

if st.button("Predict"):
    cleaned = clean_text(message)
    if not cleaned:
        st.warning("Please enter a valid message.")
    else:
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        if prediction == 1:
            st.error("🚫 This is SPAM!")
        else:
            st.success("✅ This is NOT spam.")
