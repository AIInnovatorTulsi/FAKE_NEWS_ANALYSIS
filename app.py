import streamlit as st
import joblib

# Load saved model and vectorizer
vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("lr_model.joblib")

# App title
st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")

# Input box
news_input = st.text_area("News Article:", "")

# Button click
if st.button("Check News"):
    if news_input.strip():
        # Transform input text
        transform_input = vectorizer.transform([news_input])

        # Predict
        prediction = model.predict(transform_input)

        # Show result
        if prediction[0] == 1:
            st.success("The News is Real!")
        else:
            st.error("The News is Fake!")
    else:
        st.warning("Please enter some text to analyze.")