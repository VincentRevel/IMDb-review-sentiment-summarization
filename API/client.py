import streamlit as st
import requests

# FastAPI service URL
FASTAPI_URL = "http://localhost:8000/summarize/"

# Streamlit application
st.title("Text Summarization with Pegasus")

st.write("Enter the text you want to summarize below:")

# Text input
input_text = st.text_area("Input Text")

# Button to submit the text for summarization
if st.button("Summarize"):
    if input_text:
        with st.spinner("Summarizing..."):
            # Send the text to the FastAPI summarization service
            response = requests.post(FASTAPI_URL, json={"text": input_text})
            
            if response.status_code == 200:
                summary = response.json().get("summary")
                st.write("**Summary:**")
                st.write(summary)
            else:
                st.error("An error occurred: " + response.text)
    else:
        st.warning("Please enter text to summarize.")