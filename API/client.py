import streamlit as st
import requests

# # FastAPI service URLs
# FASTAPI_SUMMARIZE_URL = "http://localhost:8000/summarize/"
# FASTAPI_SENTIMENT_URL = "http://localhost:8000/sentiment/"

# # Streamlit application
# st.title("Text Processing with FastAPI")

# # Summarization Section
# st.header("Text Summarization with Pegasus")
# st.write("Enter the text you want to summarize below:")

# # Text input for summarization
# input_text_summarize = st.text_area("Input Text for Summarization", key="summarize")

# # Button to submit the text for summarization
# if st.button("Summarize"):
#     if input_text_summarize:
#         with st.spinner("Summarizing..."):
#             # Send the text to the FastAPI summarization service
#             response = requests.post(FASTAPI_SUMMARIZE_URL, json={"text": input_text_summarize})
            
#             if response.status_code == 200:
#                 summary = response.json().get("summary")
#                 st.write("**Summary:**")
#                 st.write(summary)
#             else:
#                 st.error("An error occurred: " + response.text)
#     else:
#         st.warning("Please enter text to summarize.")

# # Sentiment Analysis Section
# st.header("Text Sentiment Analysis")
# st.write("Enter the text you want to analyze below:")

# # Text input for sentiment analysis
# input_text_sentiment = st.text_area("Input Text for Sentiment Analysis", key="sentiment")

# # Button to submit the text for sentiment analysis
# if st.button("Analyze"):
#     if input_text_sentiment:
#         with st.spinner("Analyzing..."):
#             # Send the text to the FastAPI sentiment analysis service
#             response = requests.post(FASTAPI_SENTIMENT_URL, json={"text": input_text_sentiment})
            
#             if response.status_code == 200:
#                 sentiment = response.json().get("sentiment")
#                 st.write("**Sentiment:**")
#                 st.write(sentiment)
#             else:
#                 st.error("An error occurred: " + response.text)
#     else:
#         st.warning("Please enter text to analyze.")


# FastAPI service URL
FASTAPI_URL = "http://localhost:8000/analyze/"

# Streamlit application
st.title("Movie Review Analysis")

st.write("Enter the movie title to analyze its reviews:")

# Text input
movie_title = st.text_input("Movie Title")

# Button to submit the movie title for analysis
if st.button("Analyze"):
    if movie_title:
        with st.spinner("Analyzing..."):
            # Send the movie title to the FastAPI service
            response = requests.post(FASTAPI_URL, json={"title": movie_title})
            
            if response.status_code == 200:
                reviews = response.json()
                st.write("**Reviews Analysis:**")
                for review in reviews:
                    st.write(f"**Summary:** {review['summary']}")
                    st.write(f"**Sentiment:** {review['sentiment']}")
                    st.write("---")
            else:
                st.error("An error occurred: " + response.text)
    else:
        st.warning("Please enter a movie title.")