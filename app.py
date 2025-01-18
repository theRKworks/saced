import streamlit as st
import pandas as pd
from main import process_csv, chatbot_on_sentiment_analysis

# Streamlit App Title
st.set_page_config(page_title="Customer Sentiment Analysis", layout="wide")
st.title("📊 Customer Sentiment Analysis Tool")

# File Upload Section
uploaded_file = st.file_uploader("Upload a CSV file containing customer reviews", type=["csv"])

if uploaded_file:
    st.success("✅ File Uploaded Successfully!")
    
    # Save uploaded file
    file_path = "uploaded_data.csv"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the CSV File for Sentiment Analysis
    df, modified_file = process_csv(file_path)

    if df is not None:
        st.write("### 📂 Processed Data Preview")
        st.dataframe(df.head(10))  # Show first 10 rows

        # Download Button for Modified CSV
        st.download_button(
            label="📥 Download Processed CSV with Sentiment Scores",
            data=open(modified_file, "rb"),
            file_name="modified_reviews.csv",
            mime="text/csv"
        )

        # Chatbot for Sentiment Analysis Discussion
        st.write("### 💬 Chatbot: Discuss Your Sentiment Analysis Results")

        user_query = st.text_input("Ask a question about the sentiment analysis results:")
        if user_query:
            response = chatbot_on_sentiment_analysis(user_query)
            st.write(f"🤖 Chatbot: {response}")
