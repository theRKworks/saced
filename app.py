import streamlit as st
import pandas as pd
from main import process_csv, chatbot_on_sentiment_analysis
import time

# Streamlit App Title
st.set_page_config(page_title="Customer Sentiment Analysis", layout="wide")
st.title("📊 Customer Sentiment Analysis Tool")

# File Upload Section
uploaded_file = st.file_uploader("Upload a CSV file containing customer reviews", type=["csv"])

if uploaded_file:
    st.success("✅ File Uploaded Successfully!")
    
    # Save uploaded file to disk
    file_path = "uploaded_data.csv"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Read the raw contents of the file for debugging
    raw_data = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    
    # Display raw file content (first 500 characters)
    st.write("### Raw File Content Preview:")
    st.text(raw_data[:500])  # Display first 500 characters to check the file content

    # Now attempt to load the CSV into pandas
    try:
        # Open the file using 'open' to handle encoding issues
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            df = pd.read_csv(f)
        
        # Log the column names and first few rows for debugging
        st.write("### 📂 CSV Columns:")
        st.write(df.columns)  # Display column names

        st.write("### 📂 Data Preview:")
        st.write(df.head())  # Show first few rows

        # Ensure 'Review' column exists
        if "Review" not in df.columns:
            st.error("❌ The CSV file must contain a column named 'Review' for sentiment analysis.")
        else:
            st.success("✅ File processed successfully!")
            # Continue with processing, e.g., sentiment analysis...
            
    except Exception as e:
        st.error(f"❌ An error occurred while loading the file: {e}")


    # Chatbot for Sentiment Analysis Discussion
    st.write("### 💬 Chatbot: Discuss Your Sentiment Analysis Results")

    user_query = st.text_input("Ask a question about the sentiment analysis results:")
    if user_query:
        response = chatbot_on_sentiment_analysis(user_query)
        st.write(f"🤖 Chatbot: {response}")

