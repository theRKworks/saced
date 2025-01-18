import streamlit as st
import pandas as pd
from main import analyze_sentiment  # Import function from main.py

st.title("📊 Sentiment Analysis Tool")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "review" in df.columns:  # Assuming reviews are in the "review" column
        df["Sentiment Score"] = df["review"].apply(analyze_sentiment)

        # Display updated DataFrame
        st.write("### Sentiment Analysis Results")
        st.dataframe(df.head())

        # Download modified file
        st.download_button(
            label="📥 Download Modified CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_results.csv",
            mime="text/csv"
        )

        # Sentiment Distribution Chart
        st.write("### Sentiment Score Distribution")
        st.bar_chart(df["Sentiment Score"].value_counts())

    else:
        st.error("Error: The CSV file must contain a 'review' column.")
