import streamlit as st
import pandas as pd
import plotly.express as px
from main import process_csv, chatbot_on_sentiment_analysis
from dotenv import load_dotenv
import os

# Streamlit App Title
st.set_page_config(page_title="Customer Sentiment Analysis", layout="wide")
st.title("📊 Customer Sentiment Analysis Tool")

# File Upload Section
uploaded_file = st.file_uploader("Upload a CSV file containing customer reviews", type=["csv"])

if uploaded_file:
    st.success("✅ File Uploaded Successfully!")

    # Read uploaded file into DataFrame
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8", errors="replace")

        # Log the column names and preview data
        st.write("### 📂 CSV Columns:")
        st.write(df.columns)

        st.write("### 📂 Data Preview:")
        st.dataframe(df.head())

        # Ensure the 'Review' column exists
        if "Review" not in df.columns:
            st.error("❌ The CSV file must contain a column named 'Review' for sentiment analysis.")
        else:
            st.success("✅ File contains the required 'Review' column!")

            # Process the CSV (aspect-based sentiment analysis)
            processed_df, processed_file = process_csv(uploaded_file)

            if processed_df is not None:
                # Display processed results
                st.write("### 📊 Processed Data Preview:")
                st.dataframe(processed_df.head())

                # Download button for processed CSV
                with open(processed_file, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed CSV",
                        data=f,
                        file_name="processed_reviews.csv",
                        mime="text/csv"
                    )

                # =========================
                # 🔹 Live Sentiment Dashboard
                # =========================
                st.header("📊 Sentiment Analysis Dashboard")

                # Sentiment Distribution
                sentiment_counts = processed_df["Aspect-Based Sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]

                # Display bar chart
                fig = px.bar(
                    sentiment_counts, 
                    x="Sentiment", 
                    y="Count", 
                    title="Sentiment Distribution",
                    color="Sentiment",
                    text_auto=True
                )
                st.plotly_chart(fig)

                # =========================
                # 🔹 Loyal Customer Identification
                # =========================
                st.header("🌟 Loyal Customers")

                loyal_customers_df, loyal_customers_file = get_loyal_customers(processed_df)

                if loyal_customers_df is not None:
                    st.write("### 🎖 Top Loyal Customers:")
                    st.dataframe(loyal_customers_df)

                    with open(loyal_customers_file, "rb") as f:
                        st.download_button(
                            label="📥 Download Loyal Customers List",
                            data=f,
                            file_name="loyal_customers.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("⚠ No loyal customers identified.")

            else:
                st.error("❌ Error in processing file!")

    except Exception as e:
        st.error(f"❌ An error occurred while loading the file: {e}")

# =========================
# 🔹 Chatbot Section
# =========================
st.sidebar.title("💬 Sentiment Chatbot")
user_query = st.sidebar.text_input("Ask about the sentiment analysis results:")

if user_query:
    with st.spinner("🤖 Thinking..."):
        response = chatbot_on_sentiment_analysis(user_query)
        st.sidebar.write("🧠 **Chatbot Response:**")
        st.sidebar.success(response)
