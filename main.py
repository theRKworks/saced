import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time

# Download the VADER sentiment lexicon (only required once)
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer (VADER)
sia = SentimentIntensityAnalyzer()

#copy api key from openai
openai_api_key = "Enter your api key"

llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

# Memory for tracking conversation context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# =========================
# 🔹 Function: Sentiment Analysis
# =========================
def analyze_sentiment(text_input):
    """
    Uses VADER Sentiment Analysis to determine sentiment polarity.
    
    Returns:
    - 'POSITIVE' if compound score > 0.05
    - 'NEGATIVE' if compound score < -0.05
    - 'NEUTRAL' otherwise
    """
    score = sia.polarity_scores(text_input)["compound"]
    if score > 0.05:
        return "POSITIVE"
    elif score < -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

# =========================
# 🔹 Function: Process CSV File
# =========================
def process_csv(file_path, progress_bar):
    """
    Reads a CSV file, performs sentiment analysis using GPT-4, and saves the modified file.

    Parameters:
    - file_path (str): Path to the CSV file.
    - progress_bar: Streamlit progress bar to display processing status.

    Returns:
    - modified_df (DataFrame): DataFrame with an added 'Sentiment Score' column.
    - output_file (str): Path to the saved modified CSV file.
    """
    try:
        # Load CSV file (handling different encodings)
        df = pd.read_csv(file_path, encoding="utf-8", errors="ignore")

        # Check if 'Review' column exists
        if "review" not in df.columns:
            raise ValueError("CSV must contain a column named 'Review' for sentiment analysis.")

        # Apply sentiment analysis to each review with progress bar
        total_rows = len(df)
        for idx, row in df.iterrows():
            sentiment = analyze_sentiment(row["review"])
            df.at[idx, "Sentiment Score"] = sentiment

            # Update progress bar
            progress_bar.progress(int((idx + 1) / total_rows * 100))
            time.sleep(0.1)  # Slight delay to make progress bar visible

        # Save the modified CSV file
        output_file = "modified_reviews.csv"
        df.to_csv(output_file, index=False, encoding="utf-8")

        return df, output_file

    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None


# =========================
# 🔹 Function: Chatbot for Sentiment Analysis
# =========================
def chatbot_on_sentiment_analysis(user_query):
    """
    Allows the user to chat about sentiment analysis results.

    Parameters:
    - user_query (str): User's question related to sentiment analysis.

    Returns:
    - str: Chatbot's response.
    """
    # Chatbot Prompt Template
    template = """You are an AI chatbot providing insights based on sentiment analysis.
    The user has uploaded a dataset, and the sentiment analysis results are:

    {analysis_results}

    Use this data to answer any questions about sentiment trends, issues, or improvement suggestions.

    User: {question}
    AI:"""

    prompt = PromptTemplate(template=template, input_variables=["analysis_results", "question"])

    chatbot_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    response = chatbot_chain.run({"analysis_results": memory.load_memory_variables({})["chat_history"], "question": user_query})
    return response
