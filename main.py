import pandas as pd
from transformers import pipeline
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 🔹 Load Hugging Face Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("sentiment-analysis") 

# 🔹 Initialize LangChain Chat Model (Replace with your API Key if required)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)  

# 🔹 Memory to Track Conversation Context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# =========================
# 🔹 Function: Sentiment Analysis
# =========================
def analyze_sentiment(text_input):
    """
    Uses a pre-trained Hugging Face model to analyze sentiment of a given text.
    """
    result = sentiment_pipeline(text_input)
    return result[0]['label']  # Output: 'POSITIVE', 'NEGATIVE', 'NEUTRAL'

# =========================
# 🔹 Function: Process CSV File
# =========================
def process_csv(file_path):
    """
    Reads a CSV file, analyzes sentiment for each row, adds a new column, and saves the modified file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - modified_df (DataFrame): The modified DataFrame with an added 'Sentiment Score' column.
    - output_file (str): Path to the saved modified CSV file.
    """
    try:
        # Load CSV file (handling different encodings)
        df = pd.read_csv(file_path, encoding="utf-8", errors="replace")

        # Check if the expected column exists
        if "Review" not in df.columns:
            raise ValueError("CSV must contain a column named 'Review' for sentiment analysis.")

        # Apply sentiment analysis to each review
        df["Sentiment Score"] = df["Review"].apply(analyze_sentiment)

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

# =========================
# 🔹 Function: Identify Loyal Customers
# =========================
def get_loyal_customers(df):
    """
    Identifies the most loyal customers based on sentiment analysis.

    Parameters:
    - df (DataFrame): The modified DataFrame with sentiment scores.

    Returns:
    - loyal_customers_df (DataFrame): A DataFrame of the most loyal customers.
    """
    try:
        # Assuming 'Customer Name' exists and positive reviews indicate loyalty
        if "Customer Name" not in df.columns:
            raise ValueError("CSV must contain a column named 'Customer Name'.")

        # Filter customers with positive sentiment scores
        loyal_customers_df = df[df["Sentiment Score"] == "POSITIVE"][["Customer Name"]].drop_duplicates()

        # Save loyal customers CSV
        loyal_customers_file = "loyal_customers.csv"
        loyal_customers_df.to_csv(loyal_customers_file, index=False, encoding="utf-8")

        return loyal_customers_df, loyal_customers_file

    except Exception as e:
        print(f"Error identifying loyal customers: {e}")
        return None, None
