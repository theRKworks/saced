import pandas as pd
from transformers import FlamingoProcessor, FlamingoForConditionalGeneration
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Your code here
# Load Flamingo Model for Sentiment Analysis
processor = FlamingoProcessor.from_pretrained("openai/flamingo")
model = FlamingoForConditionalGeneration.from_pretrained("openai/flamingo")

# Initialize LangChain Chat Model (Use any LLM - GPT-4, Llama, etc.)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)  

# Memory for tracking conversation context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def analyze_sentiment(text_input):
    """
    - Analyzes the given text with Flamingo to extract sentiment scores.
    - Stores sentiment results in LangChain memory for future chatbot interactions.
    - Returns only the sentiment score.
    """

    # Step 1: Process Text with Flamingo for Sentiment Analysis
    inputs = processor(text_input, return_tensors="pt")
    output = model.generate(**inputs)
    sentiment_result = processor.decode(output[0])

    # Step 2: Store Sentiment Analysis Result in Memory (For Future Use)
    memory.save_context({"analysis_result": sentiment_result}, {"question": "Initial Sentiment Analysis"})

    return sentiment_result

def process_csv(file_path, analyze_sentiment):
    """
    Reads a CSV file, analyzes sentiment for each row, adds a new column, and saves the modified file.

    Parameters:
    - file_path (str): Path to the CSV file.
    - analyze_sentiment (function): A function that takes text input and returns a sentiment score.

    Returns:
    - modified_df (DataFrame): The modified DataFrame with an added 'Sentiment Score' column.
    - output_file (str): Path to the saved modified CSV file.
    """

    try:
        # Read CSV file (handling different encodings)
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
    
# Define Chatbot Prompt Template
template = """You are an AI chatbot providing insights based on sentiment analysis.
The user has uploaded a dataset, and the sentiment analysis results are:

{analysis_results}

Use this data to answer any questions about sentiment trends, issues, or improvement suggestions.

User: {question}
AI:"""

prompt = PromptTemplate(template=template, input_variables=["analysis_results", "question"])

# Define the LangChain Chatbot Chain
chatbot_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

def chatbot_on_sentiment_analysis(user_query):
    """
    Allows the user to chat about sentiment analysis results.
    
    Parameters:
    - user_query (str): User's question related to sentiment analysis.
    
    Returns:
    - str: Chatbot's response.
    """
    response = chatbot_chain.run({"analysis_results": memory.load_memory_variables({})["chat_history"], "question": user_query})
    return response

