import pandas as pd

# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
from transformers import pipeline

# Load Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis") 

# Your code here


# Initialize LangChain Chat Model (Use any LLM - GPT-4, Llama, etc.)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)  

# Memory for tracking conversation context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# def analyze_sentiment(text_input):
#     """
#     Uses OpenFlamingo to analyze sentiment from text input.
#     """
#     model, image_processor, tokenizer = create_model_and_transforms(
#         vision_encoder_path="ViT",  # Use a vision transformer model
#         lang_encoder_path="facebook/opt-125m",  # Use a language model
#     )

#     inputs = tokenizer(text_input, return_tensors="pt")
#     outputs = model.generate(**inputs)
#     sentiment_result = tokenizer.decode(outputs[0])
#     return sentiment_result

def analyze_sentiment(text_input):
    """
    Uses a pre-trained Hugging Face model to analyze sentiment of a given text.
    """
    result = sentiment_pipeline(text_input)
    return result[0]['label']


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
    template = """You are an AI chatbot providing insights based on sentiment analysis.
    The user has uploaded a dataset, and the sentiment analysis results are:

    {analysis_results}

    Use this data to answer any questions about sentiment trends, issues, or improvement suggestions.

    User: {question}
    AI:"""
    response = chatbot_chain.run({"analysis_results": memory.load_memory_variables({})["chat_history"], "question": user_query})
    return response

