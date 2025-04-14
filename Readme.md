Sentiment Analysis & Customer Engagement Dashboard (SACED)
🎯 Submission for AI Hackathon – Kshitij 2025
🚀 Project Overview
SACED (Sentiment Analysis & Customer Engagement Dashboard) is an AI-powered tool that performs aspect-based sentiment analysis on customer reviews and provides insights through an interactive dashboard. The platform also includes a chatbot for discussing sentiment trends and a feature to identify loyal customers based on positive reviews.

🎯 Key Features
✅ Aspect-Based Sentiment Analysis – Uses GPT-4 to extract sentiments for different aspects in customer reviews.
✅ Chatbot Integration – Enables users to chat with an AI assistant to interpret sentiment insights.
✅ Live Dashboard – Displays sentiment trends and key metrics in real time.
✅ Loyal Customer Identification – Detects the most engaged and satisfied customers based on their reviews.
✅ CSV Processing – Allows users to upload CSV files and get sentiment analysis results in tabular form.

🛠️ Tech Stack
Frontend: Streamlit
Backend: Python (Flask/FastAPI)
AI Models: GPT-4 (OpenAI API)
Data Processing: Pandas, LangChain
Visualization: Plotly, Matplotlib

📌 Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/theRKworks/saced.git
cd saced

2️⃣ Create a Virtual Environment
python -m venv venv

Activate it:

Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Up OpenAI API Key
Create a .env file and add:
OPENAI_API_KEY=your_openai_api_key

5️⃣ Run the Application
streamlit run app.py

📊 How to Use
1️⃣ Upload a CSV file containing a "Review" column.
2️⃣ The tool will analyze sentiment for different aspects in each review.
3️⃣ View sentiment trends on the dashboard.
4️⃣ Chat with the AI assistant to gain deeper insights.
5️⃣ Download the processed CSV file with sentiment scores.

📌 Demo Video
📽️ [Link to demo video (To be added)]

👥 Contributors
Rishi Kushwaha (@theRKworks)
[Your Friend’s Name]
📜 License
MIT License