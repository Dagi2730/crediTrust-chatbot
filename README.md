Task 4: Interactive Chat Interface for CrediTrust AI Assistant
Overview
This task focuses on developing a clean and intuitive web-based interface for the CrediTrust AI Chat Assistant, enabling users to interact with a Retrieval-Augmented Generation (RAG) system to query customer complaints.

The goal is to create a user-friendly tool that allows non-technical stakeholders—such as customer support agents, financial analysts, and internal teams—to gain insights from historical consumer complaints data via natural language.

Objectives
Build an interactive interface using Streamlit.

Enable users to ask natural language questions about customer complaints.

Display AI-generated answers along with the retrieved source text chunks to enhance transparency.

Implement a session-aware design to maintain chat history.

Include a "Clear" button to reset the interface.

Features
Text Input Box for entering questions

"Ask" Button to submit queries

AI-Generated Answer Display using RAG

Source Context Display below each answer (retrieved text chunks)

Clear Chat Button to reset the session

Session History to track user interactions

Implementation Summary
The chatbot is powered by the following components:

Component	Description
Interface Framework	Streamlit
Embedding Model	all-MiniLM-L6-v2 (Sentence Transformers)
Vector Store	FAISS
Dataset	Preprocessed local CSV (filtered_complaints.csv)
Retrieval Pipeline	Top-k semantic similarity on vectorized complaint narratives
Response Generator	Open-source transformer LLM (local or HuggingFace-hosted)

Setup Instructions
1. Clone the Repository and Activate Environment
bash
Copy
Edit
git clone https://github.com/your-org/crediTrust-chatbot.git
cd crediTrust-chatbot
python -m venv .venv
.venv\Scripts\activate   # On Windows
# or
source .venv/bin/activate  # On macOS/Linux
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Generate the Vector Index
This step will embed complaint narratives and create the FAISS index used for retrieval.

bash
Copy
Edit
python generate_index.py
Note: To reduce latency, the script processes the first 5,000 entries from filtered_complaints.csv.

4. Launch the Streamlit App
bash
Copy
Edit
streamlit run app.py
The app will open in your default web browser at http://localhost:8501.