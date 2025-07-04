# CrediTrust Chatbot – RAG-Powered Complaint Insights

This project implements a chatbot for CrediTrust Financial, designed to help internal teams quickly extract insights from customer complaint narratives using a Retrieval-Augmented Generation (RAG) approach.

The chatbot allows users to ask natural-language questions about customer complaints across five key financial products and receive synthesized, evidence-based responses sourced from real complaint data.

## Project Objective

CrediTrust Financial offers a range of digital financial products, including:

- Credit Cards  
- Personal Loans  
- Buy Now, Pay Later (BNPL)  
- Savings Accounts  
- Money Transfers  

With a growing user base and thousands of incoming complaints monthly, teams need a way to analyze unstructured feedback efficiently. This chatbot enables:

- Semantic search on complaint narratives
- Contextual answers from a large language model (LLM)
- Filtering and comparison across multiple product categories

## Features

- Cleaned and preprocessed complaint dataset (CFPB)
- Text chunking and embedding using sentence-transformers
- Vector similarity search using FAISS or ChromaDB
- Context-based answer generation with an LLM
- Simple web interface for internal users
- Evaluation metrics for response quality

## Project Structure

crediTrust-chatbot/

├── data/              # Cleaned dataset (CSV)

├── notebooks/         # EDA & exploration notebooks

├── src/               # Core logic: preprocessing, embedding, RAG

├── vector_store/      # Stored vector database (FAISS or Chroma)

├── app/               # Chatbot UI interface

├── report/            # Markdown report, evaluation table

├── requirements.txt   # Python dependencies

└── README.md


## 📝 Tasks

| Task   | Description                             |
|--------|---------------------------------------|
| Task 1 | EDA and preprocessing of the complaint data |
| Task 2 | Text chunking, embedding, and indexing |
| Task 3 | RAG pipeline and evaluation            |
| Task 4 | Streamlit/Gradio user interface        |

---

## 🛠 Tech Stack

- Python  
- Sentence Transformers  
- FAISS / ChromaDB (for vector similarity search)  
- LangChain / Hugging Face (for language model integration)  
- Streamlit / Gradio (for building the interactive UI)  
- Pandas, Seaborn, Matplotlib (for data analysis and visualization)  

---

## 📂 Data Source

- Consumer Financial Protection Bureau (CFPB) Complaint Dataset

---

## 👤 Author

Dagmawit Andargachew
[GitHub Profile](https://github.com/Dagi2730)