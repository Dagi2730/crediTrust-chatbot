# Task 3 – Retrieval-Augmented Generation (RAG) Pipeline

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system designed to answer questions about customer complaints submitted to CrediTrust. The system combines vector-based retrieval with a language model to generate informed responses grounded in real complaint data.

## Objective

- Retrieve the most relevant complaint excerpts for a user query using vector similarity.
- Generate natural language answers based strictly on retrieved context.
- Evaluate the accuracy and usefulness of generated responses using qualitative analysis.

## Components

### 1. Retriever
- Encodes the input question using the `all-MiniLM-L6-v2` model.
- Searches a FAISS index to retrieve the top-k most relevant chunks.
- Returns these chunks as context for generation.

### 2. Prompt Engineering
A structured prompt guides the LLM to act as a helpful financial analyst and answer based only on provided context. If the context is insufficient, the model is instructed to say so.

### 3. Generator
- Uses Hugging Face’s GPT-2 model to generate a response based on the prompt.
- Limits the output to a maximum of 150 new tokens for clarity and focus.

### 4. Evaluation
- A set of 5 representative questions is used to evaluate the system’s performance.
- Results are stored in a CSV file with:
  - Question
  - Generated Answer
  - Retrieved Sources
  - Quality Score (1–5)
  - Comments

## Running the Pipeline

Ensure required files (`faiss_index.bin`, `chunked_metadata.csv`) exist in the correct paths, then run:

```bash
python src/rag_pipeline.py
