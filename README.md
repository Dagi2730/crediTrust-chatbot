# Task 2: Text Chunking, Embedding, and Vector Store Indexing

## Overview

This task converts cleaned complaint narratives into a format optimized for efficient semantic search. Since long narratives lose detail when embedded as single vectors, we split texts into smaller chunks before embedding and indexing.

---

## Key Components

### 1. Text Chunking
- Implemented using LangChain’s `RecursiveCharacterTextSplitter`.
- Splits long texts into overlapping chunks for better semantic representation.
- Parameters such as `chunk_size` and `chunk_overlap` are experimentally chosen to balance context preservation and embedding speed.

### 2. Embedding Model
- Used the pre-trained model: `sentence-transformers/all-MiniLM-L6-v2`.
- Selected for its good balance of accuracy, speed, and lightweight architecture suitable for large datasets.

### 3. Vector Store Indexing
- Vector embeddings generated for each text chunk.
- Stored in a FAISS index for fast similarity search.
- Metadata (complaint ID, product category, chunk index, chunk text) is saved alongside embeddings to trace results back to the source.

---

## Deliverables

- **Chunking, embedding, and indexing script** that:
  - Reads cleaned narratives.
  - Splits texts into chunks.
  - Generates embeddings.
  - Creates and persists a FAISS vector store.
  - Saves chunk metadata for traceability.

- **Persisted vector store** saved under the `vector_store/` directory:
  - `faiss_index.bin` — FAISS index file.
  - `chunked_metadata.csv` — CSV file with metadata for each chunk.

- **Report section** detailing:
  - The chunking strategy used, including chosen `chunk_size` and `chunk_overlap` and justification.
  - Reason for selecting the `all-MiniLM-L6-v2` embedding model.

---

## How to Use

1. Prepare cleaned complaint texts as input.
2. Run the script to produce the vector store and metadata.
3. Use the vector store for semantic search queries with similarity scoring.

---

## Dependencies

- Python 3.8+
- pandas
- faiss-cpu
- sentence-transformers
- langchain

Install via:

```bash
pip install pandas faiss-cpu sentence-transformers langchain
