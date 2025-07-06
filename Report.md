Task 2 Report: Text Chunking, Embedding, and Vector Store Indexing
Objective
The goal of this task was to convert cleaned complaint narratives into a format suitable for efficient semantic search. Since embedding long texts as a single vector often leads to loss of contextual information, we implemented a chunking strategy to split narratives into smaller, overlapping text chunks. These chunks were then embedded and indexed in a vector store for fast similarity-based retrieval.

Chunking Strategy
We used LangChain’s RecursiveCharacterTextSplitter to split each cleaned narrative into overlapping text chunks. After experimentation, the following parameters were chosen:

Chunk size: 10,000 characters

Chunk overlap: 50 characters

Justification:
A large chunk size (10,000) was selected to preserve more context per chunk, reducing fragmentation of meaning.

A small overlap (50) was used to maintain continuity between chunks without causing excessive duplication or redundancy.

This balance was chosen after testing various chunk sizes and overlaps to optimize for embedding speed and search effectiveness given computational constraints.

Embedding Model Choice
For generating vector embeddings, we selected the sentence-transformers/all-MiniLM-L6-v2 model because:

It is a lightweight, efficient model that produces high-quality sentence embeddings suitable for semantic similarity tasks.

It offers a good trade-off between accuracy and speed, enabling us to embed a large volume of text chunks in a reasonable time.

The model is well-supported and widely used for embedding tasks, making it a reliable choice for our use case.

Embedding and Indexing Process
Each chunked text was converted into a vector embedding using the chosen model.

We used Facebook AI Similarity Search (FAISS) to create an index of the embeddings for efficient similarity search.

Metadata — including complaint ID, product category, chunk index, and the chunk text — was stored alongside embeddings. This metadata allows us to trace any search result back to its original source document and text segment.

Deliverables
A script that performs chunking, embedding, and indexing.

The persisted vector store files saved in the vector_store/ directory:

faiss_index.bin — FAISS index of vector embeddings.

chunked_metadata.csv — Metadata linking chunks to their source complaints.

Documentation and this report section explaining chunking parameters and model choice.

Conclusion
This approach successfully transforms long complaint narratives into a vectorized format optimized for semantic search. By balancing chunk size and overlap, and selecting an efficient embedding model, we achieve good contextual representation and fast retrieval performance, meeting the project requirements for Task 2.