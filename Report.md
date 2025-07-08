# Task 3 Report – RAG Core Logic and Evaluation

## Objective

The goal of this task was to build a Retrieval-Augmented Generation (RAG) system to answer user questions based on customer complaint data. The system should retrieve relevant complaint chunks using vector similarity and generate helpful responses using a language model.

---

## Methodology

### 1. **Retriever Implementation**
- Used the `all-MiniLM-L6-v2` model from SentenceTransformers to embed user questions.
- Performed vector similarity search using FAISS to retrieve top-5 relevant chunks from the vector store.
- The chunk metadata was loaded from `chunked_metadata.csv`.

### 2. **Prompt Engineering**
A prompt template was created to guide the language model:


### 3. **Generation**
- Used Hugging Face's `gpt2` model for text generation.
- Generated answers were trimmed from the full output based on the prompt.

### 4. **Evaluation**
- Evaluated the system using 5 representative financial complaint questions.
- Collected answers, retrieved chunks, and added comments in a CSV file (`rag_evaluation_results.csv`).
- Manually assessed response quality.

---

## Evaluation Table (Sample)

| Question                                      | Generated Answer                             | Quality | Comments                         |
|----------------------------------------------|----------------------------------------------|---------|----------------------------------|
| What are common issues with credit card fraud? | Missing context or vague                     | 2/5     | Needs more relevant examples     |
| Are there complaints about Buy Now, Pay Later? | Mentioned payment delays                     | 4/5     | Good retrieval and clear summary |

---

## Observations

- Retrieval worked well for questions with exact matches in the data.
- GPT-2 sometimes hallucinated if the context was weak.
- Better results might be achieved with a stronger generator model (e.g., Mistral or GPT-4).
- Adding chunk filtering or reranking may improve relevance.

---

## Conclusion

The RAG pipeline was successfully implemented. It retrieves relevant complaint data and generates contextual answers. Further improvement could involve fine-tuning the LLM or experimenting with better retrieval scoring methods.

