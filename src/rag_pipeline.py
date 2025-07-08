import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# -------- Paths --------
VECTOR_STORE_DIR = "./vector_store"          # Adjust this if your vector files are elsewhere
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
METADATA_CSV_PATH = "./data/chunked_metadata.csv"
EVAL_OUTPUT_CSV = "rag_evaluation_results.csv"

# -------- Load models and data --------
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("Loading FAISS index...")
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"FAISS index file not found: {FAISS_INDEX_PATH}")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading metadata CSV...")
if not os.path.exists(METADATA_CSV_PATH):
    raise FileNotFoundError(f"Metadata CSV file not found: {METADATA_CSV_PATH}")
metadata_df = pd.read_csv(METADATA_CSV_PATH)

print("Loading text generation model...")
generator = pipeline("text-generation", model="gpt2", max_length=256, do_sample=True)

# -------- Retriever function --------
def retrieve_relevant_chunks(query, top_k=5):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    
    chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        row = metadata_df.iloc[idx]
        chunks.append({
            "chunk_text": row['chunk_text'],
            "score": dist
        })
    return chunks

# -------- Prompt builder --------
def build_prompt(context_chunks, question):
    context_text = "\n\n".join(context_chunks)
    prompt = (
        "You are a financial analyst assistant for CrediTrust.\n"
        "Use the following retrieved complaint excerpts to answer the question.\n"
        "If the answer is not contained in the context, say \"I don't have enough information.\"\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    return prompt

# -------- Answer generator --------
def generate_answer(prompt):
    response = generator(prompt, max_new_tokens=150, do_sample=True)[0]['generated_text']
    # Extract the generated answer only by removing the prompt from the output
    answer = response[len(prompt):].strip()
    return answer

# -------- Main evaluation --------
if __name__ == "__main__":
    sample_questions = [
        "What are common issues with credit card fraud?",
        "How do customers describe their problems with savings accounts?",
        "Are there complaints about Buy Now, Pay Later services?",
        "What complaints exist regarding personal loans?",
        "What issues do customers face with money transfers?"
    ]

    eval_results = []

    print("Running evaluation on sample questions...\n")
    for question in sample_questions:
        print(f"Question: {question}")
        retrieved = retrieve_relevant_chunks(question, top_k=5)
        if not retrieved:
            print("No relevant chunks found, skipping...\n")
            continue
        context_chunks = [chunk['chunk_text'] for chunk in retrieved]
        prompt = build_prompt(context_chunks, question)
        answer = generate_answer(prompt)
        print(f"Answer: {answer}\n")

        top_sources = context_chunks[:2]

        eval_results.append({
            "Question": question,
            "Generated Answer": answer,
            "Retrieved Sources": top_sources,
            "Quality Score": None,
            "Comments": ""
        })

    # Save evaluation results to CSV
    eval_df = pd.DataFrame(eval_results)
    eval_df.to_csv(EVAL_OUTPUT_CSV, index=False)
    print(f"Evaluation results saved to {EVAL_OUTPUT_CSV}")
