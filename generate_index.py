import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Constants for data paths
EMBEDDINGS_PATH = "notebooks/vector_store/embeddings.json"
TEXT_CHUNKS_PATH = "notebooks/vector_store/text_chunks.json"
FAISS_INDEX_PATH = "notebooks/vector_store/faiss_index.bin"

@st.cache(allow_output_mutation=True)
def load_models_and_data():
    # Load embedding model
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)

    # Load embeddings and text chunks
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        embeddings_data = json.load(f)
    with open(TEXT_CHUNKS_PATH, "r", encoding="utf-8") as f:
        text_chunks = json.load(f)

    # Load generation pipeline (using bart-large-cnn for summarization-style answers)
    generator = pipeline(
        "text2text-generation", model="facebook/bart-large-cnn", device=-1
    )

    return embed_model, index, embeddings_data, text_chunks, generator


def retrieve_similar_chunks(query, embed_model, index, embeddings_data, text_chunks, top_k=5):
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    D, I = index.search(query_embedding, top_k)
    results = []
    for idx in I[0]:
        if idx < len(text_chunks):
            results.append(text_chunks[idx])
    return results


def generate_answer(query, embed_model, index, embeddings_data, text_chunks, generator):
    retrieved_chunks = retrieve_similar_chunks(query, embed_model, index, embeddings_data, text_chunks, top_k=5)
    if not retrieved_chunks:
        return "Sorry, I don't have enough information to answer that question.", []
    context = "\n".join(retrieved_chunks)

    prompt = (
        f"Use the following customer complaint excerpts to answer the question clearly and concisely.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    outputs = generator(prompt, max_length=150, do_sample=False)
    answer = outputs[0]["generated_text"].strip()
    return answer, retrieved_chunks


def main():
    st.set_page_config(page_title="💬 CrediTrust AI Chat Assistant", page_icon="💬", layout="centered")
    st.title("💬 CrediTrust AI Chat Assistant")
    st.markdown("Ask questions about customer complaints — Powered by Retrieval-Augmented Generation (RAG)")

    embed_model, index, embeddings_data, text_chunks, generator = load_models_and_data()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def submit():
        user_query = st.session_state.user_input.strip()
        if not user_query:
            return

        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.spinner("Generating answer..."):
            answer, _ = generate_answer(user_query, embed_model, index, embeddings_data, text_chunks, generator)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.user_input = ""

    st.text_input("Your question:", key="user_input", on_change=submit, placeholder="Type your question here...")

    # Display chat history with chat bubble style
    for i, chat in enumerate(st.session_state.chat_history):
        if chat["role"] == "user":
            st.markdown(
                f'<div style="text-align: right; background-color:#DCF8C6; padding: 8px; border-radius: 10px; margin: 8px; max-width: 70%; float: right;">'
                f'<b>You:</b> {chat["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="text-align: left; background-color:#F1F0F0; padding: 8px; border-radius: 10px; margin: 8px; max-width: 70%; float: left;">'
                f'<b>CrediTrust AI:</b> {chat["content"]}</div>',
                unsafe_allow_html=True,
            )

    # Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.user_input = ""



if __name__ == "__main__":
    main()
