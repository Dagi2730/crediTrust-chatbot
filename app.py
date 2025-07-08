import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import os
import json

# Paths
FAISS_INDEX_PATH = "notebooks/vector_store/faiss_index.bin"
EMBEDDINGS_PATH = "notebooks/vector_store/embeddings.json"
TEXTS_PATH = "notebooks/vector_store/text_chunks.json"

# Load FAISS index and supporting files
@st.cache_resource
def load_models_and_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(EMBEDDINGS_PATH, 'r', encoding='utf-8') as f:
        embeddings_data = json.load(f)

    with open(TEXTS_PATH, 'r', encoding='utf-8') as f:
        text_chunks = json.load(f)

    generator = pipeline("text-generation", model="gpt2")

    return model, index, embeddings_data, text_chunks, generator


# Retrieve top-k similar chunks
def retrieve_context(question, k=5):
    question_embedding = embed_model.encode([question])
    D, I = index.search(np.array(question_embedding).astype('float32'), k)
    return [text_chunks[i] for i in I[0] if i < len(text_chunks)]


# Prompt template
def build_prompt(context, question):
    return f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer,
state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:
""".strip()


# Generate answer
def generate_answer(question):
    sources = retrieve_context(question)
    context = "\n".join(sources)
    context = context[:1500]  # Truncate to avoid token overflow

    prompt = build_prompt(context, question)

    try:
        result = generator(prompt, max_new_tokens=150, do_sample=True)
        response = result[0]['generated_text'].split("Answer:")[-1].strip()
    except Exception as e:
        response = f"⚠️ Error generating response: {str(e)}"

    return response, sources


# Load models and data
embed_model, index, embeddings_data, text_chunks, generator = load_models_and_data()

# Streamlit App UI
st.set_page_config(page_title="CrediTrust Chatbot", page_icon="💬", layout="wide")

st.markdown("<h1 style='text-align: center;'>💬 CrediTrust AI Chat Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions about customer complaints — Powered by RAG</p>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
question = st.chat_input("Type your question here...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = generate_answer(question)
            st.markdown(answer)

            with st.expander("🔍 Retrieved Context"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}:** {src}")

        st.session_state.messages.append({"role": "assistant", "content": answer})

# Clear button
if st.button("🔄 Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
