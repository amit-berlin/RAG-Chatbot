import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")

# ------------------------
# Helpers
# ------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return " ".join([page.extract_text() or "" for page in reader.pages])

def build_chunks(text, chunk_size=500, overlap=50):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def build_index(chunks):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(chunks)
    return vectorizer, X

def retrieve(query, vec, X, chunks, k=3):
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    top_ids = sims.argsort()[-k:][::-1]
    return [(chunks[i], sims[i]) for i in top_ids]

# ------------------------
# Streamlit UI
# ------------------------
st.title("📘 Agentic RAG Chatbot – Zero Hallucination")
file = st.file_uploader("Upload a Policy PDF", type=["pdf"])

if file:
    raw_text = extract_text_from_pdf(file)
    chunks = build_chunks(raw_text)
    vec, X = build_index(chunks)
    st.success(f"Indexed {len(chunks)} chunks from PDF ✅")

    user_q = st.text_input("Ask a policy-related question:")

    if st.button("Ask"):
        # Agent 1: Retriever
        with st.expander("🤖 Agent 1 – Retriever"):
            results = retrieve(user_q, vec, X, chunks, k=3)
            for i, (txt, sc) in enumerate(results):
                st.write(f"Chunk {i+1} (score={sc:.2f}): {txt[:200]}...")

        # Agent 2: Evidence Checker
        with st.expander("🕵️ Agent 2 – Evidence Checker"):
            best_chunk, best_score = results[0]
            st.write(f"Most relevant chunk (score={best_score:.2f}):")
            st.write(best_chunk)

        # Agent 3: Final Answer
        with st.expander("✅ Agent 3 – Final Answer"):
            if best_score > 0.2:  # simple threshold guardrail
                st.success(f"Answer (0-hallucination, from PDF):\n\n{best_chunk}")
            else:
                st.warning("Not enough evidence in PDF. Answer withheld to avoid hallucination.")
