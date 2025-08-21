import re
import numpy as np
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(page_title="Agentic PDF RAG – 5 Agents • Zero Hallucination", layout="wide")

# ---------------------------
# Small utilities
# ---------------------------
def safe_text(x: str) -> str:
    if not x: return ""
    return x.replace("\x00", " ").strip()

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join([safe_text(p.extract_text() or "") for p in reader.pages])

def split_sentences(text: str):
    # tiny sentence splitter
    sents = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sents if s and s.strip()]

def build_chunks(raw_text: str, chunk_size=800, overlap=120):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = [c.strip().replace("\n", " ") for c in splitter.split_text(raw_text)]
    # keep only non-empty chunks
    return [c for c in chunks if c]

def tfidf_fit(corpus):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, strip_accents="unicode")
    X = vec.fit_transform(corpus)
    return vec, X

def cosine(a, b):
    return float(cosine_similarity(a, b)[0][0])

def jaccard_tokens(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def tokens(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

# ---------------------------
# Agent 1 – Retriever
# ---------------------------
def agent1_retrieve(query, vec, X, corpus, top_k=5):
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    order = np.argsort(-sims)[:top_k]
    top = [{"chunk_id": int(i), "score": float(sims[i]), "text": corpus[i]} for i in order]
    return top, qv

# ---------------------------
# Agent 2 – Evidence Selector (sentence-level)
# ---------------------------
def agent2_evidence(qv, vec, retrieved, per_chunk=2):
    evidences = []
    for r in retrieved:
        sents = split_sentences(r["text"])
        if not sents: 
            continue
        S = vec.transform(sents)
        sims = cosine_similarity(qv, S).flatten()
        idxs = np.argsort(-sims)[:per_chunk]
        for j in idxs:
            evidences.append({
                "chunk_id": r["chunk_id"],
                "sentence": sents[j],
                "similarity": float(sims[j])
            })
    evidences.sort(key=lambda e: -e["similarity"])
    return evidences

# ---------------------------
# Agent 3 – Verifier (multi-metric guardrail)
# Checks: max/avg cosine on evidence, token coverage, and decides to answer/abstain
# ---------------------------
def agent3_verify(query, evidences, vec, thresholds):
    if not evidences:
        return {"safe": False, "confidence": 0.0, "reason": "no-evidence"}

    q_tokens = tokens(query)
    cosines = np.array([e["similarity"] for e in evidences])
    coverage_scores = [jaccard_tokens(q_tokens, tokens(e["sentence"])) for e in evidences]
    max_cos, avg_cos = float(cosines.max()), float(cosines.mean())
    max_cov, avg_cov = float(np.max(coverage_scores)), float(np.mean(coverage_scores))

    # decision
    safe = (max_cos >= thresholds["min_best_cos"]) and \
           (avg_cos >= thresholds["min_avg_cos"]) and \
           (max_cov >= thresholds["min_best_cov"]) and \
           (avg_cov >= thresholds["min_avg_cov"])

    # a smooth confidence in [0,1]
    confidence = float(np.clip(0.25*max_cos + 0.25*avg_cos + 0.25*max_cov + 0.25*avg_cov, 0.0, 1.0))

    return {
        "safe": bool(safe),
        "confidence": confidence,
        "max_cos": max_cos, "avg_cos": avg_cos,
        "max_cov": max_cov, "avg_cov": avg_cov
    }

# ---------------------------
# Agent 4 – Exact Token-to-Token Finder
# Scans top-evidence sentences and returns lines with strict token overlaps
# ---------------------------
def agent4_exact_matches(query, evidences, min_token_hits=1):
    q_toks = tokens(query)
    hits = []
    for e in evidences:
        line = e["sentence"]
        ltoks = tokens(line)
        common = [t for t in q_toks if t in ltoks]
        if len(common) >= min_token_hits:
            hits.append({"chunk_id": e["chunk_id"], "line": line, "token_hits": common, "similarity": e["similarity"]})
    hits.sort(key=lambda h: (-len(h["token_hits"]), -h["similarity"]))
    return hits

# ---------------------------
# Agent 5 – Answer Composer (extractive, cited)
# Builds the final answer only from evidence sentences (no generation)
# ---------------------------
def agent5_compose(evidences, max_sentences=3):
    picked, seen = [], set()
    for e in evidences:
        s = e["sentence"].strip()
        if s not in seen:
            seen.add(s)
            picked.append(e)
        if len(picked) >= max_sentences:
            break
    if not picked:
        return "", []
    parts = []
    for e in picked:
        parts.append(f"{e['sentence']} [c{e['chunk_id']}]")
    return " ".join(parts), picked

# ---------------------------
# UI – Sidebar controls
# ---------------------------
st.sidebar.title("Zero-Hallucination Controls")
top_k = st.sidebar.slider("Retriever: top_k", 1, 10, 5, 1)
per_chunk = st.sidebar.slider("Evidence per chunk", 1, 5, 2, 1)
chunk_size = st.sidebar.slider("Chunk size", 300, 1200, 800, 50)
overlap = st.sidebar.slider("Chunk overlap", 0, 300, 120, 10)

st.sidebar.markdown("---")
st.sidebar.caption("Guardrails (Agent-3)")
min_best_cos = st.sidebar.slider("min best cosine", 0.00, 1.00, 0.15, 0.01)
min_avg_cos  = st.sidebar.slider("min avg cosine",  0.00, 1.00, 0.08, 0.01)
min_best_cov = st.sidebar.slider("min best token coverage", 0.00, 1.00, 0.15, 0.01)
min_avg_cov  = st.sidebar.slider("min avg token coverage",  0.00, 1.00, 0.08, 0.01)

thresholds = {
    "min_best_cos": float(min_best_cos),
    "min_avg_cos": float(min_avg_cos),
    "min_best_cov": float(min_best_cov),
    "min_avg_cov": float(min_avg_cov),
}

st.sidebar.markdown("---")
strict_hits = st.sidebar.slider("Agent-4: min exact token hits", 1, 4, 1, 1)

# ---------------------------
# Main header + uploader
# ---------------------------
st.title("📘 Agentic PDF RAG – 5 Human-Line Agents (Zero Hallucination)")
st.write("Upload a **policy PDF** and chat. The answer passes through "
         "**Agent-1 Retriever → Agent-2 Evidence → Agent-3 Verifier → Agent-4 Token-Match → Agent-5 Composer**. "
         "If evidence is weak, the bot abstains.")

file = st.file_uploader("Upload a PDF", type=["pdf"])

# Session state
if "corpus" not in st.session_state:
    st.session_state.corpus = None
    st.session_state.vectorizer = None
    st.session_state.X = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Build index
if file is not None:
    raw = extract_text_from_pdf(file)
    if not raw:
        st.error("No extractable text found in this PDF.")
    else:
        corpus = build_chunks(raw, chunk_size=chunk_size, overlap=overlap)
        if not corpus:
            st.error("Chunking yielded no text. Try different chunk settings.")
        else:
            vec, X = tfidf_fit(corpus)
            st.session_state.corpus = corpus
            st.session_state.vectorizer = vec
            st.session_state.X = X
            st.success(f"Indexed {len(corpus)} chunks ✅")

# Chat UI
if st.session_state.corpus is not None:
    # history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Ask your question about the uploaded PDF")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # ---------- Agent 1 ----------
        retrieved, qv = agent1_retrieve(user_q, st.session_state.vectorizer, st.session_state.X,
                                        st.session_state.corpus, top_k=top_k)
        with st.chat_message("assistant"):
            st.markdown("**🔎 Agent-1 Retriever (top context)**")
            df1 = pd.DataFrame([{
                "chunk_id": r["chunk_id"],
                "similarity": round(r["score"], 3),
                "snippet": (r["text"][:240] + "…") if len(r["text"]) > 240 else r["text"]
            } for r in retrieved])
            st.dataframe(df1, use_container_width=True)

        # ---------- Agent 2 ----------
        evidences = agent2_evidence(qv, st.session_state.vectorizer, retrieved, per_chunk=per_chunk)
        with st.chat_message("assistant"):
            st.markdown("**🧪 Agent-2 Evidence (best sentences)**")
            df2 = pd.DataFrame([{
                "chunk_id": e["chunk_id"],
                "cosine": round(e["similarity"], 3),
                "sentence": e["sentence"]
            } for e in evidences])
            st.dataframe(df2, use_container_width=True)

        # ---------- Agent 3 ----------
        verdict = agent3_verify(user_q, evidences, st.session_state.vectorizer, thresholds)
        with st.chat_message("assistant"):
            st.markdown("**🛡️ Agent-3 Verifier (guardrail decision)**")
            st.write({
                "safe": verdict["safe"],
                "confidence": round(verdict["confidence"], 3),
                "max_cos": round(verdict["max_cos"], 3),
                "avg_cos": round(verdict["avg_cos"], 3),
                "max_token_cov": round(verdict["max_cov"], 3),
                "avg_token_cov": round(verdict["avg_cov"], 3),
            })
            if not verdict["safe"]:
                st.warning("Evidence too weak. I will abstain unless Agent-4 finds strong exact token matches.")

        # ---------- Agent 4 ----------
        strict_hits_list = agent4_exact_matches(user_q, evidences, min_token_hits=strict_hits)
        with st.chat_message("assistant"):
            st.markdown("**🧷 Agent-4 Token-to-Token Matches (exact lines)**")
            if strict_hits_list:
                df4 = pd.DataFrame([{
                    "chunk_id": h["chunk_id"],
                    "token_hits": ", ".join(sorted(set(h["token_hits"]))),
                    "cosine": round(h["similarity"], 3),
                    "line": h["line"]
                } for h in strict_hits_list])
                st.dataframe(df4, use_container_width=True)
            else:
                st.info("No strict token matches. Proceeding with evidence-only answer if guardrail allows.")

        # ---------- Agent 5 ----------
        # Allow Agent-4 to upgrade the safety decision if we have strong exact hits
        upgraded_safe = verdict["safe"] or (len(strict_hits_list) > 0 and verdict["confidence"] >= 0.1)
        final_answer, cited = agent5_compose(evidences, max_sentences=3)

        with st.chat_message("assistant"):
            st.markdown("**✅ Agent-5 Final Answer (extractive only)**")
            if upgraded_safe and final_answer:
                st.markdown(final_answer)
                cited_ids = sorted({c['chunk_id'] for c in cited})
                st.caption(f"Cited chunks: {', '.join(f'c{id}' for id in cited_ids)} • Confidence: {verdict['confidence']:.2f}")
            else:
                st.warning("I can’t answer from this document with enough evidence. Please rephrase or upload a richer PDF.")
                st.caption(f"Confidence: {verdict['confidence']:.2f}")

        # history add final
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer if (upgraded_safe and final_answer) else "Abstained (insufficient evidence)."
        })

else:
    st.info("Upload a PDF to start chatting.")
