# reranker.py
import threading
import streamlit as st
from sentence_transformers import CrossEncoder
from src.logger import log_status

_model_lock = threading.Lock()

# Load cached CrossEncoder model to avoid reloading weights on every graph invocation
@st.cache_resource
def load_reranker_model():
    log_status("Loading open-source BAAI bge-reranker-large model...")
    # BGE reranker outputs raw logits or probabilities depending on usage; 
    # higher score = higher relevance.
    with _model_lock:
        return CrossEncoder("BAAI/bge-reranker-large", device="cpu")

def rerank_documents(query: str, documents: list, top_k: int = 3):
    """Passes retrieved raw documents through the BGE Cross-Encoder re-ranker,
    sorts them by relevance, and returns the top-k high-confidence chunks along with the max score.
    """
    if not documents:
        return [], 0.0

    model = load_reranker_model()
    log_status(f"Re-ranking {len(documents)} retrieved documents against query...")

    # Prepare query-document pairs for the Cross-Encoder
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Predict relevance scores
    scores = model.predict(pairs)

    # Attach scores to documents and sort descending
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Extract top-k documents and the absolute top score (for CRAG thresholding)
    top_scored = scored_docs[:top_k]
    top_docs = [item[0] for item in top_scored]
    max_score = float(top_scored[0][1]) if top_scored else 0.0

    log_status(f"Re-ranking complete. Top score: {max_score:.4f}. Retaining top {len(top_docs)} docs.")
    return top_docs, max_score