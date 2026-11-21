# reranker.py
import threading
import streamlit as st
from sentence_transformers import CrossEncoder
from src.logger import log_status

_model_lock = threading.Lock()

# Load cached CrossEncoder model to avoid reloading weights on every graph invocation
@st.cache_resource
def _cached_load_reranker_model():
    """Load the CPU-backed BGE cross-encoder once per Streamlit resource cache.

    A module lock prevents concurrent first-load attempts from initializing
    the same model weights simultaneously.
    """
    # BGE reranker outputs raw logits or probabilities depending on usage; 
    # higher score = higher relevance.
    with _model_lock:
        return CrossEncoder("BAAI/bge-reranker-large", device="cpu")


def load_reranker_model():
    """Report preparation outside Streamlit's cache, then return the cached model."""
    log_status("Preparing open-source BAAI bge-reranker-large model...")
    return _cached_load_reranker_model()


def rerank_documents(query: str, documents: list, top_k: int = 3):
    """Score query/document pairs and return the highest-ranked chunks and top score.

    Scores are the cross-encoder's raw predictions, not calibrated probabilities;
    the graph uses the maximum score for its configured CRAG threshold. Empty
    input returns no documents and a zero score without loading the model.
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