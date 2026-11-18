# src/hybrid_search.py
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

class HybridRetriever:
    """Manages lexical BM25 indexing over project documentation chunks."""
    def __init__(self, documents: list[Document]):
        """Build a BM25 index over lowercased, whitespace-tokenized page content."""
        self.documents = documents
        # Tokenize corpus text for BM25 lexical matching
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def bm25_search(self, query: str, k: int = 20) -> list[tuple[Document, float]]:
        """Return up to ``k`` positive-scoring documents in descending BM25 order.

        Scores are the raw BM25 values and are not directly comparable to dense
        vector similarity scores; fusion uses each result's rank instead.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.documents[i], float(scores[i])) for i in top_indices if scores[i] > 0]


def reciprocal_rank_fusion(
    dense_results: list[Document], 
    bm25_results: list[tuple[Document, float]], 
    k: int = 60, 
    top_k: int = 20
) -> list[Document]:
    """Merge dense and BM25 rankings using Reciprocal Rank Fusion.

    Each appearance contributes ``1 / (k + rank)`` with one-based ranks. The
    first 150 characters of page content identify a document across result
    lists, so chunks sharing that prefix are treated as the same item. Return
    at most ``top_k`` unique documents in descending combined score order.
    """
    rrf_scores = {}
    doc_map = {}

    # 1. Process Dense Vector Search Ranks
    for rank, doc in enumerate(dense_results):
        doc_id = doc.page_content[:150]  # Create a stable unique fingerprint
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + (rank + 1))

    # 2. Process BM25 Lexical Search Ranks
    for rank, (doc, _) in enumerate(bm25_results):
        doc_id = doc.page_content[:150]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + (rank + 1))

    # 3. Sort merged documents by combined RRF score descending
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    merged_docs = [doc_map[doc_id] for doc_id in sorted_doc_ids[:top_k]]
    return merged_docs