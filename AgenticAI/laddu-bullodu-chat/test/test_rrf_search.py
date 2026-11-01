# validate_search.py
from src.agent_graph import load_embeddings_and_vector_store
from src.hybrid_search import HybridRetriever, reciprocal_rank_fusion

def test_retrieval_comparison():
    query = "ORD-8802"
    print(f"🔍 Testing Retrieval Methods for Query: '{query}'\n")

    # 1. Load data
    vectorstore = load_embeddings_and_vector_store()
    raw_data = vectorstore.get(include=["documents", "metadatas"])
    documents = [doc for doc in raw_data["documents"]]

    # 2. Dense Search Ranking
    dense_docs = vectorstore.similarity_search(query, k=5)
    print("--- Dense Vector Top Results ---")
    for i, doc in enumerate(dense_docs[:3]):
        print(f"{i+1}. {doc.page_content[:100]}...")

    # 3. BM25 Search Ranking
    retriever = HybridRetriever([type('Obj', (object,), {'page_content': d}) for d in documents])
    bm25_raw = retriever.bm25_search(query, k=5)
    print("\n--- BM25 Lexical Top Results ---")
    for i, (doc, score) in enumerate(bm25_raw[:3]):
        print(f"{i+1} (Score: {score:.2f}). {doc.page_content[:100]}...")

    # 4. Fused RRF Ranking
    dense_doc_objs = [type('Obj', (object,), {'page_content': d}) for d in [doc.page_content for doc in dense_docs]]
    fused = reciprocal_rank_fusion(dense_doc_objs, bm25_raw, top_k=5)
    print("\n--- Fused Hybrid (RRF) Top Results ---")
    for i, doc in enumerate(fused[:3]):
        print(f"{i+1}. {doc.page_content[:100]}...")

if __name__ == "__main__":
    test_retrieval_comparison()