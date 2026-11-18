# src/graph_nodes.py
import os
import time
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import AIMessage

from src.config_loader import load_version_config
from src.intent_router import route_intent, extract_site_parameters
from src.hybrid_search import HybridRetriever, reciprocal_rank_fusion
from src.reranker import rerank_documents
from src.guardrails import sanitize_text
from src.logger import log_status
from src.site_selector import site_selection_graph
from src.supply_chain import supply_chain_graph
from src.llm_provider import GROQ_MODEL, OLLAMA_BASE_URL, get_chat_model, trim_chat_messages
from src.metrics_logger import log_execution

EMBEDDING_MODEL = "embeddinggemma"
PERSIST_DIRECTORY = "./chroma_db"
DOCS_DIRECTORY = "./documents"
serper_search = GoogleSerperAPIWrapper()

CONFIG = load_version_config()
PROMPTS = CONFIG["prompts"]
THRESHOLDS = CONFIG["thresholds"]

def estimate_token_cost(model_name: str, text_content: str) -> float:
    """Estimate request cost from text length using a four-characters-per-token heuristic.

    The estimate is telemetry only, not a provider billing calculation; model
    names containing ``70b`` use the higher per-million-token rate.
    """
    tokens = len(text_content) / 4  # ~4 chars per token heuristic
    if "70b" in model_name.lower():
        return (tokens / 1_000_000) * 0.70  # ~$0.70 per million tokens
    return (tokens / 1_000_000) * 0.20   # ~$0.20 per million tokens

@st.cache_resource
def load_embeddings_and_vector_store(use_local_embeddings: bool = False):
    """Load the selected embedding backend and its persistent Chroma collection.

    Reads ``.txt`` and ``.md`` files from ``documents/``, sanitizes PII before
    indexing, splits content into overlapping 500-character chunks, and stores
    source metadata. Local and Ollama embeddings use separate collection names.
    """
    if use_local_embeddings:
        from src.local_embeddings import LocalEmbeddingGemmaEmbeddings
        log_status("Loading local Hugging Face EmbeddingGemma model...")
        emb_func = LocalEmbeddingGemmaEmbeddings()
        collection_name = "embeddinggemma_local_v1_guarded_collection"
    else:
        from langchain_ollama import OllamaEmbeddings
        log_status(f"Initializing Ollama Embeddings model: {EMBEDDING_MODEL}")
        emb_func = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        collection_name = "embeddinggemma_v3_guarded_collection"
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=emb_func,
        persist_directory=PERSIST_DIRECTORY
    )
    
    if os.path.exists(DOCS_DIRECTORY):
        all_docs = []
        for filename in os.listdir(DOCS_DIRECTORY):
            if filename.endswith((".txt", ".md")):
                file_path = os.path.join(DOCS_DIRECTORY, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        sanitized_content = sanitize_text(f.read())
                        doc_id = f"{filename}_{os.path.getmtime(file_path)}"
                        all_docs.append(Document(page_content=sanitized_content, metadata={"source": filename, "id": doc_id}))
                except Exception as e:
                    log_status(f"Error reading {filename}: {e}")
        
        if all_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(all_docs)
            vector_store.add_texts(
                texts=[d.page_content for d in split_docs],
                metadatas=[d.metadata for d in split_docs],
                ids=[f"{d.metadata['source']}_{i}" for i, d in enumerate(split_docs)]
            )
            log_status("Successfully upserted guarded chunks into vector store.")
    return vector_store

_hybrid_retriever = None

def get_or_init_hybrid_retriever(vectorstore, documents: list = None) -> HybridRetriever:
    """Build the process-wide BM25 index from supplied or persisted Chroma documents.

    The index is initialized lazily and reused for subsequent requests. Pass
    ``documents`` to avoid reading the collection; otherwise documents and
    metadata are fetched from Chroma. The cached index is not refreshed here.
    """
    global _hybrid_retriever
    if _hybrid_retriever is None:
        if documents is None:
            try:
                raw_data = vectorstore.get(include=["documents", "metadatas"])
                documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(raw_data["documents"], raw_data["metadatas"])]
            except Exception:
                documents = []
        if documents:
            _hybrid_retriever = HybridRetriever(documents)
    return _hybrid_retriever

def router_node(state: dict):
    """Find the latest user message, classify its intent, and start latency timing.

    Mutates and returns the graph state with ``intent`` and ``start_time`` set.
    """
    state["start_time"] = time.time()  # Stamped for latency calculation
    latest_query = ""
    for msg in reversed(state["messages"]):
        msg_type = getattr(msg, "type", "")
        if msg_type in ("human", "user") or (isinstance(msg, dict) and msg.get("role") == "user"):
            latest_query = msg.content.strip() if hasattr(msg, "content") else msg.get("content", "").strip()
            break
    state["intent"] = route_intent(latest_query)
    log_status(f"Router classified intent as: {state['intent']}")
    return state

def retrieve_node(state: dict):
    """Retrieve candidate chunks using dense search and BM25, then fuse their rankings.

    Dense and lexical result counts use ``top_k_retrieval`` from configuration;
    fused candidates are written to ``raw_docs`` for the reranker node. Groq
    selection opts into the local embedding model in this workflow.
    """
    latest_query = state["messages"][-1].content
    selected_model = state.get("selected_model", GROQ_MODEL)
    
    log_status("Executing Hybrid Retrieval (Dense Vector + BM25 Lexical Search)...")
    vectorstore = load_embeddings_and_vector_store(use_local_embeddings=selected_model == GROQ_MODEL)
    
    dense_docs = vectorstore.similarity_search(latest_query, k=THRESHOLDS.get("top_k_retrieval", 20))
    hybrid_retriever = get_or_init_hybrid_retriever(vectorstore)
    bm25_docs = hybrid_retriever.bm25_search(latest_query, k=THRESHOLDS.get("top_k_retrieval", 20)) if hybrid_retriever else []
    
    fused_docs = reciprocal_rank_fusion(dense_docs, bm25_docs, top_k=THRESHOLDS.get("top_k_retrieval", 20))
    state["raw_docs"] = fused_docs
    return state

def rerank_node(state: dict):
    """Score fused candidates with the BGE cross-encoder and store top chunks and score.

    The configured ``top_k_rerank`` determines the retained documents, while the
    maximum model score is used by the conditional CRAG edge.
    """
    latest_query = state["messages"][-1].content
    raw_docs = state.get("raw_docs", [])
    top_docs, top_score = rerank_documents(latest_query, raw_docs, top_k=THRESHOLDS.get("top_k_rerank", 3))
    state["top_docs"] = top_docs
    state["top_relevance_score"] = top_score
    return state

def domain_executor_node(state: dict):
    """Dispatch a classified non-RAG request to its domain workflow or chat model.

    Supply-chain and site-selection intents invoke their dedicated LangGraph
    subgraphs; web intent adds Serper results to the prompt; other intents use
    direct model conversation. The resulting tagged ``AIMessage`` is appended
    to the mutable graph state.
    """
    intent = state.get("intent", "LLM")
    latest_query = state["messages"][-1].content
    selected_model = state.get("selected_model", GROQ_MODEL)
    
    if intent == "SUPPLY_CHAIN":
        result = supply_chain_graph.invoke({"query": latest_query, "selected_model": selected_model})
        memo = result.get("final_mitigation_memo", "Supply chain analysis completed.")
        response_content = f"### 📦 Supply Chain Report\n\n{memo}\n\n---\n*⚡ Orchestrated via Neo4j & LangGraph*"
        badge = "[SUPPLY_CHAIN]"
    elif intent == "SITE_SELECTION":
        country, locations = extract_site_parameters(latest_query, selected_model)
        result = site_selection_graph.invoke({"country": country, "locations": locations, "raw_data": {}, "evaluations": {}, "comparison_report": "", "report_filepath": "", "selected_model": selected_model})
        response_content = f"### 🏢 Site Selection Report\n\n{result['comparison_report']}\n\n---\n*📁 Saved to: `{result['report_filepath']}`*"
        badge = "[SITE SELECTION]"
    elif intent == "WEB":
        try:
            web_results = serper_search.run(latest_query)
        except Exception as e:
            web_results = f"Web search failed: {e}"
        sys_prompt = f"{PROMPTS['web_system_prompt']}\n\n--- LIVE WEB RESULTS ---\n{web_results}\n----------------"
        messages = trim_chat_messages([{"role": "system", "content": sys_prompt}] + state["messages"], max_chars=12000)
        res = get_chat_model(selected_model).invoke(messages)
        response_content, badge = res.content, "[WEB]"
    else:
        messages = trim_chat_messages(state["messages"], max_chars=12000)
        res = get_chat_model(selected_model).invoke(messages)
        response_content, badge = res.content, "[LLM]"

    response = AIMessage(content=f"{badge} {response_content}")
    state["messages"].append(response)
    return state

def rag_generate_node(state: dict):
    """Generate an answer from re-ranked local chunks and the configured RAG prompt.

    Context is capped at 9,000 characters, then combined with bounded chat
    history before the selected model is invoked.
    """
    top_docs = state.get("top_docs", [])
    selected_model = state.get("selected_model", GROQ_MODEL)
    context_text = "\n\n".join([doc.page_content for doc in top_docs])
    
    sys_prompt = f"{PROMPTS['rag_system_prompt']}\n\n--- RE-RANKED LOCAL CONTEXT (Top 3) ---\n{context_text[:9000]}\n----------------"
    messages = trim_chat_messages([{"role": "system", "content": sys_prompt}] + state["messages"], max_chars=12000)
    response = get_chat_model(selected_model).invoke(messages)
    response.content = f"[RAG] {response.content}"
    state["messages"].append(response)
    return state

def crag_fallback_node(state: dict):
    """Answer with live Serper results when local-document relevance is too low.

    Search failures are included in the model context as text; results are
    capped before invocation and the response is tagged for downstream display.
    """
    latest_query = state["messages"][-1].content
    selected_model = state.get("selected_model", GROQ_MODEL)
    try:
        web_results = serper_search.run(latest_query)
    except Exception as e:
        web_results = f"Web search failed: {e}"
        
    sys_prompt = f"{PROMPTS['crag_fallback_prompt']}\n\n--- LIVE WEB RESULTS ---\n{str(web_results)[:9000]}\n----------------"
    messages = trim_chat_messages([{"role": "system", "content": sys_prompt}] + state["messages"], max_chars=12000)
    response = get_chat_model(selected_model).invoke(messages)
    response.content = f"[WEB - CRAG FALLBACK] {response.content}"
    state["messages"].append(response)
    return state

def output_guardrail_node(state: dict):
    """Sanitize the final response and record latency, estimated cost, and intent metrics.

    Preserves a recognized response badge while applying PII redaction to its
    content. Metrics persistence failures are reported through the status logger
    and do not replace the completed assistant response.
    """
    log_status("Checking output response against PII guardrails & calculating telemetry...")
    last_msg = state["messages"][-1]
    
    badge = "[LLM]"
    content_str = ""
    if hasattr(last_msg, "content"):
        content_str = last_msg.content
        for prefix in ("[RAG]", "[WEB]", "[WEB - CRAG FALLBACK]", "[LLM]", "[SITE SELECTION]", "[SUPPLY_CHAIN]"):
            if content_str.startswith(prefix):
                badge = prefix
                content_str = content_str[len(prefix):].strip()
                break
        last_msg.content = f"{badge} {sanitize_text(content_str)}".strip()

    # Calculate operational telemetry
    start_time = state.get("start_time", time.time())
    latency = time.time() - start_time
    intent = state.get("intent", "LLM")
    selected_model = state.get("selected_model", GROQ_MODEL)
    token_cost = estimate_token_cost(selected_model, content_str)
    
    thread_id = state.get("thread_id", "default_thread")
    
    # Log telemetry to metrics.sqlite
    try:
        log_execution(
            thread_id=thread_id,
            intent=intent,
            latency_seconds=latency,
            token_cost=token_cost,
            citation_mode=badge,
            status="SUCCESS"
        )
    except Exception as e:
        log_status(f"Metrics logging warning: {e}")

    return {"messages": state["messages"]}