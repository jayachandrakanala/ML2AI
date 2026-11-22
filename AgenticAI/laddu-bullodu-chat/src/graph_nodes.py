# src/graph_nodes.py
import logging
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
from src.llm_provider import (
    GROQ_GPT_OSS_MODEL,
    GROQ_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    get_chat_model,
    trim_chat_messages,
)
from src.metrics_logger import log_execution
from src.weather import extract_weather_location, get_weather_report
from src.llmops import (
    current_trace_id,
    rag_answer_context_overlap,
    set_run_attribute,
)

EMBEDDING_MODEL = "embeddinggemma"
PERSIST_DIRECTORY = "./chroma_db"
DOCS_DIRECTORY = "./documents"
serper_search = GoogleSerperAPIWrapper()

CONFIG = load_version_config()
PROMPTS = CONFIG["prompts"]
THRESHOLDS = CONFIG["thresholds"]

# Function: estimate_token_cost
# Purpose: Estimate request cost from text length using a four-characters-per-token heuristic. The estimate is
# telemetry only, not a provider billing calculation; model names containing ``70b`` use the higher
# per-million-token rate.
def estimate_token_cost(model_name: str, text_content: str) -> float:
    """Estimate request cost from text length using a four-characters-per-token heuristic.

    The estimate is telemetry only, not a provider billing calculation; model
    names containing ``70b`` use the higher per-million-token rate.
    """
    tokens = len(text_content) / 4  # ~4 chars per token heuristic
    if "70b" in model_name.lower():
        return (tokens / 1_000_000) * 0.70  # ~$0.70 per million tokens
    return (tokens / 1_000_000) * 0.20   # ~$0.20 per million tokens

# Function: _cached_load_embeddings_and_vector_store
# Purpose: Load the selected embedding backend and its persistent Chroma collection. Reads ``.txt`` and ``.md``
# files from ``documents/``, sanitizes PII before indexing, splits content into overlapping
# 500-character chunks, and stores source metadata. Local and Ollama embeddings use separate
# collection names.
@st.cache_resource
def _cached_load_embeddings_and_vector_store(use_local_embeddings: bool = False):
    """Load the selected embedding backend and its persistent Chroma collection.

    Reads ``.txt`` and ``.md`` files from ``documents/``, sanitizes PII before
    indexing, splits content into overlapping 500-character chunks, and stores
    source metadata. Local and Ollama embeddings use separate collection names.
    """
    # Embedding backend selection:
    # - Cloud chat models use the local Hugging Face/Gemma-compatible path.
    # - Ollama chat uses Ollama's embedding service and a separate collection.
    # Both paths write sanitized document chunks into persistent Chroma storage.
    if use_local_embeddings:
        from src.local_embeddings import LocalEmbeddingGemmaEmbeddings
        emb_func = LocalEmbeddingGemmaEmbeddings()
        collection_name = "embeddinggemma_local_v1_guarded_collection"
    else:
        from langchain_ollama import OllamaEmbeddings
        emb_func = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        collection_name = "embeddinggemma_v3_guarded_collection"
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=emb_func,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # Document ingestion is incremental: existing collection entries are reused,
    # while changed project documents are sanitized and split into 500-character
    # chunks before being stored with source and modification metadata.
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
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Error reading document %s", filename
                    )
        
        if all_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(all_docs)
            vector_store.add_texts(
                texts=[d.page_content for d in split_docs],
                metadatas=[d.metadata for d in split_docs],
                ids=[f"{d.metadata['source']}_{i}" for i, d in enumerate(split_docs)]
            )
    return vector_store


# Function: load_embeddings_and_vector_store
# Purpose: Report preparation outside Streamlit's cache, then return the cached store.
def load_embeddings_and_vector_store(use_local_embeddings: bool = False):
    """Report preparation outside Streamlit's cache, then return the cached store."""
    backend = (
        "local Hugging Face EmbeddingGemma"
        if use_local_embeddings
        else f"Ollama {EMBEDDING_MODEL}"
    )
    vector_store = _cached_load_embeddings_and_vector_store(use_local_embeddings)
    log_status("Vector store is ready.")
    return vector_store


_hybrid_retriever = None

# Function: get_or_init_hybrid_retriever
# Purpose: Build the process-wide BM25 index from supplied or persisted Chroma documents. The index is
# initialized lazily and reused for subsequent requests. Pass ``documents`` to avoid reading the
# collection; otherwise documents and metadata are fetched from Chroma. The cached index is not
# refreshed here.
def get_or_init_hybrid_retriever(vectorstore, documents: list = None) -> HybridRetriever:
    """Build the process-wide BM25 index from supplied or persisted Chroma documents.

    The index is initialized lazily and reused for subsequent requests. Pass
    ``documents`` to avoid reading the collection; otherwise documents and
    metadata are fetched from Chroma. The cached index is not refreshed here.
    """
    global _hybrid_retriever
    # BM25 is process-wide and lazy. It is initialized once from Chroma
    # documents, so subsequent retrieval calls avoid rebuilding the lexical index.
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

# Function: router_node
# Purpose: Find the latest user message, classify its intent, and start latency timing. Mutates and returns the
# graph state with ``intent`` and ``start_time`` set.
def router_node(state: dict):
    """Find the latest user message, classify its intent, and start latency timing.

    Mutates and returns the graph state with ``intent`` and ``start_time`` set.
    """
    # Search the bounded message history from newest to oldest for the latest
    # human request. The classified intent and start time are written into state
    # so downstream nodes and the guardrail can use them consistently.
    state["start_time"] = time.time()  # Stamped for latency calculation
    latest_query = ""
    for msg in reversed(state["messages"]):
        msg_type = getattr(msg, "type", "")
        if msg_type in ("human", "user") or (isinstance(msg, dict) and msg.get("role") == "user"):
            latest_query = msg.content.strip() if hasattr(msg, "content") else msg.get("content", "").strip()
            break
    state["intent"] = route_intent(latest_query)
    set_run_attribute("intent", state["intent"])
    log_status(f"Router classified intent as: {state['intent']}")
    return state

# Function: retrieve_node
# Purpose: Retrieve candidate chunks using dense search and BM25, then fuse their rankings. Dense and lexical
# result counts use ``top_k_retrieval`` from configuration; fused candidates are written to
# ``raw_docs`` for the reranker node. Cloud chat models use the local embedding model so retrieval
# does not depend on a reachable Ollama server; Ollama chat continues to use Ollama embeddings.
def retrieve_node(state: dict):
    """Retrieve candidate chunks using dense search and BM25, then fuse their rankings.

    Dense and lexical result counts use ``top_k_retrieval`` from configuration;
    fused candidates are written to ``raw_docs`` for the reranker node. Cloud
    chat models use the local embedding model so retrieval does not depend on a
    reachable Ollama server; Ollama chat continues to use Ollama embeddings.
    """
    latest_query = state["messages"][-1].content
    selected_model = state.get("selected_model", GROQ_MODEL)
    
    # Retrieve in two ways:
    # 1. Dense similarity search finds semantically related chunks.
    # 2. BM25 finds exact lexical matches.
    # Reciprocal Rank Fusion combines both ranked lists before re-ranking.
    vectorstore = load_embeddings_and_vector_store(
        use_local_embeddings=selected_model != OLLAMA_MODEL
    )
    
    dense_docs = vectorstore.similarity_search(latest_query, k=THRESHOLDS.get("top_k_retrieval", 20))
    hybrid_retriever = get_or_init_hybrid_retriever(vectorstore)
    bm25_docs = hybrid_retriever.bm25_search(latest_query, k=THRESHOLDS.get("top_k_retrieval", 20)) if hybrid_retriever else []
    
    fused_docs = reciprocal_rank_fusion(dense_docs, bm25_docs, top_k=THRESHOLDS.get("top_k_retrieval", 20))
    state["raw_docs"] = fused_docs
    return state

# Function: rerank_node
# Purpose: Score fused candidates with the BGE cross-encoder and store top chunks and score. The configured
# ``top_k_rerank`` determines the retained documents, while the maximum model score is used by the
# conditional CRAG edge.
def rerank_node(state: dict):
    """Score fused candidates with the BGE cross-encoder and store top chunks and score.

    The configured ``top_k_rerank`` determines the retained documents, while the
    maximum model score is used by the conditional CRAG edge.
    """
    # The reranker scores all fused candidates, retains the configured top-k
    # documents, and exposes the highest score to the CRAG routing edge.
    latest_query = state["messages"][-1].content
    raw_docs = state.get("raw_docs", [])
    top_docs, top_score = rerank_documents(latest_query, raw_docs, top_k=THRESHOLDS.get("top_k_rerank", 3))
    state["top_docs"] = top_docs
    state["top_relevance_score"] = top_score
    set_run_attribute("rag.reranker_score", top_score)
    set_run_attribute("rag.retrieved_context_count", len(top_docs))
    return state

# Function: domain_executor_node
# Purpose: Dispatch a classified non-RAG request to its domain workflow or chat model. Supply-chain and
# site-selection intents invoke their dedicated LangGraph subgraphs; web intent adds Serper results to
# the prompt; other intents use direct model conversation. The resulting tagged ``AIMessage`` is
# appended to the mutable graph state.
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
    
    # Domain dispatch branches to a specialist workflow or creates a direct
    # model response. The chosen badge identifies which execution path produced
    # the answer and is preserved through output guardrail processing.
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
        if "weather" in latest_query.lower() or "temperature" in latest_query.lower():
            location = extract_weather_location(latest_query)
            response_content = get_weather_report(location)
            state["messages"].append(AIMessage(content=f"[WEB] {response_content}"))
            return state
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

# Function: rag_generate_node
# Purpose: Generate an answer from re-ranked local chunks and the configured RAG prompt. Context is capped at
# 9,000 characters, then combined with bounded chat history before the selected model is invoked.
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

# Function: crag_fallback_node
# Purpose: Answer with live Serper results when local-document relevance is too low. Search failures are
# included in the model context as text; results are capped before invocation and the response is
# tagged for downstream display.
def crag_fallback_node(state: dict):
    """Answer with live Serper results when local-document relevance is too low.

    Search failures are included in the model context as text; results are
    capped before invocation and the response is tagged for downstream display.
    """
    latest_query = state["messages"][-1].content
    selected_model = state.get("selected_model", GROQ_MODEL)
    # A live-search failure is converted into textual context rather than
    # silently returning an empty result. This keeps the model prompt complete
    # and makes the fallback failure visible in the generated response.
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

# Function: output_guardrail_node
# Purpose: Sanitize the final response and record latency, estimated cost, and intent metrics. Preserves a
# recognized response badge while applying PII redaction to its content. Metrics persistence failures
# are reported through the status logger and do not replace the completed assistant response.
def output_guardrail_node(state: dict):
    """Sanitize the final response and record latency, estimated cost, and intent metrics.

    Preserves a recognized response badge while applying PII redaction to its
    content. Metrics persistence failures are reported through the status logger
    and do not replace the completed assistant response.
    """
    last_msg = state["messages"][-1]
    
    badge = "[LLM]"
    content_str = ""
    if hasattr(last_msg, "content"):
        content_str = last_msg.content
        response_badges = (
            "[RAG]",
            "[WEB]",
            "[WEB - CRAG FALLBACK]",
            "[LLM]",
            "[SITE SELECTION]",
            "[SUPPLY_CHAIN]",
            "[AUTONOMOUS]",
        )
        for prefix in response_badges:
            if content_str.startswith(prefix):
                badge = prefix
                content_str = content_str[len(prefix):].strip()
                break
        safe_content = sanitize_text(content_str)
        last_msg.content = f"{badge} {safe_content}".strip()
        if state.get("intent") == "RAG":
            contexts = [
                doc.page_content for doc in state.get("top_docs", [])
            ]
            set_run_attribute(
                "rag.answer_context_token_overlap",
                rag_answer_context_overlap(safe_content, contexts),
            )
            set_run_attribute("rag.quality_metric_kind", "lexical_heuristic")

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
            status="SUCCESS",
            model_name=selected_model,
            prompt_version=state.get("prompt_version", CONFIG.get("version")),
            workflow_mode=state.get("workflow_mode", "default"),
            graph_topology=state.get("graph_topology", "intent-routed-v1"),
            trace_id=state.get("trace_id") or current_trace_id(),
        )
    except Exception as e:
        log_status(f"Metrics logging warning: {e}")

    return {"messages": state["messages"]}