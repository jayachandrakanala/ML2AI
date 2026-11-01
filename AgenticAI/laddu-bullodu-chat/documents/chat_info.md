# 🏛️ FOUNDATION ARCHITECTURAL SPECIFICATION & IMPLEMENTATION BLUEPRINT

## **Enterprise Multi-Agent Vector & Graph DB LangGraph Workflow Engine**

---

### 1. ARCHITECTURAL VISION & DESIGN PHILOSOPHY

This document serves as the canonical reference architecture and implementation blueprint for the **Laddu Bullodu Chat Assistant (`laddu-bullodu-chat`)**. Designed as a production-grade enterprise template, this system demonstrates how to combine **Vector Databases (ChromaDB)**, **Graph Databases (Neo4j with GDS)**, **Stateful Multi-Agent Workflows (LangGraph)**, **Cross-Encoder Re-Ranking (BGE)**, and **Corrective RAG (CRAG)** into a unified, privacy-first conversational engine.

#### Core Architectural Pillars:

1. **Deterministic Multi-Intent Routing:** Decouples user requests at the entry point, routing execution into specialized sub-graphs or isolated pipelines based on semantic domain patterns.
2. **Hybrid Graph-Vector Intelligence:** Fills the gap between unstructured knowledge retrieval (RAG) and structured enterprise relationships (Neo4j multi-tier supply chain lineage and GDS pathfinding).
3. **Self-Correcting RAG (CRAG):** Employs open-source Cross-Encoder re-ranking models (`BAAI/bge-reranker-large`) to evaluate confidence scores dynamically, falling back to live web search (`Serper.dev`) when local vector context is insufficient.
4. **Strict State Management & Guardrails:** Utilizes typed state dictionaries (`TypedDict`), immutable conversation reducers (`add_messages`), and PII sanitization filters across input ingestion and output rendering.

---

### 2. END-TO-END SYSTEM TOPOLOGY & STATEGRAPH CONTROL FLOW

The application is orchestrated by a compiled LangGraph `StateGraph` that enforces modular separation of concerns. Every query passes through a rigorous, stateful lifecycle:

```
                                      [START]
                                         │
                              ┌───(router_agent)───┐
                              │                    │
              ┌───────────────┴──────────────┐     │
              ▼                              ▼     ▼
     [intent == "RAG"]              [intent == "SUPPLY_CHAIN"] ... [OTHER DOMAINS]
              │                              │
       (retrieve_node)                (domain_executor_node)
        (k=20 Chunks)                 ├─ Neo4j Text-to-Cypher
              │                       ├─ GDS Dijkstra Pathfinding
       (rerank_node)                  └─ Multi-Agent Site Selection
     (BGE Cross-Encoder)                     │
              │                              │
       ┌──────┴──────┐                       │
       ▼             ▼                       │
  [Score ≥ 0.4]  [Score < 0.4]               │
   (High Conf)  (CRAG Trigger)               │
       │             │                       │
 (rag_generate) (crag_fallback)              │
       │             │                       │
       └──────┬──────┘                       │
              │                              │
              └──────────────┬───────────────┘
                             ▼
                        (guardrail)
                    (PII Sanitization)
                             │
                           [END]

```

---

### 3. STATE SCHEMA & DATA STRUCTURES (`State`)

The global workflow state is defined using Python's `TypedDict`, ensuring all nodes share strict type contracts and support message appending via LangGraph's `add_messages` reducer.

```python
from typing import Annotated, NotRequired, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    """Immutable conversation state passed across LangGraph nodes."""
    messages: Annotated[list, add_messages]
    selected_model: NotRequired[str]       # Active inference model (e.g., 'llama3.2', 'llama-3.3-70b-versatile')
    intent: NotRequired[str]               # Classified intent ('SUPPLY_CHAIN', 'SITE_SELECTION', 'RAG', 'WEB', 'LLM')
    raw_docs: NotRequired[list]            # Unfiltered raw documents retrieved from vector store (k=20)
    top_docs: NotRequired[list]            # High-confidence re-ranked documents (Top-3)
    top_relevance_score: NotRequired[float] # Absolute relevance score returned by Cross-Encoder

```

---

### 4. CORE NODE IMPLEMENTATIONS & WORKFLOW MODULES

#### 4.1 Adaptive Intent Router (`router_node` & `route_intent`)

The router intercepts user queries and inspects them against domain keyword registries and regular expressions to determine the downstream execution path.

- **`SUPPLY_CHAIN`**: Triggered by logistics keywords (`port congestion`, `shipping delay`, `disruption`, `part shortage`), multi-tier entities (`tier 1`, `tier 2`, `bom`, `component`, `microcontroller`), impact metrics (`revenue at risk`, `order value`, `sla breach`), mitigation strategies (`reroute`, `intermodal`, `gds`, `dijkstra`), or order ID patterns (`\bord-\d+\b`, e.g., `ORD-8801`).
- **`SITE_SELECTION`**: Triggered by commercial real estate and data center terms (`site selection`, `data center`, `commercial real estate`, `industrial park`, `retail space`).
- **`WEB`**: Triggered by recency terms (`latest`, `news`, `current`, `weather`, `today`, `stock`, `price`).
- **`RAG`**: Triggered by internal project documentation terms (`laddu`, `bullodu`, `roadmap`, `architecture`, `kanala`).
- **`LLM`**: Default fallback for general reasoning.

#### 4.2 Cross-Encoder Re-Ranking & Corrective RAG (`rerank_node`, `crag_fallback_node`)

To mitigate the LLM "lost in the middle" phenomenon and eliminate hallucinations over weak context:

1. **Wide Retrieval (`retrieve_node`)**: Fetches $k=20$ raw chunks from ChromaDB.
2. **Cross-Encoder Scoring (`reranker.py`)**: Passes query-document pairs into `BAAI/bge-reranker-large` via `sentence-transformers`:

```python
from sentence_transformers import CrossEncoder

@st.cache_resource
def load_reranker_model():
    return CrossEncoder("BAAI/bge-reranker-large")

def rerank_documents(query: str, documents: list, top_k: int = 3):
    model = load_reranker_model()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    top_scored = scored_docs[:top_k]
    return [item[0] for item in top_scored], float(top_scored[0][1]) if top_scored else 0.0

```

3. **CRAG Conditional Routing (`route_after_rerank`)**: Evaluates `top_relevance_score`:

- If score $\ge 0.4$, route to `rag_generate_node`.
- If score $< 0.4$, trigger `crag_fallback_node` to execute live web search via Google Serper.dev.

#### 4.3 Neo4j Graph Intelligence & GDS Simulation (`supply_chain.py`)

Handles complex relational queries over a Neo4j instance:

- **Text-to-Cypher Generation**: Uses few-shot prompt templates and fuzzy matching (`toLower(c.name) CONTAINS 'keyword'`) to traverse multi-tier dependencies:

$$\text{(Port)} \xrightarrow{\text{:DELIVERS\_TO}} \text{(Facility)} \xleftarrow{\text{:MANUFACTURES}} \text{(Product)} \xleftarrow{\text{:REQUIRED\_FOR}} \text{(Component)} \xleftarrow{\text{:SUPPLIES}} \text{(Supplier)}$$

- **Financial Exposure Calculation**: Computes aggregate revenue at risk (`orderValue`) tied to stalled components or congested ports (e.g., Port of Los Angeles impacting `ORD-8801` and `ORD-8802` for $1,100,000).
- **GDS Dijkstra Pathfinding**: Executes in-memory shortest-path calculations balancing transit time and cost weights to recommend optimal logistics rerouting.

#### 4.4 Multi-Agent Commercial Site Selection (`site_selector.py`)

Autonomous multi-agent sub-graph for real estate evaluations:

1. **Parameter Extraction**: Parses target countries and regions.
2. **Supervisor & Sub-Agents**: Evaluates candidate locations across Power Grid availability, fiber density, tax incentives, seismic risk, and labor costs.
3. **Report Generation**: Compiles structured comparison reports and persists markdown artifacts to `./reports/`.

#### 4.5 Output Guardrail Node (`output_guardrail_node`)

Applies PII sanitization (`sanitize_text`) to assistant responses while preserving mode badges (`[SUPPLY_CHAIN]`, `[RAG]`, `[WEB]`, `[SITE SELECTION]`, `[LLM]`, `[MULTI-QUERY RAG]`).

---

### 5. DATA INGESTION & VECTOR STORAGE PIPELINE

Documents placed in `./documents` (.txt, .md) are automatically processed by `load_embeddings_and_vector_store`:

1. **Sanitization**: PII and sensitive tokens are scrubbed from raw text.
2. **Chunking**: `RecursiveCharacterTextSplitter` splits text into chunks of 500 characters with a 50-character overlap.
3. **Embedding & Upserting**: Vectorized via Ollama Embeddings (`embeddinggemma`) or local fallback embeddings, then upserted into ChromaDB (`./chroma_db`) with deterministic IDs based on file modification timestamps.

---

### 6. AUTOMATED REGRESSION TESTING FRAMEWORK (`test_supply_chain.py`)

A robust 20-scenario parametrized integration test suite validates system reliability across:

- **Financial Impact & Revenue at Risk** (`TC-01` to `TC-03`)
- **Multi-Tier Bill of Materials Lineage** (`TC-04` to `TC-06`)
- **Single Points of Failure & Bottlenecks** (`TC-07` to `TC-09`)
- **GDS Dijkstra Pathfinding & Rerouting** (`TC-10` to `TC-12`)
- **Mitigation Strategy Trade-offs** (`TC-13` to `TC-15`)
- **Order SLA Breaches & Due Dates** (`TC-16` to `TC-17`)
- **Conceptual Reasoning & Executive Summaries** (`TC-18` to `TC-20`)

#### Execution Commands:

```bash
# Run full regression suite with verbose logging
python -m pytest test_supply_chain.py -v

# Run targeted scenario tests
python -m pytest test_supply_chain.py -k "TC-01 or TC-02 or TC-04" -v

```

---

### 7. QUICKSTART DEPLOYMENT & IMPLEMENTATION CHECKLIST

1. **Environment Setup**: Install Python 3.13, Streamlit, LangGraph, LangChain, ChromaDB, Neo4j driver, and sentence-transformers.
2. **Local Services**: Ensure Ollama is running locally (`http://localhost:11434`) with `llama3.2` and `embeddinggemma` pulled.
3. **API Keys**: Configure `.env` with `SERPER_API_KEY`, `GROQ_API_KEY`, and Neo4j connection credentials (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`).
4. **Run Application**:

```bash
streamlit run app.py

```
