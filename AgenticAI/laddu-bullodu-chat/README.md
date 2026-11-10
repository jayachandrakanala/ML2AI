# 🥭 Laddu Bullodu: Agentic RAG and AI Operations Assistant

**Project:** Agentic Retrieval-Augmented Generation (RAG), supply-chain intelligence, commercial real-estate site selection, and live web research  
**Interface:** Streamlit  
**Orchestration:** LangGraph and LangChain

---

## Table of Contents

- [🥭 Laddu Bullodu: Agentic RAG and AI Operations Assistant](#-laddu-bullodu-agentic-rag-and-ai-operations-assistant)
  - [Table of Contents](#table-of-contents)
  - [📌 Project Overview](#-project-overview)
  - [🚀 Technical Highlights](#-technical-highlights)
  - [🛠️ Technology Stack](#️-technology-stack)
  - [Architecture and Data Flow](#architecture-and-data-flow)
  - [Key Features](#key-features)
    - [🏢 Multi-Agent Site Selection](#-multi-agent-site-selection)
    - [🎯 Intent Routing](#-intent-routing)
    - [🔎 Hybrid Retrieval and Corrective RAG](#-hybrid-retrieval-and-corrective-rag)
    - [🛡️ Guardrails and Privacy](#️-guardrails-and-privacy)
    - [📦 Supply-Chain Graph Intelligence](#-supply-chain-graph-intelligence)
  - [Project Structure](#project-structure)
  - [Module Responsibilities](#module-responsibilities)
    - [`app.py`](#apppy)
    - [`src/agent_graph.py`](#srcagent_graphpy)
    - [`src/graph_nodes.py`](#srcgraph_nodespy)
    - [`src/site_selector.py`](#srcsite_selectorpy)
    - [`src/supply_chain.py`](#srcsupply_chainpy)
    - [`src/guardrails.py`](#srcguardrailspy)
    - [`src/logger.py` and `src/metrics_logger.py`](#srcloggerpy-and-srcmetrics_loggerpy)
  - [Setup Project](#setup-project)
    - [Prerequisites](#prerequisites)
    - [1. Clone and create a virtual environment](#1-clone-and-create-a-virtual-environment)
    - [2. Install dependencies](#2-install-dependencies)
    - [3. Pull the Ollama models](#3-pull-the-ollama-models)
    - [4. Configure environment variables](#4-configure-environment-variables)
    - [5. Start the application](#5-start-the-application)
  - [Neo4j Setup](#neo4j-setup)
  - [Runtime Data](#runtime-data)
  - [Configuration](#configuration)
  - [Running Tests](#running-tests)
  - [Screenshots](#screenshots)
  - [Concept Notes](#concept-notes)
- [LLM Evaluation \& Observability Platforms: Comparison Guide](#llm-evaluation--observability-platforms-comparison-guide)
  - [Quick Comparison](#quick-comparison)
  - [Detailed Breakdown](#detailed-breakdown)
    - [🛠️ LangSmith](#️-langsmith)
    - [🧪 Braintrust](#-braintrust)
    - [🌐 Langfuse](#-langfuse)
  - [Decision Matrix](#decision-matrix)
  - [What is LangSmith?](#what-is-langsmith)
    - [Key Benefits of LangSmith for the laddu bullodu Chat Assistant](#key-benefits-of-langsmith-for-the-laddu-bullodu-chat-assistant)
      - [1. Deep Observability into LangGraph Multi-Agent Flows](#1-deep-observability-into-langgraph-multi-agent-flows)
      - [2. Fine-Grained Prompt \& Parameter Extraction Debugging](#2-fine-grained-prompt--parameter-extraction-debugging)
      - [3. RAG Retrieval \& Vector Store Inspection](#3-rag-retrieval--vector-store-inspection)
      - [4. Guardrails \& PII Sanitization Verification](#4-guardrails--pii-sanitization-verification)
      - [5. Cost, Latency \& Token Usage Analytics](#5-cost-latency--token-usage-analytics)
    - [Step-by-Step Instructions: Setting Up LangSmith in laddu bullodu](#step-by-step-instructions-setting-up-langsmith-in-laddu-bullodu)
      - [Step 1: Create a Free LangSmith Account \& Get an API Key](#step-1-create-a-free-langsmith-account--get-an-api-key)
      - [Step 2: Configure Environment Variables in `.env`](#step-2-configure-environment-variables-in-env)
      - [Step 3: Run the Assistant and View Live Traces](#step-3-run-the-assistant-and-view-live-traces)
      - [Step 4: Inspect execution runs in the LangSmith UI](#step-4-inspect-execution-runs-in-the-langsmith-ui)
      - [Step 5: Step-by-Step Guide to Enable Graph View](#step-5-step-by-step-guide-to-enable-graph-view)
    - [Neo4j Setup](#neo4j-setup-1)
      - [Neo4j running under Docker](#neo4j-running-under-docker)
      - [Neo4j from browser](#neo4j-from-browser)
  - [🚀 Recent Architecture \& Feature Enhancements](#-recent-architecture--feature-enhancements)
    - [1. Advanced Neo4j Graph Intelligence \& Text-to-Cypher Optimization](#1-advanced-neo4j-graph-intelligence--text-to-cypher-optimization)
    - [2. Comprehensive Automated Regression Test Suite](#2-comprehensive-automated-regression-test-suite)
    - [3. Open-Source Cross-Encoder Re-Ranking \& Corrective RAG (CRAG)](#3-open-source-cross-encoder-re-ranking--corrective-rag-crag)
    - [4. Stateful Multi-Node LangGraph Refactoring](#4-stateful-multi-node-langgraph-refactoring)
    - [5. Enhanced Intent Classification](#5-enhanced-intent-classification)
  - [5 sample questions](#5-sample-questions)
    - [1. Supply Chain Intelligence \& Graph Analytics (`SUPPLY_CHAIN`)](#1-supply-chain-intelligence--graph-analytics-supply_chain)
    - [2. Multi-Agent Commercial Real Estate Site Selection (`SITE_SELECTION`)](#2-multi-agent-commercial-real-estate-site-selection-site_selection)
    - [3. Re-Ranked Vector RAG \& Cross-Encoder Filtering (`RAG`)](#3-re-ranked-vector-rag--cross-encoder-filtering-rag)
    - [4. Live Web Intelligence Search (`WEB`)](#4-live-web-intelligence-search-web)
    - [5. General LLM Reasoning \& Guardrails (`LLM`)](#5-general-llm-reasoning--guardrails-llm)
  - [Use Re-Ranking in a LangGraph Agentic System](#use-re-ranking-in-a-langgraph-agentic-system)
    - [1. In a Pre-Generation Node (Filtering Noise before the LLM)](#1-in-a-pre-generation-node-filtering-noise-before-the-llm)
    - [2. Inside a Multi-Query or Query-Expansion Loop](#2-inside-a-multi-query-or-query-expansion-loop)
    - [3. To Guard Conditional Edge Routing (Corrective RAG)](#3-to-guard-conditional-edge-routing-corrective-rag)
    - [4. How Huggingface BGE works?](#4-how-huggingface-bge-works)
      - [4.1. Architectural Context: What Happens in the rerank\_node](#41-architectural-context-what-happens-in-the-rerank_node)
      - [4.2. Understanding the Model: bge-reranker-large](#42-understanding-the-model-bge-reranker-large)
      - [4.3. Why This Matters to Your System State](#43-why-this-matters-to-your-system-state)
  - [BM25 in RAG Systems](#bm25-in-rag-systems)
    - [How BM25 Works (Key Principles)](#how-bm25-works-key-principles)
    - [How BM25 Improves Your RAG System](#how-bm25-improves-your-rag-system)
  - [vLLM vs SGLang Core Difference](#vllm-vs-sglang-core-difference)
    - [Architectural Highlights](#architectural-highlights)
  - [💡 Neo4j: Native Graph Storage \& Processing](#-neo4j-native-graph-storage--processing)
    - [🔎 Cypher Query Language](#-cypher-query-language)
    - [📊 Flexibility (Schemaless Model)](#-flexibility-schemaless-model)
    - [🛠 Key Industry Use Cases](#-key-industry-use-cases)
  - [About TypedDict](#about-typeddict)
    - [Why Use TypedDict?](#why-use-typeddict)
    - [Key Features \& Constraints](#key-features--constraints)
  - [LangGraph Streams](#langgraph-streams)
    - [What each event contains](#what-each-event-contains)
    - [Why use it?](#why-use-it)
    - [https://www.datacamp.com/tutorial/guide-to-autoresearch](#httpswwwdatacampcomtutorialguide-to-autoresearch)
  - [Prompt Versioning (config/prompts\_v1.json)](#prompt-versioning-configprompts_v1json)
    - [How to Validate Prompt Versioning](#how-to-validate-prompt-versioning)
- [LangGraph Persistence \& Checkpointing Architecture](#langgraph-persistence--checkpointing-architecture)
  - [💾 Short-Term Memory Providers](#-short-term-memory-providers)
    - [1. Development Implementation: `MemorySaver`](#1-development-implementation-memorysaver)
    - [2. Production Implementation: External Databases](#2-production-implementation-external-databases)
      - [Basic Setup Example (PostgreSQL)](#basic-setup-example-postgresql)
  - [🗄️ Database Schema \& Data Lifecycle](#️-database-schema--data-lifecycle)
    - [The Execution State Loop](#the-execution-state-loop)
  - [🚀 Supported Backend Integrations](#-supported-backend-integrations)
  - [⚠️ Architectural Guardrails](#️-architectural-guardrails)
  - [LangGraph MemorySaver Checkpointer](#langgraph-memorysaver-checkpointer)
    - [How to Validate MemorySaver](#how-to-validate-memorysaver)
    - [Upgrade to SQLite](#upgrade-to-sqlite)
    - [How SQLite Works \& How to Validate](#how-sqlite-works--how-to-validate)
    - [Method 2: The Server Restart \& State Persistence Test](#method-2-the-server-restart--state-persistence-test)
    - [Why This Solves the Problem](#why-this-solves-the-problem)
  - [SQLite vs mem0](#sqlite-vs-mem0)
    - [1. What They Actually Do](#1-what-they-actually-do)
    - [2. Direct Feature Breakdown](#2-direct-feature-breakdown)
    - [3. Which Is Best for Your Application?](#3-which-is-best-for-your-application)
      - [**Scenario A: Stick with `SqliteSaver` (Plus LangGraph's native Store) if:**](#scenario-a-stick-with-sqlitesaver-plus-langgraphs-native-store-if)
      - [**Scenario B: Integrate Mem0 if:**](#scenario-b-integrate-mem0-if)
    - [**The Recommended Production Architecture**](#the-recommended-production-architecture)
  - [OpenTelemetry (OTel) is the industry-standard approach for modern agentic applications.](#opentelemetry-otel-is-the-industry-standard-approach-for-modern-agentic-applications)
  - [**Why OpenTelemetry is Superior to Custom Telemetry**](#why-opentelemetry-is-superior-to-custom-telemetry)
    - [**1. Tracing Nested \& Asynchronous Graph Execution**](#1-tracing-nested--asynchronous-graph-execution)
    - [**2. Zero-Boilerplate Instrumentation**](#2-zero-boilerplate-instrumentation)
    - [**3. Vendor-Agnostic Telemetry Export**](#3-vendor-agnostic-telemetry-export)
    - [**Comparison: Current Custom Setup vs. OpenTelemetry**](#comparison-current-custom-setup-vs-opentelemetry)
    - [**How OpenTelemetry Integrates with LangGraph**](#how-opentelemetry-integrates-with-langgraph)
    - [**Recommended Migration Strategy**](#recommended-migration-strategy)
      - [How to Validate Your New Observability Stack](#how-to-validate-your-new-observability-stack)
  - [OpenAPI Spec (OAS) vs Model Context Protocol (MCP)](#openapi-spec-oas-vs-model-context-protocol-mcp)
    - [Architectural Shift: How They Work](#architectural-shift-how-they-work)
    - [Why MCP Matters (The M × N Problem)](#why-mcp-matters-the-m--n-problem)
    - [Coexistence in Production](#coexistence-in-production)
- [LLM Evaluation Framework Comparison: Ragas vs. DeepEval](#llm-evaluation-framework-comparison-ragas-vs-deepeval)
  - [Core Focus \& Use Case](#core-focus--use-case)
  - [Key Differences](#key-differences)
  - [Metrics Compared](#metrics-compared)
  - [Decision Matrix](#decision-matrix-1)

---

---

---

## 📌 Project Overview

**Laddu Bullodu** is a multi-agent AI assistant built around LangGraph workflows and a Streamlit chat interface. It combines locally stored documents, hybrid retrieval, cross-encoder re-ranking, live web search, Neo4j supply-chain analysis, commercial real-estate research, and direct LLM reasoning.

The application supports per-browser chat threads persisted in SQLite and streams graph progress into the chat UI. Responses pass through a guardrail node that removes detected personal information and records latency, estimated cost, intent, and status in the local metrics database.

![alt text](images/image_agent_flow.png)
![alt text](images/image_rag_vectordb.png)

> Add your own documents to `documents/`; application startup indexes supported `.md` and `.txt` files into the local ChromaDB collection.

## 🚀 Technical Highlights

- **Intent-driven LangGraph workflow:** routes requests among `RAG`, `WEB`, `SUPPLY_CHAIN`, `SITE_SELECTION`, and `LLM`.
- **Hybrid local retrieval:** combines dense vector search with BM25 lexical search and Reciprocal Rank Fusion.
- **Cross-encoder re-ranking:** scores fused candidates with BAAI `bge-reranker-large` and keeps the strongest context.
- **Corrective RAG fallback:** web search is used when the top re-ranking relevance score is below the configured CRAG threshold.
- **Neo4j supply-chain intelligence:** generates and executes Cypher, with optional Neo4j GDS weighted routing analysis.
- **Multi-agent site selection:** gathers market evidence, evaluates candidates, and writes Markdown investment reports under `reports/`.
- **Model selection:** switch between local Ollama `llama3.2` and the Groq Qwen model configured in the UI.
- **Persistent state:** LangGraph checkpoints browser conversations in `checkpoints.sqlite`.
- **Observability:** local metrics are stored in SQLite, and application traces can be exported to Phoenix/OpenTelemetry.

## 🛠️ Technology Stack

| Area              | Technology used by the current codebase                                  |
| ----------------- | ------------------------------------------------------------------------ |
| UI                | Streamlit                                                                |
| Agent framework   | LangGraph, LangChain                                                     |
| Chat models       | Ollama, Groq, and OpenAI via their LangChain integrations                |
| Local embeddings  | Ollama `embeddinggemma`, or local Hugging Face EmbeddingGemma embeddings |
| Vector database   | ChromaDB                                                                 |
| Lexical retrieval | `rank_bm25`                                                              |
| Re-ranking        | Sentence Transformers / BAAI BGE cross-encoder                           |
| Web search        | Google Serper API via LangChain                                          |
| Graph database    | Neo4j and Neo4j Graph Data Science                                       |
| Privacy           | Microsoft Presidio plus custom regex-backed sanitization                 |
| Checkpointing     | LangGraph SQLite saver                                                   |
| Metrics           | SQLite                                                                   |
| Tracing           | OpenInference LangChain instrumentation, OpenTelemetry, Phoenix          |

## Architecture and Data Flow

```text
Streamlit chat UI
       │
       ▼
LangGraph router
       │
       ├── RAG ──► Hybrid retrieval ──► BGE re-ranking ──► answer
       │                │ low confidence
       │                └──────────────► web-backed CRAG fallback
       │
       ├── WEB ───────────────► Serper search ──► LLM synthesis
       ├── SUPPLY_CHAIN ──────► Text-to-Cypher ──► Neo4j/GDS ──► memo
       ├── SITE_SELECTION ────► research ──► evaluate ──► save Markdown report
       └── LLM ───────────────► direct model response
                                     │
                                     ▼
                              PII guardrail + metrics
```

The main graph is implemented by `src/agent_graph.py`. Conversation state contains bounded chat messages, the selected model, classified intent, retrieved documents, the re-ranked relevance score, and the workflow start time. Responses are streamed to `app.py` with real-time status updates.

## Key Features

### 🏢 Multi-Agent Site Selection

The site-selection subgraph has three stages: supervisor, evaluator, and synthesizer.

1. The supervisor performs three focused Serper searches per location: zoning/environment, power infrastructure, and financial incentives.
2. The evaluator asks the selected chat model to produce structured scores and pros/cons.
3. The synthesizer compares locations and saves an investment memo under `reports/`.

Regional defaults are available for the United States, Canada, Asia-Pacific, and Europe, with safeguards that override U.S. defaults for explicitly international queries.

### 🎯 Intent Routing

The keyword-first router in `src/intent_router.py` classifies a query as:

- `SUPPLY_CHAIN` for logistics, supply disruption, tiers, components, routing, orders, and graph-analysis terms.
- `SITE_SELECTION` for commercial real estate, data centers, property development, and location investment terms.
- `WEB` for current, recent, news, weather, price, or search-oriented requests.
- `RAG` for Laddu Bullodu project or local-document questions.
- `LLM` for everything else.

### 🔎 Hybrid Retrieval and Corrective RAG

The RAG branch retrieves the configured number of dense and BM25 candidates, merges them with Reciprocal Rank Fusion, and sends the best candidates to the BGE cross-encoder. The current defaults in `config/prompts_v1.json` are:

- retrieval candidates: 20
- re-ranked documents: 3
- CRAG relevance threshold: 0.4

If the highest relevance score is below the threshold, the assistant performs a Serper web search and asks the selected model to answer from those live results instead.

### 🛡️ Guardrails and Privacy

The application sanitizes local documents before chunking and sanitizes generated responses before they are shown. This reduces accidental exposure of common personal data, but it does not replace organizational privacy review, secure secret handling, access controls, or compliance processes.

### 📦 Supply-Chain Graph Intelligence

Supply-chain queries can generate Neo4j Cypher from natural language, query connected supplier/component/product/facility/warehouse/order data, and run Neo4j Graph Data Science Dijkstra analysis for weighted logistics routes. Supply-chain execution requires Neo4j and the Graph Data Science plugin, as described below.

## Project Structure

```text
app.py                         Streamlit UI and session handling
agent_graph.py                 LangGraph CLI entry configuration
langgraph.json                 LangGraph CLI configuration
config/prompts_v1.json         Prompt and retrieval configuration
documents/                     Source documents indexed at startup
src/
  agent_graph.py               Main LangGraph state graph
  graph_nodes.py               Router, retrieval, RAG, web, domain, and guardrail nodes
  intent_router.py             Intent classification and site parameter extraction
  hybrid_search.py             BM25 and Reciprocal Rank Fusion
  reranker.py                  BGE cross-encoder loading and re-ranking
  local_embeddings.py          Local Hugging Face EmbeddingGemma wrapper
  llm_provider.py              Ollama/Groq model configuration
  site_selector.py             Site-selection subgraph and report generation
  supply_chain.py              Neo4j Text-to-Cypher and GDS subgraph
  guardrails.py                PII sanitization
  metrics_logger.py            SQLite operational metrics
  telemetry.py                 Phoenix/OpenTelemetry initialization
reports/                       Generated site-selection Markdown reports
test/                          Routing, retrieval, evaluation, and supply-chain tests
```

## Module Responsibilities

### `app.py`

- Configures the Streamlit page and chat UI.
- Keeps an independent `thread_id` for each browser session.
- Streams LangGraph updates into `st.status()` containers.
- Resumes interrupted SQLite checkpoint executions.
- Displays latency, cost, failure rate, and chat-model controls.

### `src/agent_graph.py`

- Defines the main typed LangGraph state and bounds stored messages to a recent 12-message context window.
- Creates graph nodes for routing, retrieval, re-ranking, domain execution, generation, CRAG fallback, and output guardrails.
- Persists graph state in `checkpoints.sqlite` and creates thread-local SQLite connections for streamed invocations.

### `src/graph_nodes.py`

- Loads documents and ChromaDB.
- Adds dense vector and BM25 results to the hybrid retrieval pipeline.
- Executes the RAG, CRAG, web, site-selection, supply-chain, and direct LLM paths.
- Sanitizes the final response and records execution telemetry.

### `src/site_selector.py`

- Implements the supervisor → evaluator → synthesizer subgraph.
- Searches Serper for zoning, power, and financial evidence for each candidate.
- Writes Markdown reports with sanitized filenames.

### `src/supply_chain.py`

- Generates Cypher from natural language and queries Neo4j.
- Includes fallback graph queries and weighted Neo4j GDS route analysis.
- Synthesizes a mitigation memo from graph context.

### `src/guardrails.py`

- Sanitizes common personally identifiable information in indexed documents and model output.

### `src/logger.py` and `src/metrics_logger.py`

- Bridge LangGraph status updates to Streamlit without circular imports.
- Record execution latency, estimated cost, intent, citation mode, status, and summary statistics.

## Setup Project

### Prerequisites

- Python 3.10 or newer
- Git
- Ollama with `llama3.2` and `embeddinggemma`
- Groq API key only when using the Groq chat model
- Serper API key for `WEB`, CRAG fallback, and `SITE_SELECTION` workflows
- Neo4j plus the Graph Data Science plugin for supply-chain graph workflows

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/jayachandrakanala/ML2AI.git
cd AgenticAI/ML2AI/laddu-bullodu-chat
python -m venv venv
```

Activate the environment:

macOS/Linux:

```bash
source venv/bin/activate
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Pull the Ollama models

```bash
ollama pull llama3.2
ollama pull embeddinggemma
```

The default Ollama server URL in the current code is `http://192.168.0.169:11434`. Change `OLLAMA_BASE_URL` in `src/llm_provider.py` if Ollama is running on another machine or localhost. If Ollama runs locally, use:

```python
OLLAMA_BASE_URL = "http://localhost:11434"
```

### 4. Configure environment variables

Create `.env` in the project root:

```dotenv
# Required for Groq model selection
GROQ_API_KEY=your_groq_api_key

# Required for OpenAI model selection
OPENAI_API_KEY=your_openai_api_key

# Required for live web and site-selection workflows
SERPER_API_KEY=your_serper_api_key

# Required only for SUPPLY_CHAIN
NEO4J_URI=neo4j+s://your-neo4j-host
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j

# Optional LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=laddu-bullodu-assistant
```

Never commit `.env` or real credentials. The repository `.gitignore` excludes environment files, SQLite state, virtual environments, generated ChromaDB data, Phoenix telemetry, caches, and reports.

### 5. Start the application

```bash
ollama serve
streamlit run app.py
```

Open `http://localhost:8501`.

Ollama, Groq, and OpenAI are available in the chat-model selector. Ollama can provide local chat and embeddings; Groq requires `GROQ_API_KEY`; OpenAI requires `OPENAI_API_KEY`. RAG chooses local Hugging Face embedding resources when Groq is selected and Ollama embeddings for Ollama and OpenAI selections.

## Neo4j Setup

Supply-chain workflows can be enabled with a Neo4j instance and the Graph Data Science library:

```bash
docker run --name laddu-bullodu-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/change-this-password \
  -e NEO4J_PLUGINS='["graph-data-science"]' \
  -e NEO4J_dbms_security_procedures_unrestricted='gds.*' \
  neo4j:latest
```

Load your supply-chain fixture using the project's Cypher setup, such as `neo4j_data.sql`, and then configure the matching Neo4j values in `.env`.

## Runtime Data

The application creates or updates these local artifacts during normal use:

- `chroma_db/`: generated document vectors
- `checkpoints.sqlite`: LangGraph conversation checkpoints
- `data/metrics.sqlite`: operational metrics
- `data/phoenix/`: Phoenix trace storage
- `reports/`: generated site-selection reports
- `.langgraph_api/`: optional LangGraph CLI state

These runtime and generated files should remain excluded from public source control.

## Configuration

Versioned prompts and thresholds live in `config/prompts_v1.json`. Update that file to change:

- `crag_relevance_threshold`
- `top_k_retrieval`
- `top_k_rerank`
- RAG, CRAG, web, and site-selection prompts

## Running Tests

Run the focused repository tests from the project root:

```bash
pytest test/test_rrf_search.py test/test_supply_chain.py test/test_evals_ragas.py
```

Supply-chain integration tests may require Neo4j configuration, seeded graph data, and API credentials. Unit and retrieval tests can run independently.

## Screenshots

![Laddu Bullodu interface](images/image.png)
![Site selection workflow](images/image-1.png)
![Architecture overview](images/image-2.png)

## Concept Notes

The remainder of this README preserves deeper explanations of the concepts and integrations used by the project, including re-ranking, BM25, Neo4j, LangGraph persistence, OpenTelemetry, prompt versioning, MCP/OpenAPI, and evaluation frameworks.

---

# LLM Evaluation & Observability Platforms: Comparison Guide

Choosing between **LangSmith**, **Braintrust**, and **Langfuse** depends heavily on your existing orchestration stack, infrastructure constraints, and development workflow.

---

## Quick Comparison

| Feature / Focus        | LangSmith                                                     | Braintrust                                              | Langfuse                                              |
| :--------------------- | :------------------------------------------------------------ | :------------------------------------------------------ | :---------------------------------------------------- |
| **Primary Strength**   | Native tracing & debugging for LangChain/LangGraph ecosystems | Dataset-driven evals, experiments, and regression gates | Open-source, framework-agnostic LLM observability     |
| **Open Source Status** | Proprietary / Closed-source SaaS                              | Proprietary platform (Open SDKs)                        | MIT Open-source core                                  |
| **Self-Hosting**       | Restricted to Enterprise tier                                 | Hybrid / Enterprise (BYOC options)                      | Available across all tiers (Fully self-hostable)      |
| **Best Suited For**    | Heavy LangChain/LangGraph users                               | Quality-first teams running automated CI/CD checks      | Multi-framework teams, cost control, and data privacy |

---

## Detailed Breakdown

### 🛠️ LangSmith

- **Overview:** Built by LangChain, LangSmith provides deep step-by-step tracing, prompt debugging, dataset creation, and evaluation tailored for complex agentic workflows.
- **Best for:** Teams deeply invested in the **LangChain** and **LangGraph** ecosystems who want a managed, tightly integrated SaaS environment for graph inspection and agent state tracking.
- **Trade-offs:** Closed-source core with self-hosting locked behind enterprise agreements.

### 🧪 Braintrust

- **Overview:** Focuses heavily on the evaluation loop—turning real-world failures into reusable test cases, running prompt/model experiments, and blocking regressions via CI/CD pipelines.
- **Best for:** Engineering teams where systematic evaluation, structured logging, and quantitative model regression testing drive the deployment cycle.
- **Trade-offs:** Proprietary platform control plane, though SDKs and evaluator libraries are open-source.

### 🌐 Langfuse

- **Overview:** Offers open-source LLM observability, prompt management, tracing, token/cost tracking, and scoring with a framework-neutral footprint.
- **Best for:** Multi-framework stacks (custom code, LlamaIndex, LangChain), tight budgets, or strict data residency/self-hosting requirements.
- **Trade-offs:** Less out-of-the-box specialization for native LangChain/LangGraph agent orchestration primitives compared to LangSmith.

---

## Decision Matrix

- Choose **LangSmith** if you are building production-grade agents with LangChain/LangGraph and want a unified, managed toolchain.
- Choose **Braintrust** if your main bottleneck is quantitative quality control, prompt experimentation, and setting up automated CI/CD regression gates.
- Choose **Langfuse** if you need an MIT-licensed, self-hostable, framework-agnostic solution with zero infrastructure lock-in.

## What is LangSmith?

**LangSmith** is an enterprise-grade observability, tracing, evaluation, and monitoring platform built by LangChain. It provides complete visibility into LLM-powered applications and multi-agent workflows.

Instead of treating an LLM execution as a black box where you only see the final response, LangSmith captures the **entire execution tree**—from the exact user prompt, routing decisions, and vector database retrievals to sub-agent reasoning loops, API payload calls, token usage, latency, and costs.

---

### Key Benefits of LangSmith for the laddu bullodu Chat Assistant

Integrating LangSmith into **laddu bullodu** solves several key operational challenges across your multi-agent architecture (`agent_graph.py`, `site_selector.py`, `guardrails.py`, and `app.py`):

#### 1. Deep Observability into LangGraph Multi-Agent Flows

- **The Problem:** In your site selection engine (`site_selector.py`), a single query triggers a supervisor node, Serper web search workers, sub-agent evaluations, and report generation. If a sub-agent generates an incomplete report, pinpointing which node failed can be difficult.
- **LangSmith Benefit:** Captures state transitions, inputs, and outputs node-by-node. You can inspect the exact payload sent to Serper.dev and see how each sub-agent scored zoning, power, or financial metrics.

#### 2. Fine-Grained Prompt & Parameter Extraction Debugging

- **The Problem:** The `extract_site_parameters()` function uses local Ollama calls (`llama3.2`) to extract target countries and locations into JSON formats. When parsing fails or takes too long, local console logs alone may not capture the raw text generated by Ollama.
- **LangSmith Benefit:** Logs raw model outputs, token generation speed, and temperature settings per call. You can immediately see if Ollama timed out or returned malformed JSON.

#### 3. RAG Retrieval & Vector Store Inspection

- **The Problem:** When queries trigger `[RAG]`, ChromaDB returns context chunks derived from local documents.
- **LangSmith Benefit:** Shows the exact top-$k$ document chunks retrieved from ChromaDB alongside similarity scores. This helps confirm whether context chunking in `agent_graph.py` provided sufficient data to answer questions correctly.

#### 4. Guardrails & PII Sanitization Verification

- **The Problem:** The app routes data through `guardrails.py` to sanitize PII prior to vector store upsert and UI rendering.
- **LangSmith Benefit:** You can compare raw text outputs against sanitized outputs directly within trace spans to ensure sensitive information (like emails or IDs) is properly redacted before reaching the UI.

#### 5. Cost, Latency & Token Usage Analytics

- **The Problem:** Running local LLMs (`llama3.2`) combined with external Serper API searches requires tracking overall pipeline latency.
- **LangSmith Benefit:** Provides dashboards tracking P50/P99 latency, token consumption, and error rates per execution path (`[SITE_SELECTION]`, `[RAG]`, `[WEB]`, `[LLM]`).

---

### Step-by-Step Instructions: Setting Up LangSmith in laddu bullodu

LangSmith integrates natively with LangChain and LangGraph. Enabling full workspace tracing requires no structural code rewrites—only zero-code environment variable setup.

#### Step 1: Create a Free LangSmith Account & Get an API Key

1. Go to [smith.langchain.com](https://smith.langchain.com) and sign up/log in.
2. Click on **Settings** (gear icon) in the bottom-left corner $\rightarrow$ **API Keys**.
3. Click **Create API Key**, copy the key (starts with `lsv2_pt_...`), and save it securely.

#### Step 2: Configure Environment Variables in `.env`

Open your project's `.env` file (`E:\Products\laddu-bullodu-chat\.env`) and add the following lines:

```env
# Enable LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_actual_langsmith_api_key_here"
LANGCHAIN_PROJECT="laddu-bullodu-assistant"

```

#### Step 3: Run the Assistant and View Live Traces

1. Start the Streamlit application:

```bash
streamlit run app.py

```

2. Send a query in the chat interface (for example: _"Is Malaysia a good investment location for buying a shopping complex?"_).
3. Open your browser to [smith.langchain.com](https://smith.langchain.com) and select the **`laddu-bullodu-assistant`** project.

#### Step 4: Inspect execution runs in the LangSmith UI

Inside the LangSmith dashboard, click on any execution trace to view:

- **The Root Node (`app_graph`)**: Displays overall latency and completion status.
- **`router_agent` Node**: Shows how `route_intent` evaluated the query and dispatched it.
- **`site_selection_graph` Sub-Graph**: Shows child spans for `supervisor`, `evaluator`, and `synthesizer` nodes, including raw Serper search results and model responses.
- **`guardrail` Node**: Displays input vs. sanitized output comparisons.

#### Step 5: Step-by-Step Guide to Enable Graph View

Step 1: Install the LangGraph CLI
Open your terminal inside your project directory and install the development CLI package:

Bash
pip install -U "langgraph-cli[inmem]"

Step 2: Add a langgraph.json Configuration File
Create a new file named langgraph.json in your project root (E:\Products\laddu-bullodu-chat\langgraph.json) to point LangSmith Studio to your compiled graph:

{
"dependencies": ["."],
"graphs": {
"laddu_bullodu_agent": "./agent_graph.py:app_graph"
},
"env": ".env"
}

Step 3: Launch the Local Agent Server
Run the local development server in your terminal:

Bash Command:
langgraph dev

### Neo4j Setup

#### Neo4j running under Docker

docker run \
 --name neo4j-supply-chain \
 -p 7687:7687 -p 7474:7474 \
 -e NEO4J*AUTH=neo4j/p!@1-4w \
 -e NEO4J_PLUGINS='["graph-data-science"]' \
 -e NEO4J_dbms_security_procedures_unrestricted='gds.*' \
 -e NEO4J*dbms_security_procedures_allowlist='gds.*' \
 neo4j:latest

#### Neo4j from browser

http://localhost:7474/browser/

Here is a comprehensive summary of today's session, formatted in clean Markdown so you can easily append or integrate it directly into your `README.md` file:

---

## 🚀 Recent Architecture & Feature Enhancements

### 1. Advanced Neo4j Graph Intelligence & Text-to-Cypher Optimization

- **Dynamic Cypher Generation:** Refined the few-shot prompting strategy in `supply_chain.py` to prevent LLM hallucinations and ensure accurate relationship traversals.
- **Component-Level Fuzzy Matching:** Added robust relationship mapping and string matching (`toLower(c.name) CONTAINS 'keyword'`) to seamlessly trace multi-tier lineages from upstream suppliers and components (e.g., _A16 Microcontrollers_) through assembly facilities to customer orders.
- **Financial Exposure Validation:** Successfully validated end-to-end calculations, such as tracking port congestion at the _Port of Los Angeles_ to compute real-time revenue at risk ($1,100,000 across impacted orders `ORD-8801` and `ORD-8802`).

### 2. Comprehensive Automated Regression Test Suite

- **20 Complex Scenarios:** Implemented an expanded `pytest` suite in `test_supply_chain.py` covering financial impact analysis, multi-tier BOM lineage, single points of failure (SPOF), GDS Dijkstra pathfinding, and SLA breach risks.
- **Parameterized Validation:** Automated validation checks across intent routing, sub-graph execution, and end-to-end agent guardrail integration.

### 3. Open-Source Cross-Encoder Re-Ranking & Corrective RAG (CRAG)

- **High-Precision Re-Ranking (`reranker.py`):** Integrated HuggingFace's open-source `BAAI/bge-reranker-large` model via `sentence-transformers` to score and prioritize retrieved context.
- **Pre-Generation Noise Filtering:** The system fetches a wide net of raw documents (`k=20`) and filters them down to the top 3 high-confidence chunks, eliminating the LLM "lost in the middle" phenomenon.
- **Corrective RAG (CRAG) Thresholding:** Implemented a dynamic confidence check (threshold < 0.4). If local vector search yields low confidence, the workflow automatically reroutes to live web search (`Serper.dev`) as a fallback.
- **Multi-Query Expansion Loop:** Added support for parallel query expansion and global result de-duplication to maximize search coverage.

### 4. Stateful Multi-Node LangGraph Refactoring

- **Granular Control Flow:** Refactored the monolithic `adaptive_router_node` into dedicated stateful nodes in `agent_graph.py`:
- `router_node`: Intent classification (`SUPPLY_CHAIN`, `SITE_SELECTION`, `RAG`, `WEB`, `LLM`).
- `retrieve_node` & `rerank_node`: Isolated vector retrieval and cross-encoder re-ranking stages.
- `crag_fallback_node`: Automated fallback routing for low-confidence RAG queries.
- `domain_executor_node`: Execution hub for Neo4j graph simulations and multi-agent site selection.
- `guardrail`: Centralized PII sanitization and mode-badge preservation before returning responses to the Streamlit interface.

### 5. Enhanced Intent Classification

- Expanded keyword groups in `route_intent()` within `agent_graph.py` to robustly catch technical queries related to component shortages, supplier tiers, financial exposure, and GDS routing metrics without false negatives.

---

## 5 sample questions

Here are **5 sample questions** designed to validate every distinct feature, mode, and workflow branch of your multi-agent RAG and graph application:

---

### 1. Supply Chain Intelligence & Graph Analytics (`SUPPLY_CHAIN`)

- **Sample Prompt:**

  > _"What is our total revenue at risk and which customer orders are impacted by the port congestion at the Port of Los Angeles affecting our microcontrollers?"_

- **What it Validates:**
- Intent routing to `SUPPLY_CHAIN`.
- Dynamic Text-to-Cypher generation using fuzzy matching (`toLower(c.name) CONTAINS 'microcontroller'`).
- Multi-hop Neo4j graph traversal connecting Ports $\rightarrow$ Facilities $\rightarrow$ Products $\rightarrow$ Warehouses $\rightarrow$ Orders.
- Financial exposure calculation (returning impacted orders like `ORD-8801` and `ORD-8802` and aggregate dollar amounts).

---

### 2. Multi-Agent Commercial Real Estate Site Selection (`SITE_SELECTION`)

- **Sample Prompt:**

  > _"Evaluate the top commercial real estate and data center site options in Northern Virginia for a new high-density cloud infrastructure deployment."_

- **What it Validates:**
- Intent routing to `SITE_SELECTION`.
- Parameter extraction (`extract_site_parameters`) capturing country ("United States") and locations ("Northern Virginia").
- Hierarchical LangGraph supervisor and sub-agent execution evaluating power grids, fiber density, and tax incentives.
- Persistence of the generated markdown report to disk (`./reports/`).

---

### 3. Re-Ranked Vector RAG & Cross-Encoder Filtering (`RAG`)

- **Sample Prompt:**

  > _"What are the core architecture and multi-agent workflow features of the Laddu Bullodu project?"_

- **What it Validates:**
- Intent routing to `RAG` (via project keywords like _Laddu Bullodu_).
- Wide vector retrieval ($k=20$) from ChromaDB using `EmbeddingGemma`.
- BGE Cross-Encoder re-ranking (`BAAI/bge-reranker-large`) scoring and filtering down to the top 3 high-confidence chunks.
- Corrective RAG (CRAG) threshold evaluation ($\ge 0.4$ score triggering high-confidence local generation).

---

### 4. Live Web Intelligence Search (`WEB`)

- **Sample Prompt:**

  > _"What are the latest developments and news in artificial intelligence agents this week?"_

- **What it Validates:**
- Intent routing to `WEB` (triggered by recency keywords like _latest_).
- Integration with Google Serper.dev API (`GoogleSerperAPIWrapper`).
- Real-time web result ingestion into the system prompt for web-connected reasoning.

---

### 5. General LLM Reasoning & Guardrails (`LLM`)

- **Sample Prompt:**

  > _"Explain the architectural differences between SAML 2.0 and OIDC authentication flows in zero-trust enterprise networks."_

- **What it Validates:**
- Intent routing to `LLM` (fallback general reasoning mode).
- Direct execution via Ollama or Groq inference providers.
- Output PII guardrail node (`output_guardrail_node`) running sanitization filters before returning the response to Streamlit.

## Use Re-Ranking in a LangGraph Agentic System

In a LangGraph multi-agent or stateful workflow, re-ranking is not just a static step after retrieval—it is explicitly integrated into the graph's control flow, node definitions, and routing decisions.
You should use re-ranking in your LangGraph implementation during the following scenarios:

### 1. In a Pre-Generation Node (Filtering Noise before the LLM)

• The Scenario: Your agent handles complex, multi-turn user queries that require pulling information from a vector database (e.g., a Chroma or Pinecone instance).
• LangGraph Integration: You define a retrieval node (retrieve_docs) that fetches 20 raw chunks. The graph transitions to a rerank_docs node, which passes those 20 chunks through a re-ranking API (like Cohere, BGE-Reranker, or Jina). The state is updated to keep only the top 3 high-confidence documents before routing the payload to the generate_answer node. This prevents LLM "lost in the middle" phenomena and reduces token consumption.

### 2. Inside a Multi-Query or Query-Expansion Loop

• The Scenario: An upstream agent node takes a user query and breaks it down or expands it into 3 alternative variations to maximize search coverage.
• LangGraph Integration: If your graph runs 3 separate retrievals in parallel (or sequentially), you will accumulate a massive, overlapping list of documents. You use a re-ranking node to de-duplicate and prioritize the global results, selecting the absolute best context across all expanded queries before proceeding down the graph.

### 3. To Guard Conditional Edge Routing (Corrective RAG)

• The Scenario: You want your agent to dynamically decide whether it has enough good information to answer, or if it needs to look somewhere else (like a web search tool).
• LangGraph Integration: After the re-ranking node, you can read the absolute relevance scores returned by the re-ranker model. If the top document score falls below a specific threshold (e.g., a score of < 0.4), a conditional edge routes the graph away from generation and toward a web search node or a clarification node.

Procedural Work: How Re-Ranking Fits Into a LangGraph Workflow

The following visual workflow demonstrates how a re-ranking mechanism operates as a distinct architectural gate inside a stateful LangGraph cycle, filtering down broad vectors and driving conditional routing logic.

![alt text](images/image_rag_rr.png)

### 4. How Huggingface BGE works?

The moment your LangGraph workflow execution transitions into the re-ranking phase. It signals that your agent is initializing a local, open-source Cross-Encoder architecture (bge-reranker-large) developed by BAAI (Beijing Academy of Artificial Intelligence) to transform a broad list of potential answers into a highly precise, curated context pool.

#### 4.1. Architectural Context: What Happens in the rerank_node

In a LangGraph setup, your state graph routes the execution flow into a specific node (often named rerank_node). When this node triggers, it executes a Python function that loads (or accesses from memory) the heavy machine learning model to evaluate document relevance. Instead of passing dozens of raw chunks directly to your LLM generator node, this node serves as a strict mathematical quality filter.

![alt text](images/image_vdb_bge.png)

#### 4.2. Understanding the Model: bge-reranker-large

• The "BGE" Moniker: BGE stands for BAAI General Embedding. It is globally recognized on public leaderboards (like MTEB) for state-of-the-art retrieval accuracy.
• The "Large" Scale: This is a heavy, 560-million parameter model. While it requires real compute resources (VRAM/RAM), it is robust enough to catch complex context matches that smaller models ignore.
• The Output Score: Unlike vector distance metrics (which might just return arbitrary spatial numbers), a Cross-Encoder outputs a raw logit score or a standardized probability score (usually scaled between 0 and 1). This score answers a definitive question: "What is the exact probability that this specific document answers this specific query?"

#### 4.3. Why This Matters to Your System State

By initializing this model inside the rerank_node, your agent gains a massive architectural advantage:
• Drastic Noise Reduction: If your vector database pulls text containing overlapping keywords but completely wrong context, the BGE cross-encoder will slash its score down close to zero.
• Empowering the Next Edge: Because these scores are highly accurate reflections of real document utility, you can reliably use them inside your next LangGraph conditional edge router to make a programmatic choice: either proceed to generate an answer with confidence, or divert the state graph into a fallback web-search node because the documentation lacks the answer.

## BM25 in RAG Systems

BM25 (Best Matching 25) is a classic, non-neural keyword-based (lexical) search algorithm. While modern RAG systems rely heavily on dense vector embeddings (semantic search) to understand meaning, vector search has well-known blind spots.

BM25 is introduced into RAG architectures to bridge those exact gaps, forming what is known as Hybrid Search (combining BM25 lexical search with Dense Vector semantic search).

The Problem BM25 Solves in Pure Vector RAG
Dense embedding models (like EmbeddingGemma, OpenAI text-embedding-3, or bge-large-en) convert text into high-dimensional mathematical concepts. They excel at understanding synonyms and general context, but they often fail at:

Exact Matches & Out-of-Vocabulary (OOV) Terms: Vector models frequently struggle with exact product SKUs, serial numbers, part IDs, error codes, and proprietary jargon (e.g., matching exact strings like ORD-8802, A16-MCU-REV2, or ERR_SYSTEM_404).

Rare Proper Nouns & Names: Specific foreign names, technical terminology, or obscure acronyms may not have strong vector representation in the embedding model's pre-trained vocabulary.

Exact Keyphrase Precision: If a user searches for an exact phrase, vector embeddings might retrieve semantically "similar" documents that completely omit the specific keyword required.

### How BM25 Works (Key Principles)

BM25 evaluates document relevance based on exact token matching, enhanced by three key adjustments:

TF (Term Frequency) Saturation: Unlike simple keyword counting, BM25 caps the advantage of repeating a word multiple times in a document. The score increases rapidly for the first few occurrences of a keyword, then levels off.

IDF (Inverse Document Frequency): Rare words across your document collection (e.g., A16 Microcontroller or Dijkstra) receive significantly higher relevance weights than common words (e.g., system, component, data).

Document Length Normalization: Longer documents naturally contain more words. BM25 penalizes long documents so that a brief document mentioning the exact keyword isn't penalized compared to a 100-page manual containing the word by chance.

### How BM25 Improves Your RAG System

When added to a RAG pipeline alongside dense vector search and Cross-Encoder re-ranking, BM25 provides distinct advantages:

1. Delivers High Recall via Hybrid SearchBy combining candidate results from both BM25 ($k=20$) and Dense Embeddings ($k=20$), the system achieves maximum retrieval coverage:Vector Search handles: "Why is the chip delivery delayed?"BM25 Search handles: "Where is order ORD-8802 currently stored?"

2. Enhances Re-Ranker Efficiency
   A Cross-Encoder model (like BAAI/bge-reranker-large) is computationally heavy. If your initial retrieval fails to fetch the chunk containing the exact SKU or part number, the re-ranker cannot evaluate it. BM25 guarantees that exact-match candidate chunks make it into the top retrieval pool for the re-ranker to process.

3. Low Latency & Zero GPU Overhead
   BM25 relies on an inverted index (using libraries like rank_bm25 or bm25s in Python). It executes in milliseconds on standard CPU hardware without requiring GPU acceleration or embedding API calls.

## vLLM vs SGLang Core Difference

• vLLM is built for high-throughput enterprise production servers handling many independent users concurrently on dedicated GPUs.
• SGLang is optimized for complex agent workflows, multi-turn chat, and RAG pipelines where users repeatedly share the same system prompts or context.
• llama.cpp is the ultimate tool for resource-constrained consumer hardware, local offline development, and cross-platform flexibility (CPU, Mac, mobile).

### Architectural Highlights

1. vLLM: The Enterprise Workhorse

vLLM drastically increased serving efficiency through PagedAttention. In standard engines, the Keys and Values (KV) generated during long text generation must be stored in continuous blocks of GPU memory, leading to massive memory waste (internal fragmentation). vLLM breaks the KV cache into small blocks and pages them like an operating system's virtual memory. This unlocks up to 95% VRAM utilization and pairs with continuous batching to instantly handle new arrivals without waiting for an entire batch to finish processing.

2. SGLang: The Agent & RAG Accelerator

Built on top of concepts similar to vLLM, SGLang specializes in RadixAttention. Instead of wiping out the KV cache after a request completes, SGLang holds historical request prefixes in a persistent tree structure. On forums like r/StrixHalo, consensus highlights its strong performance with specific model architectures like DeepSeek and Qwen, particularly for tasks outputting highly structured JSON. If 50 users or an agent loops ask questions based on a fixed 100-page policy document, SGLang reuses that context natively rather than re-computing it every single turn.

## 💡 Neo4j: Native Graph Storage & Processing

Neo4j stores data as a network of nodes (entities) and relationships (connections).
Here is a breakdown of what makes Neo4j uniquely powerful:

In a traditional database, connecting data across multiple tables requires complex, resource-heavy SQL JOIN operations. As the dataset grows, these joins slow down exponentially.
• Index-Free Adjacency: Neo4j ensures each data node points directly to its physically adjacent neighbor nodes on the hard drive.
• Constant-Time Traversals: Because connections are stored natively, querying millions of complex connections takes milliseconds, regardless of how massive the total database gets.

### 🔎 Cypher Query Language

Neo4j created Cypher, a declarative graph query language that is highly visual and intuitive. It uses ASCII-art syntax to represent visual patterns.
• Example: To find friends of friends, instead of writing nested SQL joins, you write:
MATCH (user:User)-[:FRIEND]->(friend)-[:FRIEND]->(fof) RETURN fof

### 📊 Flexibility (Schemaless Model)

Neo4j uses the Property Graph Model. Every node and relationship can hold its own key-value pairs (properties). Because it is schemaless, you can dynamically add new properties, entities, or entire relationship types to a live database without running massive migration scripts or disrupting the existing schema.

### 🛠 Key Industry Use Cases

Because of this architecture, Neo4j excels at solving problems where the context and connections matter more than individual data points:
• Recommendation Engines: Powers real-time product or content recommendations by instantly connecting a user’s history, preferences, and friend activities (e.g., "People you may know" or "Customers who bought this also bought...").
• Fraud Detection: Uncovers sophisticated financial crime rings by tracking hidden links, shared bank accounts, or overlapping phone numbers across millions of transactions in real time.
• Knowledge Graphs / GenAI: Increasingly used to power Retrieval-Augmented Generation (RAG) architectures. It maps structured business knowledge so Large Language Models (LLMs) can retrieve highly accurate, contextual answers.
• Identity & Access Management: Maps complex enterprise hierarchies, permissions, and cloud infrastructure assets to evaluate access control paths instantly.

## About TypedDict

It is a type hint that allows you to specify the exact keys and value types expected in a Python dictionary. Unlike a standard dict[str, Any] which accepts any string key and any value, TypedDict provides structure for heterogeneous dictionaries—making it highly useful for handling JSON payloads, configuration objects, and structured API data. Because TypedDict is a typing-only construct, it runs with zero runtime overhead. At execution time, the object remains a regular Python dictionary.

### Why Use TypedDict?

1. Autocompletion & IDE Support: Editors like VS Code parse the definition to auto-suggest keys as you type (e.g., typing user[ brings up "username", "age", etc.).

2. Static Code Analysis: Tools like Pyright or Mypy flag an error if you assign an incorrect type or supply an unexpected key.

### Key Features & Constraints

1. Optional vs. Required Keys (NotRequired)
   By default, all fields in a TypedDict are required. If a key is missing, your type checker will throw an error. You can make individual fields optional using

2. Nested Structured Data
   You can nest TypedDict structures within one another to mirror complex JSON payloads:

3. Inheritance
   You can extend an existing TypedDict definition to inherit its parameters:

## LangGraph Streams

graph_builder.stream() runs your graph step-by-step and yields updates as they happen, instead of making you wait for the entire graph to finish executing.

### What each event contains

By default, each event printed in this loop is a dictionary that represents a snapshot of the graph's execution state at a specific step:
• The Key: The name of the node that just executed (e.g., {'chatbot': ...}).
• The Value: The data or state updates produced by that specific node (e.g., the messages or variables added to the state).

### Why use it?

• Real-Time Visibility: It allows you to monitor the internal flow of your agentic graph as it moves from node to node.
• Better UX: It serves as the foundation for streaming token outputs or intermediate steps directly to a user interface, preventing the application from feeling frozen during long LLM calls.
• If you are looking to stream a specific type of data, you can pass a mode argument like graph_builder.stream(inputs, mode="values") or mode="updates" to customize exactly what information each event contains.

### https://www.datacamp.com/tutorial/guide-to-autoresearch

## Prompt Versioning (config/prompts_v1.json)

The Benefits
Code and Logic Decoupling: System prompts, user instructions, and execution thresholds (such as the crag_relevance_threshold of 0.4) are extracted out of Python source files and housed in a clean JSON configuration file.

Zero-Downtime Tuning & Experimentation: You can refine prompt phrasing, adjust tuning parameters, or roll out new prompt variations (prompts_v2.json) without modifying a single line of Python code or restarting core graph architecture.

Centralized Governance: All prompt strings across diverse nodes (RAG generation, CRAG fallback, web search, data extraction) are managed in one predictable place.

### How to Validate Prompt Versioning

Open your configuration file at config/prompts_v1.json.

Modify a system prompt string (e.g., change "You are a helpful, professional AI assistant." to "You are a strict, highly technical enterprise engineering assistant.").

Send a RAG-based query in your Streamlit chat without restarting or modifying agent_graph.py or nodes.py.

Observe: The assistant's tone and formatting will immediately shift to adopt the newly configured prompt on the next execution turn.

markdown

# LangGraph Persistence & Checkpointing Architecture

LangGraph utilizes **Checkpointers** to provide thread-level, short-term memory. They automatically save snapshots of the graph state at every execution super-step, enabling conversational continuity, fault-tolerant error recovery, multi-turn state tracking, and state rewinding ("time travel").

---

## 💾 Short-Term Memory Providers

### 1. Development Implementation: `MemorySaver`

The `MemorySaver` (or `InMemorySaver`) is a **non-durable, in-memory checkpointer** engineered exclusively for prototyping, testing, and local execution pipelines.

- **Underlying Architecture:** Implements the core `BaseCheckpointSaver` interface using an internal, nested Python dictionary.
- **Indexing Structure:** Maps state histories hierarchically via:
  - `thread_id`: Distinctly segments separate user conversations or isolated processing streams.
  - `checkpoint_id`: Identifies the exact temporal sequence number or unique step version within a single thread.
- **Lifecycle:** Data resides strictly within volatile process RAM. **All state data is lost when the Python application terminates, crashes, or restarts.**

### 2. Production Implementation: External Databases

In production systems, memory operations are decoupled from volatile RAM and pushed to persistent data layers through specialized package wrappers (such as `langgraph-checkpoint-postgres` or `langgraph-checkpoint-sqlite`).

#### Basic Setup Example (PostgreSQL)

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/mydb"

# Establish database connection and automatically generate internal schemas
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # Provisions system tables safely

    # Compile the state graph with the persistent checkpointer layer
    graph = builder.compile(checkpointer=checkpointer)
```

---

## 🗄️ Database Schema & Data Lifecycle

When initializing an enterprise database checkpointer using `checkpointer.setup()`, LangGraph auto-provisions a standardized set of system tables:

| System Table Name           | Operational Role           | High-Level Contents                                                                                                                              |
| :-------------------------- | :------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| **`checkpoints`**           | **State Metadata Engine**  | Tracks structural lineage, linking `thread_id`, unique `checkpoint_id`, chronological timestamps, and parent sequence pointers for step history. |
| **`checkpoint_blobs`**      | **Serialized State Data**  | Stores actual variables, execution parameters, and message arrays, packed into raw **binary blobs** via serialization channels.                  |
| **`checkpoint_writes`**     | **Task Operations Log**    | Manages ephemeral task tokens, intermediate node writes, and state change deltas introduced between discrete execution ticks.                    |
| **`checkpoint_migrations`** | **Schema Version Control** | Logs library schema versions to run schema upgrades across underlying package updates safely.                                                    |

### The Execution State Loop

1. **Hydration Phase:** When the graph receives an execution payload matched to a specific thread context (`config = {"configurable": {"thread_id": "session_abc"}}`), the checkpointer queries the `checkpoints` and `checkpoint_blobs` tables to fetch and deserialize the latest state payload back into active Python objects.
2. **Execution Phase:** Computational graph nodes modify state variables or evaluate business rules based on the hydrated parameters.
3. **Dehydration Phase:** Once a super-step evaluates cleanly, LangGraph serializes the updated variables and appends a fresh row to the database, mapping it directly back to the preceding checkpoint record as its absolute parent.

---

## 🚀 Supported Backend Integrations

LangGraph isolates driver boundaries into pluggable micro-packages to ensure infrastructural portability:

- **PostgreSQL (`PostgresSaver` / `AsyncPostgresSaver`):** Robust, enterprise-grade storage supporting concurrent connection pools and async execution cycles.
- **SQLite (`SqliteSaver` / `AsyncSqliteSaver`):** Lightweight, zero-configuration local file system storage suitable for single-node runtimes or embedded architectures.
- **MongoDB (`MongoDBSaver`):** Document-oriented store mapping internal graph checkpoints directly into native BSON collections.
- **Redis (`RedisSaver`):** High-throughput, ultra-low latency memory store utilizing background disk persistence parameters.

---

## ⚠️ Architectural Guardrails

- **Short-Term Context Limitation:** Checkpointers record chronological, sequential execution histories. Appending messages continuously to thread states will eventually collapse downstream LLM context windows.
- **Long-Term State Management:** For long-term memory metrics (e.g., cross-session profile fields or global configuration traits), decouple state tracking away from short-term checkpointers and leverage **LangGraph Stores** (`InMemoryStore`, `PostgresStore`).
  Use code with caution.
  Let me know if you would like me to add code configurations for async data layers (AsyncPostgresSaver), Docker Compose config setups, or an example section demonstrating how to write state recovery / time travel code inside your README.

## LangGraph MemorySaver Checkpointer

The Benefits
Short-Term Conversational State Persistence: LangGraph checkpoints every graph super-step against a unique session thread_id. This ensures conversation state, intermediate node outputs, and multi-turn message history remain synchronized.

Fault Tolerance & Resilience: If an execution node encounters a transient error midway through a graph traversal, the checkpointer preserves the state leading up to that point.

Enterprise-Grade Stateful Architecture: Fulfills production agent requirements by allowing asynchronous state inspection, conversation resumption, and multi-turn context retention across user prompts.

### How to Validate MemorySaver

Run your Streamlit application (streamlit run app.py) and initiate a conversation with a multi-turn context:

Turn 1: "What logistics details are tied to order ORD-8802?"

Turn 2: "What component was listed in that order?" (referencing the previous turn without repeating the order ID).

Observe: The router_agent and retrieve_node will successfully resolve the context because MemorySaver maintains the conversation history state across turns via the session thread_id.

Code Inspection Check: You can verify that checkpoints are actively recording by inspecting the graph's state history in your terminal or debugger using app_graph.get_state(thread_config).

### Upgrade to SQLite

To replace the volatile MemorySaver (in-memory storage) with a persistent, database-backed solution that survives application restarts, SQLite is the ideal open database choice. It requires zero server setup, runs natively in Python via built-in libraries, and integrates seamlessly with LangGraph via the official langgraph-checkpoint-sqlite package.

### How SQLite Works & How to Validate

Persistent Storage File: When you run your application, a local file named checkpoints.sqlite will be automatically generated in your project root directory. This file stores all graph execution states, thread checkpoints, and historical message turns.

Restart Resilience Validation:

Open your Streamlit app (streamlit run app.py).

Ask a multi-turn conversational question (e.g., “What logistics details are tied to order ORD-8802?” followed by a follow-up referencing it).

Stop your Streamlit server (Ctrl + C) and completely shut down the Python process.

Restart Streamlit.

Observe: Because the conversation history is bound to the session's database-backed thread ID, the context survives the server restart cleanly.

### Method 2: The Server Restart & State Persistence Test

Because SqliteSaver writes data directly to disk (checkpoints.sqlite), conversation states survive application restarts.

Start Streamlit: Run streamlit run app.py.

Chat: Send a question to the assistant (e.g., "What logistics details are tied to order ORD-8802?").

Kill the Server: Go to your terminal and press Ctrl + C to completely terminate the Streamlit process and Python backend.

Restart: Run streamlit run app.py again.

Verify: Notice how the database file preserves your session history and execution context, allowing the checkpointer to resume seamlessly without losing state.

When deciding between **`SqliteSaver`** (paired with a LangGraph application) and **Mem0** for your assistant, it helps to understand a fundamental architectural truth: **They solve completely different problems, and they are not mutually exclusive.**

Comparing them directly is like comparing a **car's transmission (SqliteSaver)** to its **GPS memory profile of favorite destinations (Mem0)**.

### Why This Solves the Problem

True Fault Tolerance: If a tool times out or the server crashes mid-flight during heavy sub-graphs like Site Selection or Supply Chain Analysis, the intermediate evaluations and raw data collected up to that second remain securely saved in ./checkpoints.sqlite.

Automatic Pick-Up: Upon restarting the server, the application detects the pending graph step (current_state.next), invokes execution with None, and seamlessly finishes processing the remaining locations without starting over from scratch.

---

## SQLite vs mem0

### 1. What They Actually Do

- **`SqliteSaver` (The Execution Checkpointer):**
- **Core Job:** Saves the raw operational state of your LangGraph workflow (messages, active node variables, routing states, and thread data) so your agent can pause, resume, survive server restarts, and handle multi-step flows.
- **Scope:** Thread-specific and session-bound. It remembers _everything_ raw said in a specific chat thread, but it doesn't intelligently summarize facts or synthesize a user's cross-session profile.

- **Mem0 (The Semantic Long-Term Memory Layer):**
- **Core Job:** Acts as an intelligent memory management service. It uses LLMs behind the scenes to actively extract atomic facts, preferences, and entities from conversations (e.g., extracting _"Loves vegetarian South Indian food"_ or _"Prefers data centers in Montreal"_ from a chat) and stores them in a vector or graph database.
- **Scope:** User-specific and cross-session. If a user starts a brand-new chat thread next week, Mem0 can inject their historical preferences into the prompt, whereas a clean thread using only `SqliteSaver` would know nothing about past chats.

---

### 2. Direct Feature Breakdown

| Feature                  | `SqliteSaver`                                                              | Mem0                                                                      |
| ------------------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Primary Purpose**      | Graph execution state and fault tolerance                                  | Semantic long-term user profiling and fact extraction                     |
| **What it stores**       | Raw message history and workflow variables                                 | Compact, distilled user facts and preferences                             |
| **Cross-Session Recall** | ❌ No (tied strictly to a single `thread_id`)                              | ✅ Yes (retrieves facts across any thread via `user_id`)                  |
| **Infrastructure**       | A single local file (`checkpoints.sqlite`) with zero external dependencies | Requires embedding generation and vector/graph indexing (or Mem0 Cloud)   |
| **Cost / Tokens**        | Free (pure local state serialization)                                      | Incurs extra LLM calls for background fact extraction and vector searches |

---

### 3. Which Is Best for Your Application?

Given that your application is an advanced, production-grade agentic system featuring multi-agent workflows, domain-specific execution (Neo4j supply chain, multi-regional site selection), and OpenTelemetry observability, here is how you should choose:

#### **Scenario A: Stick with `SqliteSaver` (Plus LangGraph's native Store) if:**

- You only need conversational continuity **within the same chat thread** (i.e., users resume a specific chat and want the agent to remember where they left off in that session).
- You want to keep your architecture lean, self-hosted, and running entirely on local SQLite files without adding external vector database overhead or extra background LLM calls for fact extraction.
- _Note:_ LangGraph features a native state persistence layer (`Store`) if you want custom long-term cross-session memory without third-party libraries.

#### **Scenario B: Integrate Mem0 if:**

- You want **true cross-thread personalization**. For instance, if a user specifies their regional constraints, tech stack preferences, or enterprise profile in Chat A, you want the agent to automatically remember those preferences next month when they open Chat B under a new `thread_id`.
- You want automated LLM-driven background extraction that distills thousands of lines of chat history into clean, minimal, token-efficient user profiles rather than bloating your context window with old chat logs.

---

### **The Recommended Production Architecture**

In mature enterprise setups, **you use both together**:

1. **`SqliteSaver`** handles the low-level engine execution—keeping your LangGraph state safe, handling node transitions, supporting time-travel debugging, and ensuring active tool calls don't drop if a worker restarts.
2. **Mem0** (or a similar semantic memory layer) sits alongside the graph as a middleware tool, injecting persistent user preferences into your system prompt at the start of a run.

## OpenTelemetry (OTel) is the industry-standard approach for modern agentic applications.

Your current custom setup (`metrics.sqlite` + manual timing) works well as a lightweight prototype, but as your system scales with multi-turn agents, nested sub-graphs (Neo4j, Site Selection), and multiple LLM vendors (Groq, Ollama), managing observability manually becomes fragile and difficult to maintain.

---

## **Why OpenTelemetry is Superior to Custom Telemetry**

### **1. Tracing Nested & Asynchronous Graph Execution**

- **Bits & Pieces Problem:** Manual timing in `router_node` only measures the overall turn-level latency. It doesn't show whether a delay occurred during ChromaDB hybrid vector search, BGE cross-encoder re-ranking, or Neo4j Cypher querying.
- **OpenTelemetry Solution:** OTel introduces **Spans and Traces**. Every LangGraph node and external API call becomes a child span under a single trace ID, giving you an accurate timeline waterfall for every request.

### **2. Zero-Boilerplate Instrumentation**

- **Bits & Pieces Problem:** Manually adding timestamps, estimating token counts, and catching exceptions inside `output_guardrail_node` litters your core graph logic with observability code.
- **OpenTelemetry Solution:** OpenTelemetry integrates directly with LangChain and LangGraph via standard SDKs (e.g., **OpenInference** / **LangSmith** / **Phoenix / OpenTelemetry LangChain Instrumentation**). It automatically hooks into model calls to capture precise token counts, latencies, inputs, and outputs without custom wrapper functions.

### **3. Vendor-Agnostic Telemetry Export**

- **Bits & Pieces Problem:** Storing telemetry in `metrics.sqlite` locks your observability metrics into a local file that is difficult to query, visualize, or scale across multiple instances.
- **OpenTelemetry Solution:** OTel standardizes your logs, metrics, and traces into standard formats (OTLP). You can export your data to open-source UI platforms like **Arize Phoenix**, **Jaeger**, **Grafana/Prometheus**, or enterprise platforms like **LangSmith**, **Datadog**, or **Honeycomb** simply by changing an environment variable.

---

### **Comparison: Current Custom Setup vs. OpenTelemetry**

| Feature              | Current Bits & Pieces (`metrics.sqlite`)           | OpenTelemetry Standard                           |
| -------------------- | -------------------------------------------------- | ------------------------------------------------ |
| **Setup Complexity** | Low (custom Python code)                           | Low to Medium (install SDK)                      |
| **Granularity**      | Turn-level latency only                            | Full trace waterfall (node-by-node)              |
| **Token Tracking**   | Manual character/heuristic approximation           | Exact LLM token usage from API response metadata |
| **Maintenance**      | High (must update custom code when schema changes) | Zero (handled by OTel instrumentors)             |
| **Visualization**    | Custom Streamlit widgets                           | Built-in dashboards, flamegraphs, & metrics      |
| **Scalability**      | Single local SQLite file                           | Distributed, enterprise-grade storage            |

---

### **How OpenTelemetry Integrates with LangGraph**

In a production LangGraph setup, OpenTelemetry operates as a non-intrusive background collector:

```text
[Streamlit UI] ──► [LangGraph Workflow]
                        │
                        ├─► (Auto-instrumented Spans: Retrieve, Rerank, LLM)
                        │
                        ▼
            [OpenTelemetry Collector / Phoenix / Jaeger]
                        │
                        ▼
       [Dashboard: Latency Waterfalls, Exact Tokens, P95, Failures]

```

---

### **Recommended Migration Strategy**

1. **Keep `metrics.sqlite` temporarily** to power your current Streamlit sidebar without breaking existing UI features.
2. **Add OTel auto-instrumentation** (such as `openinference-instrumentation-langchain` or **Arize Phoenix**) by initializing the tracer at app startup.
3. **Deprecate manual calculation blocks** in `output_guardrail_node` once your OTel dashboard collector is active.

#### How to Validate Your New Observability Stack

Launch the App: Run your Streamlit application:

Bash
streamlit run app.py
Check Terminal Output: Look for the Phoenix dashboard link printed in your terminal (typically http://localhost:6006).

Execute Chat Queries: Send a few prompts through the chat interface (e.g., asking about hybrid search docs or supply chain orders).

Inspect Traces in Phoenix UI: Open http://localhost:6006 in your browser. You will see:

Trace Waterfalls: Exact breakdown of how long router_agent, retrieve_node, rerank_node, and LLM generation took.

Token Counts & Costs: Automated metrics for prompt tokens, completion tokens, and estimated cost per model tier.

p50 / p95 Latencies & Failure Rates: Aggregated production telemetry calculated automatically across all sessions.

## OpenAPI Spec (OAS) vs Model Context Protocol (MCP)

OpenAPI is a blueprint for humans to connect apps, while MCP is a protocol for AI models to connect tools.
Instead of competing, they operate at different layers of the software stack: MCP acts as an orchestration layer that often wraps around existing OpenAPI endpoints, translating rigid web APIs into a dynamic format that Large Language Models (LLMs) can autonomously understand and navigate. [1] (https://www.truefoundry.com/blog/mcp-vs-api), [2] (https://daily.dev/posts/mcp-vs-api-explained-do-you-really-need-mcp--6yi4vkilk)

![alt text](image-3.png)

### Architectural Shift: How They Work

1. OpenAPI: Built for Deterministic Software

When you expose a service via OpenAPI, you define fixed endpoints (e.g., GET /users/{id}) and strict payloads. A human software engineer writes explicit code defining exactly when that endpoint is triggered, how authentication headers are attached, and how errors are handled. If the backend changes, the application code must be manually rewritten to adapt. [1] (https://www.scalekit.com/blog/mcp-vs-apis-how-are-they-different), [2] (https://www.youtube.com/watch?v=185XGEMefgc&t=4), [3] (https://www.youtube.com/watch?v=UYC0YLk1SJ8&vl=en&t=34), [4] (https://daily.dev/posts/mcp-vs-api-explained-do-you-really-need-mcp--6yi4vkilk)

2. MCP: Built for Non-Deterministic Agents

Introduced by Anthropic, MCP shifts the invocation logic entirely into the model's reasoning layer. When an MCP-compliant agent connects to an MCP server, the first flight is a dynamic handshake: "What can you do?" The server responds with a runtime-readable map of available tools, prompts, and resources described via JSON schemas. [1] (https://www.andreagemelli.me/posts/mcp/), [2] (https://www.youtube.com/watch?v=185XGEMefgc&t=4), [3] (https://www.youtube.com/watch?v=gDkXFL9pAQw), [4] (https://circleci.com/blog/mcp-vs-api/), [5] (https://www.dash0.com/knowledge/mcp-vs-api)
The LLM evaluates this list in real-time, matching its capabilities against the user's natural language request to decide which tool to fire. [1] (https://circleci.com/blog/mcp-vs-api/), [2] (https://www.youtube.com/watch?v=UYC0YLk1SJ8&vl=en&t=34), [3] (https://www.youtube.com/watch?v=gDkXFL9pAQw)

### Why MCP Matters (The M × N Problem)

If you have 5 different developer tools (like VS Code, Cursor, and internal LLM apps) and 10 different backend services (like GitHub, Jira, and Slack), traditional API architectures require you to write 50 separate custom integrations (5 × 10). [1] (https://daily.dev/posts/mcp-vs-api-explained-do-you-really-need-mcp--6yi4vkilk)
MCP reduces this completely: you write 5 MCP Clients and 10 MCP Servers (5 + 10 = 15 components total). Any MCP-enabled host application can instantly pull context from and execute actions on any MCP-enabled server without custom glue code. [1] (https://www.youtube.com/watch?v=185XGEMefgc&t=4), [2] (https://daily.dev/posts/mcp-vs-api-explained-do-you-really-need-mcp--6yi4vkilk)

### Coexistence in Production

In an enterprise environment, you don't delete your OpenAPI specs. Instead, you keep your OpenAPI definition as the secure source of truth for your HTTP endpoints. You then deploy an MCP Gateway or middleware server that ingests that OpenAPI spec, translates its schemas into an MCP format, and exposes it safely to your agentic loops.

# LLM Evaluation Framework Comparison: Ragas vs. DeepEval

Both **Ragas** and **DeepEval** are leading open-source frameworks used to evaluate Large Language Model (LLM) and Retrieval-Augmented Generation (RAG) systems, though they cater to different workflows and design philosophies.

## Core Focus & Use Case

- **Ragas:** Purpose-built specifically for evaluating RAG pipelines. It originated as a research-backed framework focused heavily on academic and dataset-first metrics for retrieval and generation quality.
- **DeepEval:** Built as a comprehensive, developer-centric testing framework by Confident AI. It treats LLM evaluation like traditional software unit testing, covering general LLM apps, chatbots, agents, and RAG pipelines.

---

## Key Differences

| Feature / Dimension          | Ragas                                                                                  | DeepEval                                                                                                                           |
| :--------------------------- | :------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **Primary Scope**            | Specialized RAG evaluation (retrieval + generation)                                    | Broad LLM evaluation (RAG, general LLMs, safety, bias, toxicity, agents)                                                           |
| **Workflow Style**           | Dataset-and-pipeline evaluation (`evaluate()`)                                         | Pytest-native assertions (`deepeval test run`)                                                                                     |
| **CI/CD Integration**        | Runs evaluations, needs custom wrapper code for pass/fail gates                        | Native pass/fail blocking in CI/CD pipelines (fails builds on threshold drop)                                                      |
| **Debugging & Explanations** | Numerical/aggregate scores; can sometimes return `NaN` on invalid JSON from judge LLMs | Built-in score reasoning, JSON-confinement, and metric-level debugging                                                             |
| **Ecosystem & Interop**      | Operates standalone or integrates via platforms like MLflow                            | Integrates with Confident AI for dashboards and allows importing Ragas metrics directly (`from deepeval.metrics.ragas import ...`) |

---

## Metrics Compared

Both frameworks use an **LLM-as-a-judge** approach and share core RAG metrics, though their underlying evaluation mechanics differ:

- **Faithfulness (Grounding):**
  - _Ragas_ uses strict logical entailment via claim-to-context ratios.
  - _DeepEval_ allows for flexible LLM reasoning that accounts for reasonable external or pragmatic facts.
- **Response / Answer Relevancy:**
  - _Ragas_ employs hypothetical question generation and embedding similarity.
  - _DeepEval_ evaluates statement-level relevance through direct LLM reasoning.
- **Retrieval Metrics:** Both support context precision and context recall, though DeepEval allows you to run Ragas metrics natively inside its own pipeline with added caching and debugging.

---

## Decision Matrix

- **Choose Ragas if:** You are focused tightly on a document-heavy RAG architecture, prefer a lightweight, dataset-centric approach for exploration, or want to analyze retrieval-specific performance over time on dashboards.
- **Choose DeepEval if:** You want developer-grade unit testing integrated directly into `pytest`, need hard quality gates to block broken pull requests in CI/CD, or need to test broader behaviors like safety, bias, toxicity, and multi-agent workflows.
- **Hybrid Approach:** Many engineering teams **use both together**—leveraging Ragas for deep RAG diagnostic analytics and DeepEval for strict regression testing and CI gates.
