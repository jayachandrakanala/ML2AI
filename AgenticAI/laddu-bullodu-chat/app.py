# app.py
import os
import logging
import warnings
import uuid

# Keep CPU-backed ML models out of Metal/PyTorch worker crashes on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Keep third-party startup diagnostics out of the application terminal.
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore", category=Warning, module="sqlalchemy")
warnings.filterwarnings("ignore", category=Warning, module="pydantic")
warnings.filterwarnings("ignore", message=".*unsupported reflection.*")
warnings.filterwarnings("ignore", message=".*not be serialized.*")
logging.getLogger("torch").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.telemetry import init_telemetry
init_telemetry()

from src.agent_graph import app_graph, set_status_callback, get_latest_thread_state, stream_graph, invoke_graph
from src.llm_provider import GROQ_MODEL, MODEL_LABELS, OLLAMA_MODEL

st.set_page_config(
    page_title="Laddu Bullodu RAG Assistant",
    page_icon="🟠",
    layout="centered"
)

st.header("✨ Laddu🟠Bullodu👦RAG Multi-Agent Chat ✨")
st.markdown("Powered by Python, Streamlit, LangGraph, ChromaDB, EmbeddingGemma, Serper.dev, & OpenTelemetry")

with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Observability Dashboard")
    
    from src.metrics_logger import get_metrics_summary
    metrics = get_metrics_summary()
    
    col1, col2 = st.columns(2)
    col1.metric("p50 Latency", f"{metrics['p50_latency']}s")
    col2.metric("p95 Latency", f"{metrics['p95_latency']}s")
    
    col3, col4 = st.columns(2)
    col3.metric("Total Cost", f"${metrics['total_cost']}")
    col4.metric("Failure Rate", f"{metrics['failure_rate']}%")

with st.sidebar:
    st.subheader("Chat model")
    selected_model = st.selectbox(
        "Choose a model",
        options=list(MODEL_LABELS),
        format_func=lambda model: MODEL_LABELS[model],
        index=list(MODEL_LABELS).index(GROQ_MODEL),
        key="selected_model",
    )
    st.caption("Groq and OpenAI require their API keys in your .env file.")

# ------------------------------------------------------------------------------
# 1. Thread Persistence & Session Setup
# ------------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    # Browser tabs have independent Streamlit session state; never share the default checkpoint.
    st.session_state.thread_id = f"browser-{uuid.uuid4().hex}"

thread_config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your laddu bullodu assistant. Ask me about local docs, web search news, or request commercial real estate site selection reports!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------------------------------------------------------------------
# 2. Checkpoint Auto-Resumption Logic on Startup
# ------------------------------------------------------------------------------
current_checkpoint = get_latest_thread_state(st.session_state.thread_id)

if current_checkpoint and current_checkpoint.next:
    st.info("⚡ Detected interrupted workflow execution. Automatically resuming from last saved SQLite checkpoint...")
    
    ai_response = ""
    with st.chat_message("assistant"):
        with st.status("Resuming agentic workflow from checkpoint...", expanded=True) as status:
            def update_status(msg: str):
                status.update(label=msg)
                st.write(f"• {msg}")

            set_status_callback(update_status)
            try:
                for event in stream_graph(None, config=thread_config, stream_mode="updates"):
                    for node_name, state_update in event.items():
                        update_status(f"Resuming step: `{node_name}`...")
                        if "messages" in state_update and state_update["messages"]:
                            latest_msg = state_update["messages"][-1]
                            ai_response = getattr(latest_msg, "content", str(latest_msg))

                status.update(label="Workflow successfully resumed and completed!", state="complete", expanded=False)
            except Exception as e:
                ai_response = f"Error during workflow resumption: {str(e)}"
                status.update(label="Resumption failed!", state="error", expanded=True)

        if ai_response:
            st.markdown(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
else:
    pass

# ------------------------------------------------------------------------------
# 3. Handle New User Input
# ------------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question or request a site selection report..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    graph_messages = []
    for msg in st.session_state.messages[-8:]:
        if msg["role"] == "user":
            graph_messages.append(("user", msg["content"]))
        elif msg["role"] == "assistant":
            graph_messages.append(("assistant", msg["content"]))

    ai_response = ""
    with st.chat_message("assistant"):
        with st.status("Initializing agentic workflow...", expanded=True) as status:
            def update_status(msg: str):
                status.update(label=msg)
                st.write(f"• {msg}")

            set_status_callback(update_status)

            try:
                state_input = {
                    "messages": graph_messages,
                    "selected_model": selected_model or OLLAMA_MODEL,
                }
                
                node_descriptions = {
                    "router_agent": "Analyzing user intent and selecting execution path...",
                    "retrieve_node": "Executing Hybrid Retrieval (Dense Vector + BM25) (k=20)...",
                    "rerank_node": "Scoring and filtering documents using BAAI BGE Cross-Encoder...",
                    "crag_fallback_node": "Corrective RAG triggered: executing web search...",
                    "domain_executor_node": "Executing specialized domain intelligence (Neo4j / Site Selection / Web)...",
                    "rag_generate_node": "Synthesizing final response...",
                    "guardrail": "Validating output against PII and security guardrails..."
                }

                for event in stream_graph(state_input, config=thread_config, stream_mode="updates"):
                    for node_name, state_update in event.items():
                        desc = node_descriptions.get(node_name, f"Executing node: `{node_name}`")
                        update_status(desc)
                        if "messages" in state_update and state_update["messages"]:
                            latest_msg = state_update["messages"][-1]
                            if hasattr(latest_msg, "content"):
                                ai_response = latest_msg.content
                            elif isinstance(latest_msg, dict):
                                ai_response = latest_msg.get("content", "")

                if not ai_response:
                    final_result = invoke_graph(state_input, config=thread_config)
                    ai_response = final_result["messages"][-1].content

                status.update(label="Workflow completed successfully!", state="complete", expanded=False)
            except Exception as e:
                ai_response = f"Error processing request: {str(e)}"
                status.update(label="Workflow failed!", state="error", expanded=True)

        st.markdown(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})