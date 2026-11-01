# src/agent_graph.py
import sqlite3
import warnings
import os

# Suppress noisy terminal warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="sqlalchemy")

from typing import Annotated, NotRequired, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config_loader import load_version_config
from src.logger import set_status_callback
from src.graph_nodes import (
    router_node,
    retrieve_node,
    rerank_node,
    domain_executor_node,
    rag_generate_node,
    crag_fallback_node,
    output_guardrail_node
)

CONFIG = load_version_config()
THRESHOLDS = CONFIG["thresholds"]

def add_bounded_messages(existing: list, new_messages: list) -> list:
    """Merge new messages, then retain only the recent context window."""
    merged = add_messages(existing, new_messages)
    return merged[-12:]

class State(TypedDict):
    """Conversation state passed between nodes in the assistant graph."""
    messages: Annotated[list, add_bounded_messages]
    selected_model: NotRequired[str]
    intent: NotRequired[str]
    raw_docs: NotRequired[list]
    top_docs: NotRequired[list]
    top_relevance_score: NotRequired[float]
    start_time: NotRequired[float]

def route_after_router(state: State):
    """Directs graph flow based on classified intent."""
    return "retrieve_node" if state.get("intent", "LLM") == "RAG" else "domain_executor_node"

def route_after_rerank(state: State):
    """Corrective RAG conditional edge based on re-ranker threshold."""
    return "crag_fallback_node" if state.get("top_relevance_score", 0.0) < THRESHOLDS.get("crag_relevance_threshold", 0.4) else "rag_generate_node"

# Assemble the StateGraph Workflow
workflow = StateGraph(State)

workflow.add_node("router_agent", router_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("rerank_node", rerank_node)
workflow.add_node("domain_executor_node", domain_executor_node)
workflow.add_node("rag_generate_node", rag_generate_node)
workflow.add_node("crag_fallback_node", crag_fallback_node)
workflow.add_node("guardrail", output_guardrail_node)

workflow.add_edge(START, "router_agent")
workflow.add_conditional_edges("router_agent", route_after_router, {"retrieve_node": "retrieve_node", "domain_executor_node": "domain_executor_node"})
workflow.add_edge("retrieve_node", "rerank_node")
workflow.add_conditional_edges("rerank_node", route_after_rerank, {"crag_fallback_node": "crag_fallback_node", "rag_generate_node": "rag_generate_node"})

workflow.add_edge("domain_executor_node", "guardrail")
workflow.add_edge("rag_generate_node", "guardrail")
workflow.add_edge("crag_fallback_node", "guardrail")
workflow.add_edge("guardrail", END)

# ------------------------------------------------------------------------------
# Persistent SQLite Database Checkpointer Setup
# ------------------------------------------------------------------------------
DB_PATH = "./checkpoints.sqlite"
db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
db_conn.execute("PRAGMA journal_mode=WAL")
db_conn.execute("PRAGMA busy_timeout=30000")
sqlite_checkpointer = SqliteSaver(db_conn)

# Keep a default graph for tests and non-browser callers.
app_graph = workflow.compile(checkpointer=sqlite_checkpointer)


def stream_graph(state, config, stream_mode="updates"):
    """Stream one session with a connection owned by this execution thread."""
    with SqliteSaver.from_conn_string(DB_PATH) as session_checkpointer:
        session_checkpointer.conn.execute("PRAGMA busy_timeout=30000")
        graph = workflow.compile(checkpointer=session_checkpointer)
        yield from graph.stream(state, config=config, stream_mode=stream_mode)


def invoke_graph(state, config):
    """Invoke one session with thread-local SQLite checkpoint ownership."""
    with SqliteSaver.from_conn_string(DB_PATH) as session_checkpointer:
        session_checkpointer.conn.execute("PRAGMA busy_timeout=30000")
        graph = workflow.compile(checkpointer=session_checkpointer)
        return graph.invoke(state, config=config)


def get_latest_thread_state(thread_id: str):
    """Read one browser session's checkpoint using an isolated SQLite connection."""
    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string(DB_PATH) as session_checkpointer:
        session_checkpointer.conn.execute("PRAGMA busy_timeout=30000")
        graph = workflow.compile(checkpointer=session_checkpointer)
        return graph.get_state(config)