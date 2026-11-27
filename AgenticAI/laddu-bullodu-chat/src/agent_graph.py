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
from src.llmops import crag_fallback_required, set_run_attribute, workflow_span
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
from src.image_store import is_visual_query

CONFIG = load_version_config()
THRESHOLDS = CONFIG["thresholds"]

# Function: add_bounded_messages
# Purpose: Merge LangGraph message updates and retain the latest 12 conversation entries. This function is the
# reducer for ``State.messages``; LangGraph calls it when nodes return message updates, so the bound
# applies across graph transitions.
def add_bounded_messages(existing: list, new_messages: list) -> list:
    """Merge LangGraph message updates and retain the latest 12 conversation entries.

    This function is the reducer for ``State.messages``; LangGraph calls it when
    nodes return message updates, so the bound applies across graph transitions.
    """
    merged = add_messages(existing, new_messages)
    return merged[-12:]

# Class: State
# Purpose: Shared data contract for the assistant graph's nodes. ``messages`` is accumulated through
# ``add_bounded_messages``. Other keys are optional because only nodes on the selected intent path
# populate them.
class State(TypedDict):
    """Shared data contract for the assistant graph's nodes.

    ``messages`` is accumulated through ``add_bounded_messages``. Other keys are
    optional because only nodes on the selected intent path populate them.
    """
    messages: Annotated[list, add_bounded_messages]
    selected_model: NotRequired[str]
    intent: NotRequired[str]
    raw_docs: NotRequired[list]
    top_docs: NotRequired[list]
    image_docs: NotRequired[list]
    retrieved_images: NotRequired[list]
    top_relevance_score: NotRequired[float]
    start_time: NotRequired[float]
    thread_id: NotRequired[str]
    trace_id: NotRequired[str]
    workflow_mode: NotRequired[str]
    prompt_version: NotRequired[str]
    graph_topology: NotRequired[str]

# Function: route_after_router
# Purpose: Send local-document intents to retrieval and all other intents to domain handling.
def route_after_router(state: State):
    """Send local-document intents to retrieval and all other intents to domain handling."""
    return "retrieve_node" if state.get("intent", "LLM") == "RAG" else "domain_executor_node"

# Function: route_after_rerank
# Purpose: Choose local RAG generation or web fallback using the configured CRAG score threshold. A missing
# relevance score is treated as zero, which routes to the fallback.
def route_after_rerank(state: State):
    """Choose local RAG generation or web fallback using the configured CRAG score threshold.

    Local image candidates satisfy explicit visual requests independently of the
    text reranker score. Other requests use the configured CRAG threshold.
    """
    latest_query = state["messages"][-1].content
    if is_visual_query(latest_query) and state.get("retrieved_images"):
        return "rag_generate_node"
    return (
        "crag_fallback_node"
        if crag_fallback_required(
            state.get("top_relevance_score", 0.0),
            THRESHOLDS.get("crag_relevance_threshold", 0.4),
        )
        else "rag_generate_node"
    )

# Assemble the StateGraph Workflow
# The graph has one START entry, a router, specialist retrieval/domain paths,
# RAG or CRAG generation, and a shared guardrail that leads to END.
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
# Checkpoint writes allow an interrupted browser session to resume from the
# next incomplete LangGraph node instead of restarting the full conversation.
DB_PATH = "./checkpoints.sqlite"
db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
db_conn.execute("PRAGMA journal_mode=WAL")
db_conn.execute("PRAGMA busy_timeout=30000")
sqlite_checkpointer = SqliteSaver(db_conn)

# Keep a default graph for tests and non-browser callers.
app_graph = workflow.compile(checkpointer=sqlite_checkpointer)


# Function: stream_graph
# Purpose: Yield LangGraph updates while owning a short-lived SQLite checkpointer connection. A fresh
# connection is opened for each invocation to avoid sharing SQLite connections across Streamlit
# execution threads. ``state`` may be ``None`` when resuming an existing checkpoint; ``config``
# supplies the thread ID.
def stream_graph(state, config, stream_mode="updates"):
    """Yield LangGraph updates while owning a short-lived SQLite checkpointer connection.

    A fresh connection is opened for each invocation to avoid sharing SQLite
    connections across Streamlit execution threads. ``state`` may be ``None``
    when resuming an existing checkpoint; ``config`` supplies the thread ID.
    """
    with workflow_span(state, config, "default"):
        with SqliteSaver.from_conn_string(DB_PATH) as session_checkpointer:
            session_checkpointer.conn.execute("PRAGMA busy_timeout=30000")
            graph = workflow.compile(checkpointer=session_checkpointer)
            for event in graph.stream(state, config=config, stream_mode=stream_mode):
                for update in event.values():
                    if "intent" in update:
                        set_run_attribute("intent", update["intent"])
                    if "top_relevance_score" in update:
                        set_run_attribute(
                            "rag.reranker_score", update["top_relevance_score"]
                        )
                yield event


# Function: invoke_graph
# Purpose: Run the graph to completion with a short-lived, thread-owned checkpoint connection. The returned
# value is the final graph state; ``config`` identifies the checkpoint thread to continue or create.
def invoke_graph(state, config):
    """Run the graph to completion with a short-lived, thread-owned checkpoint connection.

    The returned value is the final graph state; ``config`` identifies the
    checkpoint thread to continue or create.
    """
    with SqliteSaver.from_conn_string(DB_PATH) as session_checkpointer:
        session_checkpointer.conn.execute("PRAGMA busy_timeout=30000")
        graph = workflow.compile(checkpointer=session_checkpointer)
        return graph.invoke(state, config=config)


# Function: get_latest_thread_state
# Purpose: Read the saved LangGraph state for one browser thread using an isolated connection. The graph is
# compiled against the same SQLite checkpoint database used by streaming and invocation, allowing the
# UI to detect and resume interruptions.
def get_latest_thread_state(thread_id: str):
    """Read the saved LangGraph state for one browser thread using an isolated connection.

    The graph is compiled against the same SQLite checkpoint database used by
    streaming and invocation, allowing the UI to detect and resume interruptions.
    """
    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string(DB_PATH) as session_checkpointer:
        session_checkpointer.conn.execute("PRAGMA busy_timeout=30000")
        graph = workflow.compile(checkpointer=session_checkpointer)
        return graph.get_state(config)