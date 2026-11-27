# app.py
import os
import logging
import warnings
import uuid
from html import escape

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

warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change in a future version",
    category=PendingDeprecationWarning,
)

from src.agent_graph import set_status_callback
from src.config_loader import load_version_config
from src.llm_provider import GROQ_MODEL, MODEL_LABELS, OLLAMA_MODEL
from src.image_store import resolve_image_asset

st.set_page_config(
    page_title="Laddu Bullodu RAG Assistant",
    page_icon="🟠",
    layout="wide",
)


def render_retrieved_images(images: list[dict[str, str | int]]):
    """Display validated local image assets attached to an assistant response."""
    for image in images:
        if not isinstance(image, dict):
            continue
        image_path = resolve_image_asset(
            str(image.get("asset_id", "")),
            str(image.get("extension", "")),
        )
        if image_path is None:
            st.warning("A retrieved document image is no longer available.")
            continue
        st.image(str(image_path), caption=str(image.get("caption", "Document image")))


st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --lb-bg: #050914;
        --lb-panel: #0a1221;
        --lb-panel-soft: #0d192b;
        --lb-border: rgba(148, 180, 218, 0.18);
        --lb-text: #f3f7fc;
        --lb-muted: #9aacc3;
        --lb-blue: #83c7ff;
        --lb-cyan: #7ce7ef;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background:
            radial-gradient(ellipse at 50% 5%, rgba(32, 111, 190, 0.17), transparent 38rem),
            var(--lb-bg);
        color: var(--lb-text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1240px;
        padding: 1.6rem 2.5rem 8rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1423 0%, #080e19 100%);
        border-right: 1px solid var(--lb-border);
    }

    [data-testid="stSidebar"] * {
        color: var(--lb-text);
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--lb-muted);
    }

    .lb-nav {
        align-items: center;
        border: 1px solid var(--lb-border);
        border-radius: 16px;
        background: rgba(8, 15, 27, 0.78);
        display: flex;
        gap: 12px;
        margin: 0.15rem 0 4.7rem;
        padding: 12px 18px;
    }

    .lb-mark {
        align-items: center;
        background: linear-gradient(135deg, #8dd3ff, #738aff);
        border-radius: 11px;
        color: #06101e;
        display: inline-flex;
        font-size: 1.15rem;
        font-weight: 850;
        height: 38px;
        justify-content: center;
        width: 38px;
    }

    .lb-brand {
        color: var(--lb-text);
        font-size: 1.05rem;
        font-weight: 720;
        letter-spacing: -0.025em;
    }

    .lb-brand-sub {
        color: var(--lb-muted);
        font-size: 0.66rem;
        letter-spacing: 0.15em;
        margin-left: 9px;
        text-transform: uppercase;
    }

    .lb-nav-meta {
        color: var(--lb-muted);
        font-size: 0.82rem;
        margin-left: auto;
    }

    .lb-status-dot {
        background: #64e4bd;
        border-radius: 50%;
        box-shadow: 0 0 12px rgba(100, 228, 189, 0.65);
        display: inline-block;
        height: 7px;
        margin: 0 8px 1px 0;
        width: 7px;
    }

    .lb-hero {
        margin: 0 auto;
        max-width: 930px;
        padding: 0 0 2rem;
        text-align: center;
    }

    .lb-eyebrow {
        border: 1px solid rgba(131, 199, 255, 0.28);
        border-radius: 999px;
        color: #b7dfff;
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 650;
        letter-spacing: 0.14em;
        margin-bottom: 1.45rem;
        padding: 8px 14px;
        text-transform: uppercase;
    }

    .lb-hero h1 {
        color: var(--lb-text);
        font-size: clamp(2.8rem, 6vw, 5rem);
        font-weight: 540;
        letter-spacing: -0.065em;
        line-height: 1.04;
        margin: 0 auto 1.25rem;
        max-width: 900px;
    }

    .lb-hero h1 span {
        background: linear-gradient(90deg, #8ed3ff 5%, #9db8ff 55%, #8de8eb);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
    }

    .lb-hero-copy {
        color: #a7b8cd;
        font-size: clamp(1rem, 1.5vw, 1.18rem);
        line-height: 1.75;
        margin: 0 auto;
        max-width: 710px;
    }

    .lb-capabilities {
        color: #7fc4fb;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        margin: 1.45rem 0 2rem;
    }

    .lb-section-label {
        color: #b5c7dc;
        font-size: 0.77rem;
        font-weight: 650;
        letter-spacing: 0.11em;
        margin: 1rem 0 0.8rem;
        text-transform: uppercase;
    }

    .lb-feature {
        background: linear-gradient(145deg, rgba(15, 29, 49, 0.92), rgba(8, 16, 29, 0.94));
        border: 1px solid var(--lb-border);
        border-radius: 16px;
        height: 100%;
        min-height: 145px;
        padding: 19px 20px;
    }

    .lb-feature-icon {
        align-items: center;
        background: rgba(115, 186, 250, 0.12);
        border: 1px solid rgba(131, 199, 255, 0.14);
        border-radius: 10px;
        color: var(--lb-blue);
        display: inline-flex;
        font-size: 1rem;
        height: 34px;
        justify-content: center;
        margin-bottom: 13px;
        width: 34px;
    }

    .lb-feature-title {
        color: var(--lb-text);
        font-size: 0.96rem;
        font-weight: 680;
        margin-bottom: 6px;
    }

    .lb-feature-copy {
        color: var(--lb-muted);
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .lb-flow-board {
        background: linear-gradient(145deg, rgba(10, 20, 36, 0.97), rgba(7, 13, 24, 0.98));
        border: 1px solid rgba(131, 199, 255, 0.24);
        border-radius: 18px;
        margin: 0.75rem 0 1rem;
        overflow: hidden;
        padding: 18px;
    }

    .lb-flow-heading {
        align-items: flex-start;
        display: flex;
        justify-content: space-between;
        margin-bottom: 16px;
    }

    .lb-flow-title {
        color: #eef6ff;
        font-size: 0.95rem;
        font-weight: 680;
    }

    .lb-flow-subtitle {
        color: #94a8c1;
        font-size: 0.76rem;
        margin-top: 4px;
    }

    .lb-flow-state {
        border: 1px solid rgba(131, 199, 255, 0.24);
        border-radius: 999px;
        color: #9dcfff;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        padding: 6px 10px;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .lb-flow-state.complete {
        border-color: rgba(100, 228, 189, 0.35);
        color: #83e8c4;
    }

    .lb-flow-state.failed {
        border-color: rgba(255, 120, 139, 0.42);
        color: #ff9aa9;
    }

    .lb-flow-track {
        align-items: stretch;
        display: grid;
        gap: 8px;
        grid-template-columns: repeat(6, minmax(0, 1fr));
    }

    .lb-flow-stage {
        background: rgba(13, 25, 43, 0.9);
        border: 1px solid rgba(129, 158, 194, 0.19);
        border-radius: 12px;
        min-height: 94px;
        padding: 11px 10px;
        position: relative;
    }

    .lb-flow-stage:not(:last-child)::after {
        color: #58708e;
        content: "›";
        font-size: 1.2rem;
        position: absolute;
        right: -8px;
        top: 34px;
        z-index: 1;
    }

    .lb-flow-stage.active {
        animation: lb-flow-pulse 1.8s ease-in-out infinite;
        background: linear-gradient(145deg, rgba(23, 58, 91, 0.98), rgba(13, 31, 52, 0.98));
        border-color: rgba(131, 199, 255, 0.82);
        box-shadow: 0 0 22px rgba(70, 161, 239, 0.15);
    }

    .lb-flow-stage.done {
        background: linear-gradient(145deg, rgba(15, 54, 57, 0.74), rgba(12, 29, 39, 0.94));
        border-color: rgba(105, 220, 194, 0.35);
    }

    .lb-flow-stage.failed {
        background: rgba(61, 25, 39, 0.8);
        border-color: rgba(255, 120, 139, 0.6);
    }

    .lb-flow-index {
        color: #7e9ab8;
        font-size: 0.62rem;
        letter-spacing: 0.11em;
    }

    .lb-flow-stage.active .lb-flow-index {
        color: #a6dcff;
    }

    .lb-flow-stage.done .lb-flow-index {
        color: #83e8c4;
    }

    .lb-flow-stage-title {
        color: #e8f2ff;
        font-size: 0.77rem;
        font-weight: 670;
        line-height: 1.3;
        margin-top: 8px;
    }

    .lb-flow-detail {
        color: #92a7c0;
        font-size: 0.67rem;
        line-height: 1.4;
        margin-top: 5px;
        overflow-wrap: anywhere;
    }

    .lb-flow-live {
        align-items: center;
        border-top: 1px solid rgba(129, 158, 194, 0.15);
        color: #a9bfd8;
        display: flex;
        font-size: 0.76rem;
        gap: 9px;
        margin-top: 13px;
        padding-top: 12px;
    }

    .lb-flow-live-dot {
        animation: lb-flow-pulse 1.4s ease-in-out infinite;
        background: #79c7ff;
        border-radius: 50%;
        flex: 0 0 7px;
        height: 7px;
    }

    .lb-agent-flow-scroll {
        overflow-x: auto;
        width: 100%;
    }

    .lb-agent-flow {
        display: block;
        min-width: 760px;
        width: 100%;
    }

    .lb-agent-flow .flow-node rect {
        fill: #111d30;
        stroke: #39516c;
        stroke-width: 1.5;
        transition: fill 180ms ease, stroke 180ms ease;
    }

    .lb-agent-flow .flow-node.active rect {
        animation: lb-flow-pulse 1.8s ease-in-out infinite;
        fill: #153958;
        stroke: #83c7ff;
        stroke-width: 2.5;
    }

    .lb-agent-flow .flow-node.done rect {
        fill: #12363a;
        stroke: #65cfb5;
    }

    .lb-agent-flow .flow-node.failed rect {
        fill: #3d1927;
        stroke: #ff788b;
    }

    .lb-agent-flow .flow-node.selected rect {
        fill: #172f49;
        stroke: #79bdf0;
    }

    .lb-agent-flow .node-title {
        fill: #edf6ff;
        font-family: inherit;
        font-size: 15px;
        font-weight: 650;
        text-anchor: middle;
    }

    .lb-agent-flow .node-subtitle {
        fill: #9ab0c9;
        font-family: inherit;
        font-size: 10px;
        text-anchor: middle;
    }

    .lb-agent-flow .flow-label {
        fill: #91bde1;
        font-family: inherit;
        font-size: 10px;
        font-weight: 650;
        letter-spacing: 1.5px;
        text-anchor: middle;
    }

    .lb-agent-flow .flow-edge {
        fill: none;
        marker-end: url(#lb-arrow);
        stroke: #6386a8;
        stroke-width: 2;
    }

    .lb-agent-flow .flow-edge.loop {
        marker-end: url(#lb-arrow-loop);
        stroke: #75bce9;
        stroke-dasharray: 5 7;
    }

    .lb-agent-flow .flow-edge.context {
        marker-end: url(#lb-arrow-loop);
        opacity: 0.7;
        stroke-dasharray: 3 6;
        stroke-width: 1.5;
    }

    .lb-agent-flow .context-node rect {
        fill: #0c1727;
        stroke: #2e465f;
        stroke-width: 1.2;
    }

    .lb-agent-flow .context-title {
        fill: #c3d9ee;
        font-family: inherit;
        font-size: 11px;
        font-weight: 600;
        text-anchor: middle;
    }

    .lb-agent-flow .context-subtitle {
        fill: #819ab4;
        font-family: inherit;
        font-size: 9px;
        text-anchor: middle;
    }

    .lb-agent-loop-note {
        background: rgba(15, 34, 54, 0.75);
        border: 1px solid rgba(117, 188, 233, 0.2);
        border-radius: 10px;
        color: #a9c5df;
        font-size: 0.73rem;
        line-height: 1.5;
        margin-top: 8px;
        padding: 9px 12px;
    }

    .lb-flow-state.complete + .lb-flow-track .lb-flow-stage {
        animation: none;
    }

    @keyframes lb-flow-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.67; }
    }

    div.stButton > button {
        background: rgba(12, 24, 41, 0.84);
        border: 1px solid var(--lb-border);
        border-radius: 12px;
        color: #dcecff;
        min-height: 3.1rem;
        transition: border-color 140ms ease, background 140ms ease, transform 140ms ease;
    }

    div.stButton > button:hover {
        background: rgba(24, 48, 75, 0.95);
        border-color: rgba(131, 199, 255, 0.52);
        color: white;
        transform: translateY(-1px);
    }

    [data-testid="stChatMessage"] {
        background: rgba(10, 18, 33, 0.86);
        border: 1px solid var(--lb-border);
        border-radius: 16px;
        margin: 0.65rem 0;
    }

    [data-testid="stChatInput"] {
        background: rgba(8, 15, 27, 0.96);
        border: 1px solid rgba(131, 199, 255, 0.32);
        border-radius: 16px;
        box-shadow: 0 14px 50px rgba(0, 0, 0, 0.28);
    }

    [data-testid="stBottom"] {
        background: linear-gradient(180deg, rgba(5, 9, 20, 0), #050914 35%);
    }

    [data-testid="stBottomBlockContainer"] {
        background: transparent;
        padding-top: 1rem;
    }

    [data-testid="stChatInput"] textarea {
        color: var(--lb-text);
    }

    [data-testid="stMetric"] {
        background: rgba(16, 28, 45, 0.75);
        border: 1px solid var(--lb-border);
        border-radius: 12px;
        padding: 10px 12px;
    }

    @media (max-width: 720px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 7rem;
        }

        .lb-nav {
            margin-bottom: 3rem;
            padding: 10px 12px;
        }

        .lb-nav-meta,
        .lb-brand-sub {
            display: none;
        }

        .lb-hero h1 {
            font-size: clamp(2.55rem, 12vw, 3.8rem);
        }

        .lb-flow-track {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .lb-flow-stage:not(:last-child)::after {
            content: "";
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Render a workflow-specific live diagram from streamed LangGraph node updates.
def render_execution_flow(
    workflow_mode: str,
    completed_nodes: set[str],
    active_node: str,
    active_worker: str,
    intent: str,
    current_activity: str,
    run_state: str = "running",
) -> str:
    """Render a workflow-specific live diagram from streamed LangGraph node updates."""
    is_autonomous = workflow_mode == "Autonomous multi-agent"
    if is_autonomous:
        return render_autonomous_agent_flow(
            completed_nodes,
            active_node,
            active_worker,
            current_activity,
            run_state,
        )
    selected_intent = intent.replace("_", " ").title() if intent else "Intent-based route"
    if intent == "RAG":
        specialist_detail = "Dense + BM25 → BGE reranking → RAG or CRAG."
    elif intent:
        specialist_detail = f"{selected_intent} specialist workflow."
    else:
        specialist_detail = "RAG pipeline or selected domain workflow."
    stages = (
        ("request", "Request", "User goal and selected model."),
        ("router", "Intent router", "Classify request and choose route."),
        ("specialist", "Specialist", specialist_detail),
        ("answer", "Response", "Generate or synthesize the answer."),
        ("guardrail", "Safety check", "Sanitize output and record metrics."),
        ("complete", "Delivered", "Return result to chat."),
    )
    workflow_label = "Default intent-routed agentic workflow"
    loop_note = "RAG uses a relevance threshold to choose local context or web-backed CRAG."

    html_stages = []
    for index, (stage_id, title, detail) in enumerate(stages, start=1):
        if stage_id == "request":
            node_names = set()
        elif stage_id == "router":
            node_names = {"router_agent"}
        elif stage_id == "specialist":
            node_names = {
                "retrieve_node",
                "rerank_node",
                "domain_executor_node",
                "crag_fallback_node",
            }
        elif stage_id == "answer":
            node_names = {"rag_generate_node"}
        elif stage_id == "complete":
            node_names = set()
        else:
            node_names = {stage_id}

        if stage_id == "guardrail":
            node_names = {"guardrail"}

        state_class = ""
        if run_state == "complete":
            state_class = "done"
        elif run_state == "failed" and active_node in node_names:
            state_class = "failed"
        elif active_node in node_names:
            state_class = "active"
        elif node_names.intersection(completed_nodes):
            state_class = "done"
        elif stage_id == "request" and (
            "router_agent" in completed_nodes or active_node == "router_agent"
        ):
            state_class = "done"
        elif stage_id == "complete" and run_state == "complete":
            state_class = "done"

        safe_title = escape(title)
        safe_detail = escape(detail)
        html_stages.append(
            f"""
            <div class="lb-flow-stage {state_class}">
              <div class="lb-flow-index">STEP {index:02d}</div>
              <div class="lb-flow-stage-title">{safe_title}</div>
              <div class="lb-flow-detail">{safe_detail}</div>
            </div>
            """
        )

    state_label = {
        "running": "In progress",
        "complete": "Completed",
        "failed": "Needs attention",
    }.get(run_state, "In progress")
    state_class = escape(run_state)
    safe_activity = escape(current_activity or "Starting workflow...")
    safe_loop_note = escape(loop_note)
    return f"""
    <div class="lb-flow-board">
      <div class="lb-flow-heading">
        <div>
          <div class="lb-flow-title">{escape(workflow_label)}</div>
          <div class="lb-flow-subtitle">{safe_loop_note}</div>
        </div>
        <div class="lb-flow-state {state_class}">{escape(state_label)}</div>
      </div>
      <div class="lb-flow-track">{''.join(html_stages)}</div>
      <div class="lb-flow-live">
        <span class="lb-flow-live-dot"></span>
        <span>{safe_activity}</span>
      </div>
    </div>
    """


# Render the autonomous model/action/observation loop and its live specialist.
def render_autonomous_agent_flow(
    completed_nodes: set[str],
    active_node: str,
    active_worker: str,
    current_activity: str,
    run_state: str,
) -> str:
    """Render the autonomous model/action/observation loop and its live specialist."""
    worker_tools = (
        ("delegate_to_rag", "RAG retrieval", "Local docs · rerank · CRAG"),
        ("delegate_to_web_search", "Web research", "Serper · current information"),
        ("delegate_to_site_selection", "Site selection", "Power · fiber · locations"),
        ("delegate_to_supply_chain", "Supply chain", "Neo4j · entities · routes"),
    )
    selected_tool = f"delegate_to_{active_worker.lower().replace(' ', '_')}"

    # Implement the node_state function used by this application workflow.
    def node_state(node_name: str) -> str:
        if run_state == "complete":
            return "done"
        if run_state == "failed" and active_node == node_name:
            return "failed"
        if active_node == node_name:
            return "active"
        return "done" if node_name in completed_nodes else ""

    # Implement the card function used by this application workflow.
    def card(
        x: int,
        y: int,
        width: int,
        height: int,
        title: str,
        subtitle: str,
        state: str = "",
    ) -> str:
        return f"""
        <g class="flow-node {escape(state)}">
          <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" />
          <text class="node-title" x="{x + width / 2}" y="{y + 27}">{escape(title)}</text>
          <text class="node-subtitle" x="{x + width / 2}" y="{y + 45}">{escape(subtitle)}</text>
        </g>
        """

    tool_cards = []
    tool_x_positions = (30, 260, 490, 720)
    for (tool_name, title, subtitle), x in zip(worker_tools, tool_x_positions):
        state = "selected" if selected_tool == tool_name else ""
        if (
            run_state == "failed"
            and active_node == "worker"
            and selected_tool == tool_name
        ):
            state = "failed"
        elif active_node == "worker" and selected_tool == tool_name:
            state = "active"
        elif "worker" in completed_nodes and selected_tool == tool_name:
            state = "done"
        tool_cards.append(card(x, 270, 210, 66, title, subtitle, state))

    request_state = (
        "done" if run_state == "complete" or "initialize" in completed_nodes else ""
    )
    model_state = node_state("supervisor")
    critic_state = node_state("critic")
    if run_state == "failed":
        result_state = "failed"
    elif run_state == "complete" or "guardrail" in completed_nodes:
        result_state = "done"
    elif active_node in ("finalize", "guardrail"):
        result_state = "active"
    else:
        result_state = ""
    state_label = {
        "running": "In progress",
        "complete": "Completed",
        "failed": "Needs attention",
    }.get(run_state, "In progress")
    activity = escape(current_activity or "Starting autonomous run...")
    selected_label = escape(active_worker or "Awaiting supervisor selection")
    return f"""
    <div class="lb-flow-board">
      <div class="lb-flow-heading">
        <div>
          <div class="lb-flow-title">Autonomous agent execution</div>
          <div class="lb-flow-subtitle">Model → action → observation → result</div>
        </div>
        <div class="lb-flow-state {escape(run_state)}">{escape(state_label)}</div>
      </div>
      <div class="lb-agent-flow-scroll">
        <svg class="lb-agent-flow" viewBox="0 0 960 575" role="img"
             aria-label="Autonomous flow from user request through model, specialist tools, critic feedback loop, and final result">
          <defs>
            <marker id="lb-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4"
                    orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L8,4 L0,8 z" fill="#6386a8" />
            </marker>
            <marker id="lb-arrow-loop" markerWidth="8" markerHeight="8" refX="6" refY="4"
                    orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L8,4 L0,8 z" fill="#75bce9" />
            </marker>
          </defs>
          <path class="flow-edge" d="M480 72 V112" />
          <path class="flow-edge" d="M480 182 V226" />
          <path class="flow-edge" d="M135 245 H825" />
          <path class="flow-edge" d="M135 245 V268 M365 245 V268 M595 245 V268 M825 245 V268" />
          <path class="flow-edge" d="M135 336 V360 H480 V388" />
          <path class="flow-edge" d="M365 336 V360" />
          <path class="flow-edge" d="M595 336 V360" />
          <path class="flow-edge" d="M825 336 V360 H480" />
          <path class="flow-edge" d="M595 418 H665" />
          <path class="flow-edge loop" d="M365 418 H18 V147 H382" />
          <path class="flow-edge context" d="M150 495 V470 H420 V183" />
          <path class="flow-edge context" d="M480 495 V470 H420 V183" />
          <path class="flow-edge context" d="M810 495 V470 H540 V183" />
          <path class="flow-edge" d="M578 147 H940 V418 H908" />
          <text class="flow-label" x="480" y="214">ACTION · TOOL CALL</text>
          <text class="flow-label" x="480" y="380">OBSERVATION</text>
          <text class="flow-label" x="80" y="405">CRITIQUE / RETRY</text>
          <text class="flow-label" x="810" y="170">FINISH</text>
          {card(410, 20, 140, 52, "User request", "Goal for this run", request_state)}
          {card(382, 112, 196, 70, "Supervisor model", "Chooses a tool or finish", model_state)}
          {''.join(tool_cards)}
          {card(365, 388, 230, 60, "Critic · observation", "Check evidence and missing gaps", critic_state)}
          {card(665, 388, 243, 60, "Synthesized result", "Guardrail · metrics · chat", result_state)}
          <g class="context-node">
            <rect x="50" y="495" width="200" height="52" rx="10" />
            <text class="context-title" x="150" y="517">System instructions</text>
            <text class="context-subtitle" x="150" y="534">Goal · routing rules · safety</text>
          </g>
          <g class="context-node">
            <rect x="380" y="495" width="200" height="52" rx="10" />
            <text class="context-title" x="480" y="517">Run state</text>
            <text class="context-subtitle" x="480" y="534">Findings · critique · checkpoint</text>
          </g>
          <g class="context-node">
            <rect x="710" y="495" width="200" height="52" rx="10" />
            <text class="context-title" x="810" y="517">Connected sources</text>
            <text class="context-subtitle" x="810" y="534">Local docs · Serper · Neo4j</text>
          </g>
        </svg>
      </div>
      <div class="lb-agent-loop-note">
        <strong>Selected specialist:</strong> {selected_label}
        &nbsp;·&nbsp; Critic feedback loops back to the supervisor when more evidence or a refinement is needed.
      </div>
      <div class="lb-flow-live">
        <span class="lb-flow-live-dot"></span>
        <span>{activity}</span>
      </div>
    </div>
    """


# Update the live diagram from either a graph node or an in-node status message.
def update_execution_progress(
    progress: dict[str, str],
    message: str,
    completed_nodes: set[str],
    workflow_mode: str,
    flow_placeholder,
) -> None:
    """Update the live diagram from either a graph node or an in-node status message."""
    progress["activity"] = message
    lowered_message = message.lower()

    delegation_marker = "delegated to "
    if delegation_marker in lowered_message:
        tool_name = message[lowered_message.index(delegation_marker) + len(delegation_marker):]
        tool_name = tool_name.strip().rstrip(".")
        progress["worker"] = tool_name.replace("delegate_to_", "").replace("_", " ").title()
        progress["active_node"] = "worker"
    elif "supervisor" in lowered_message:
        progress["active_node"] = "supervisor"
    elif "critic" in lowered_message:
        progress["active_node"] = "critic"
    elif "synthesizing findings" in lowered_message or "synthesizing" in lowered_message:
        progress["active_node"] = "finalize"
    elif "validating output" in lowered_message or "guardrail" in lowered_message:
        progress["active_node"] = "guardrail"
    elif "hybrid retrieval" in lowered_message or "embedding model" in lowered_message:
        progress["active_node"] = "retrieve_node"
    elif "re-ranking" in lowered_message or "reranking" in lowered_message:
        progress["active_node"] = "rerank_node"
    elif "intent" in lowered_message or "router" in lowered_message:
        progress["active_node"] = "router_agent"

    flow_placeholder.html(
        render_execution_flow(
            workflow_mode,
            completed_nodes,
            progress["active_node"],
            progress["worker"],
            progress["intent"],
            progress["activity"],
        )
    )


with st.sidebar:
    st.markdown("### 🟠 Laddu Bullodu")
    st.caption("Agent workspace · LangGraph powered")
    st.subheader("Workflow")
    workflow_mode = st.radio(
        "Choose a workflow",
        options=("Default agentic AI flow", "Autonomous multi-agent"),
        index=0,
        key="workflow_mode",
        help=(
            "The default keeps the existing intent-routed graph. Autonomous mode "
            "lets a tool-calling supervisor delegate to specialist workflows and "
            "uses a critic with a bounded retry loop."
        ),
    )

if workflow_mode == "Autonomous multi-agent":
    from src.autonomous_graph import (
        get_latest_autonomous_thread_state as get_latest_thread_state,
        stream_autonomous_graph as stream_graph,
    )

    workflow_thread_suffix = "-autonomous"
else:
    from src.agent_graph import (
        get_latest_thread_state,
        stream_graph,
    )

    workflow_thread_suffix = ""

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
    st.caption(
        "Cloud models require their provider key in .env. Gemini and OpenRouter "
        "offer free-tier options with provider limits."
    )

# ------------------------------------------------------------------------------
# 1. Thread Persistence & Session Setup
# ------------------------------------------------------------------------------
# Each browser session receives a unique thread ID. The workflow suffix keeps
# default and autonomous checkpoints separate, even if both modes use the same
# browser session. Model and prompt metadata are attached to each run.
if "thread_id" not in st.session_state:
    # Browser tabs have independent Streamlit session state; never share the default checkpoint.
    st.session_state.thread_id = f"browser-{uuid.uuid4().hex}"

thread_config = {
    "configurable": {
        "thread_id": f"{st.session_state.thread_id}{workflow_thread_suffix}"
    },
    "metadata": {
        "model_name": selected_model or OLLAMA_MODEL,
        "prompt_version": load_version_config().get("version", "unknown"),
        "workflow_mode": (
            "autonomous"
            if workflow_mode == "Autonomous multi-agent"
            else "default"
        ),
        "graph_topology": (
            "autonomous-supervisor-v1"
            if workflow_mode == "Autonomous multi-agent"
            else "intent-routed-v1"
        ),
    },
}


# Persist a PII-redacted failure candidate with the current run lineage.
def record_workflow_failure(
    error: Exception, query: str, failed_node: str, intent: str
) -> None:
    """Persist a PII-redacted failure candidate with the current run lineage."""
    from src.guardrails import sanitize_text
    from src.metrics_logger import log_workflow_failure

    metadata = thread_config["metadata"]
    log_workflow_failure(
        thread_id=thread_config["configurable"]["thread_id"],
        query=sanitize_text(query)[:2000],
        failed_node=failed_node or "unknown",
        error_type=type(error).__name__,
        model_name=metadata["model_name"],
        prompt_version=metadata["prompt_version"],
        workflow_mode=metadata["workflow_mode"],
        graph_topology=metadata["graph_topology"],
        trace_id=metadata.get("trace_id"),
        intent=intent or "UNKNOWN",
    )


# Return the most recent user message from the Streamlit session.
def latest_user_query() -> str:
    """Return the most recent user message from the Streamlit session."""
    for message in reversed(st.session_state.messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
    <div class="lb-nav">
      <div class="lb-mark">L</div>
      <div>
        <span class="lb-brand">Laddu Bullodu</span>
        <span class="lb-brand-sub">Agent workspace</span>
      </div>
      <div class="lb-nav-meta"><span class="lb-status-dot"></span>LangGraph runtime ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

has_user_messages = any(
    message.get("role") == "user" for message in st.session_state.messages
)
suggested_prompt = None

if not has_user_messages:
    st.markdown(
        """
        <section class="lb-hero">
          <div class="lb-eyebrow">Retrieval · Research · Site intelligence</div>
          <h1>Put your knowledge<br><span>to work with AI agents.</span></h1>
          <p class="lb-hero-copy">
            Ask questions across your documents, research the live web, explore
            supply-chain relationships, or compare data-center locations—with
            one assistant built around your workflows.
          </p>
          <div class="lb-capabilities">Grounded answers&nbsp;&nbsp;·&nbsp;&nbsp;Specialist workflows&nbsp;&nbsp;·&nbsp;&nbsp;Human-readable reports</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="lb-section-label">Choose a place to start</div>', unsafe_allow_html=True)
    feature_columns = st.columns(3)
    feature_cards = (
        ("⌕", "Ask your documents", "Find answers in local project documents with hybrid retrieval."),
        ("↗", "Research a location", "Compare power, connectivity, risk, and other site criteria."),
        ("⟳", "Orchestrate agents", "Let a supervisor route a multi-part goal across specialists."),
    )
    for column, (icon, title, copy) in zip(feature_columns, feature_cards):
        with column:
            st.markdown(
                f"""
                <div class="lb-feature">
                  <div class="lb-feature-icon">{icon}</div>
                  <div class="lb-feature-title">{title}</div>
                  <div class="lb-feature-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="lb-section-label">Try a prompt</div>', unsafe_allow_html=True)
    prompt_columns = st.columns(3)
    prompt_suggestions = (
        (
            "Ask about features",
            "Explain the architecture and features of this chat assistant.",
        ),
        (
            "Compare Caribbean sites",
            "Compare data-center site options in the West Indies.",
        ),
        (
            "Explore agent mode",
            "How does autonomous multi-agent mode route and review work?",
        ),
    )
    for column, (button_label, prompt_suggestion) in zip(
        prompt_columns, prompt_suggestions
    ):
        with column:
            if st.button(button_label, use_container_width=True):
                suggested_prompt = prompt_suggestion

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_retrieved_images(message.get("images", []))

# ------------------------------------------------------------------------------
# 2. Checkpoint Auto-Resumption Logic on Startup
# ------------------------------------------------------------------------------
# LangGraph writes the next node to SQLite before a run ends. If an interrupted
# run has pending nodes, the UI streams them from the checkpoint instead of
# starting a duplicate assistant turn.
current_checkpoint = get_latest_thread_state(thread_config["configurable"]["thread_id"])

if current_checkpoint and current_checkpoint.next:
    st.info("⚡ Detected interrupted workflow execution. Automatically resuming from the selected workflow's SQLite checkpoint...")
    
    ai_response = ""
    retrieved_images = []
    with st.chat_message("assistant"):
        completed_nodes = set()
        progress = {
            "active_node": "",
            "worker": "",
            "intent": "",
            "activity": "Resuming the saved workflow...",
        }
        flow_placeholder = st.empty()
        flow_placeholder.html(
            render_execution_flow(
                workflow_mode,
                completed_nodes,
                progress["active_node"],
                progress["worker"],
                progress["intent"],
                progress["activity"],
            )
        )
        with st.status("Resuming agentic workflow from checkpoint...", expanded=True) as status:
            # Show a checkpoint-resumption update in the active Streamlit status panel.
            def update_status(msg: str):
                """Show a checkpoint-resumption update in the active Streamlit status panel."""
                update_execution_progress(
                    progress, msg, completed_nodes, workflow_mode, flow_placeholder
                )
                status.update(label=msg)
                st.write(f"• {msg}")

            set_status_callback(update_status)
            try:
                for event in stream_graph(None, config=thread_config, stream_mode="updates"):
                    for node_name, state_update in event.items():
                        if state_update.get("retrieved_images") is not None:
                            retrieved_images = state_update["retrieved_images"]
                        completed_nodes.add(node_name)
                        if state_update.get("intent"):
                            progress["intent"] = state_update["intent"]
                        if state_update.get("next_agent"):
                            progress["worker"] = state_update["next_agent"].replace(
                                "delegate_to_", ""
                            ).replace("_", " ").title()
                        worker_results = state_update.get("worker_results", [])
                        if worker_results:
                            progress["worker"] = worker_results[-1]["agent"].replace(
                                "delegate_to_", ""
                            ).replace("_", " ").title()
                        progress["active_node"] = node_name
                        progress["activity"] = f"Resuming step: {node_name}..."
                        flow_placeholder.html(
                            render_execution_flow(
                                workflow_mode,
                                completed_nodes,
                                progress["active_node"],
                                progress["worker"],
                                progress["intent"],
                                progress["activity"],
                            )
                        )
                        update_status(progress["activity"])
                        if "messages" in state_update and state_update["messages"]:
                            latest_msg = state_update["messages"][-1]
                            ai_response = getattr(latest_msg, "content", str(latest_msg))

                if not retrieved_images:
                    resumed_checkpoint = get_latest_thread_state(
                        thread_config["configurable"]["thread_id"]
                    )
                    if resumed_checkpoint:
                        retrieved_images = resumed_checkpoint.values.get(
                            "retrieved_images", []
                        )
                flow_placeholder.html(
                    render_execution_flow(
                        workflow_mode,
                        completed_nodes,
                        "",
                        progress["worker"],
                        progress["intent"],
                        "Workflow completed and response is ready.",
                        "complete",
                    )
                )
                status.update(label="Workflow successfully resumed and completed!", state="complete", expanded=False)
            except Exception as e:
                try:
                    record_workflow_failure(
                        e,
                        latest_user_query(),
                        progress.get("active_node", ""),
                        progress.get("intent", ""),
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Could not persist workflow resumption failure"
                    )
                ai_response = f"Error during workflow resumption: {str(e)}"
                flow_placeholder.html(
                    render_execution_flow(
                        workflow_mode,
                        completed_nodes,
                        progress["active_node"],
                        progress["worker"],
                        progress["intent"],
                        "The workflow stopped before completion.",
                        "failed",
                    )
                )
                status.update(label="Resumption failed!", state="error", expanded=True)
            finally:
                set_status_callback(None)

        if ai_response:
            st.markdown(ai_response)
            render_retrieved_images(retrieved_images)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": ai_response,
                    "images": retrieved_images,
                }
            )
else:
    pass

# ------------------------------------------------------------------------------
# 3. Handle New User Input
# ------------------------------------------------------------------------------
# Convert the most recent user turn into LangGraph messages, limit conversation
# history to the latest eight entries, and stream new graph events back to UI.
prompt = st.chat_input(
    "Ask about your documents, research, or site selection...",
    key="chat_prompt_input",
) or suggested_prompt

if prompt:
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
    retrieved_images = []
    with st.chat_message("assistant"):
        completed_nodes = set()
        progress = {
            "active_node": "",
            "worker": "",
            "intent": "",
            "activity": "Preparing the selected workflow...",
        }
        flow_placeholder = st.empty()
        flow_placeholder.html(
            render_execution_flow(
                workflow_mode,
                completed_nodes,
                progress["active_node"],
                progress["worker"],
                progress["intent"],
                progress["activity"],
            )
        )
        with st.status(
            "Initializing autonomous multi-agent workflow..." if workflow_mode == "Autonomous multi-agent"
            else "Initializing agentic workflow...",
            expanded=True,
        ) as status:
            # Render graph-node progress in the active Streamlit status panel.
            def update_status(msg: str):
                """Render graph-node progress in the active Streamlit status panel."""
                update_execution_progress(
                    progress, msg, completed_nodes, workflow_mode, flow_placeholder
                )
                status.update(label=msg)
                st.write(f"• {msg}")

            set_status_callback(update_status)

            try:
                state_input = {
                    "messages": graph_messages,
                    "selected_model": selected_model or OLLAMA_MODEL,
                    "thread_id": thread_config["configurable"]["thread_id"],
                    "workflow_mode": thread_config["metadata"]["workflow_mode"],
                    "prompt_version": thread_config["metadata"]["prompt_version"],
                    "graph_topology": thread_config["metadata"]["graph_topology"],
                }
                
                node_descriptions = {
                    "initialize": "Preparing the goal and autonomous run state...",
                    "supervisor": "Supervisor choosing a specialist or deciding to finish...",
                    "worker": "Executing the supervisor-selected specialist...",
                    "critic": "Critic reviewing evidence and identifying gaps...",
                    "finalize": "Synthesizing findings into a final answer...",
                    "guardrail": "Validating output against PII guardrails and recording metrics...",
                    "router_agent": "Analyzing user intent and selecting execution path...",
                    "retrieve_node": "Executing Hybrid Retrieval (Dense Vector + BM25) (k=20)...",
                    "rerank_node": "Scoring and filtering documents using BAAI BGE Cross-Encoder...",
                    "crag_fallback_node": "Corrective RAG triggered: executing web search...",
                    "domain_executor_node": "Executing specialized domain intelligence (Neo4j / Site Selection / Web)...",
                    "rag_generate_node": "Synthesizing final response...",
                }

                for event in stream_graph(state_input, config=thread_config, stream_mode="updates"):
                    for node_name, state_update in event.items():
                        if state_update.get("retrieved_images") is not None:
                            retrieved_images = state_update["retrieved_images"]
                        completed_nodes.add(node_name)
                        if state_update.get("intent"):
                            progress["intent"] = state_update["intent"]
                        if state_update.get("next_agent"):
                            progress["worker"] = state_update["next_agent"].replace(
                                "delegate_to_", ""
                            ).replace("_", " ").title()
                        worker_results = state_update.get("worker_results", [])
                        if worker_results:
                            progress["worker"] = worker_results[-1]["agent"].replace(
                                "delegate_to_", ""
                            ).replace("_", " ").title()
                        desc = node_descriptions.get(node_name, f"Executing node: `{node_name}`")
                        progress["active_node"] = node_name
                        progress["activity"] = desc
                        flow_placeholder.html(
                            render_execution_flow(
                                workflow_mode,
                                completed_nodes,
                                progress["active_node"],
                                progress["worker"],
                                progress["intent"],
                                progress["activity"],
                            )
                        )
                        update_status(desc)
                        if "messages" in state_update and state_update["messages"]:
                            latest_msg = state_update["messages"][-1]
                            if hasattr(latest_msg, "content"):
                                ai_response = latest_msg.content
                            elif isinstance(latest_msg, dict):
                                ai_response = latest_msg.get("content", "")

                if not retrieved_images:
                    final_checkpoint = get_latest_thread_state(
                        thread_config["configurable"]["thread_id"]
                    )
                    if final_checkpoint:
                        retrieved_images = final_checkpoint.values.get(
                            "retrieved_images", []
                        )
                if not ai_response:
                    final_checkpoint = get_latest_thread_state(
                        thread_config["configurable"]["thread_id"]
                    )
                    final_messages = (
                        final_checkpoint.values.get("messages", [])
                        if final_checkpoint
                        else []
                    )
                    latest_user_index = -1
                    for index, message in enumerate(final_messages):
                        role = (
                            message.get("role")
                            if isinstance(message, dict)
                            else getattr(message, "type", "")
                        )
                        if role in ("human", "user"):
                            latest_user_index = index
                    for message in final_messages[latest_user_index + 1:]:
                        role = (
                            message.get("role")
                            if isinstance(message, dict)
                            else getattr(message, "type", "")
                        )
                        content = (
                            message.get("content", "")
                            if isinstance(message, dict)
                            else getattr(message, "content", "")
                        )
                        if content and role in ("ai", "assistant"):
                            ai_response = content
                    if not ai_response:
                        raise RuntimeError(
                            "Workflow stream ended without a final assistant response."
                        )

                flow_placeholder.html(
                    render_execution_flow(
                        workflow_mode,
                        completed_nodes,
                        "",
                        progress["worker"],
                        progress["intent"],
                        "Workflow completed and response is ready.",
                        "complete",
                    )
                )
                status.update(label="Workflow completed successfully!", state="complete", expanded=False)
            except Exception as e:
                try:
                    record_workflow_failure(
                        e, prompt, progress.get("active_node", ""),
                        progress.get("intent", "")
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Could not persist workflow failure"
                    )
                ai_response = f"Error processing request: {str(e)}"
                flow_placeholder.html(
                    render_execution_flow(
                        workflow_mode,
                        completed_nodes,
                        progress["active_node"],
                        progress["worker"],
                        progress["intent"],
                        "The workflow stopped before completion.",
                        "failed",
                    )
                )
                status.update(label="Workflow failed!", state="error", expanded=True)
            finally:
                set_status_callback(None)

        st.markdown(ai_response)
        render_retrieved_images(retrieved_images)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": ai_response,
                "images": retrieved_images,
            }
        )