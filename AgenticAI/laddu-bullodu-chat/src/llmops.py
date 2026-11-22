"""Shared trace metadata and lightweight runtime quality checks."""

from contextlib import contextmanager
import re
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from src.config_loader import load_version_config

_TRACER = trace.get_tracer("laddu-bullodu-chat")


# Function: workflow_span
# Purpose: Create a parent span carrying safe, versioned workflow lineage.
@contextmanager
def workflow_span(
    state: dict[str, Any] | None,
    config: dict[str, Any],
    workflow_mode: str,
) -> Generator[Span, None, None]:
    """Create a parent span carrying safe, versioned workflow lineage."""
    metadata = config.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        config["metadata"] = metadata
    configurable = config.get("configurable") or {}
    configuration = load_version_config()
    thread_id = (
        metadata.get("thread_id")
        or configurable.get("thread_id")
        or (state or {}).get("thread_id", "unknown")
    )
    run_metadata = {
        "llmops.model_name": metadata.get(
            "model_name", (state or {}).get("selected_model", "unknown")
        ),
        "llmops.prompt_version": metadata.get(
            "prompt_version", configuration.get("version", "unknown")
        ),
        "llmops.workflow_mode": workflow_mode,
        "llmops.graph_topology": metadata.get(
            "graph_topology",
            "autonomous-supervisor-v1"
            if workflow_mode == "autonomous"
            else "intent-routed-v1",
        ),
        "llmops.thread_id": thread_id,
    }

    with _TRACER.start_as_current_span("assistant.workflow") as span:
        for key, value in run_metadata.items():
            span.set_attribute(key, str(value))

        span_context = span.get_span_context()
        if span_context.is_valid:
            trace_id = f"{span_context.trace_id:032x}"
            metadata["trace_id"] = trace_id
            if state is not None:
                state["trace_id"] = trace_id

        if state is not None:
            state.setdefault("workflow_mode", workflow_mode)
            state.setdefault("prompt_version", run_metadata["llmops.prompt_version"])
            state.setdefault("graph_topology", run_metadata["llmops.graph_topology"])

        try:
            yield span
        except Exception as error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
            raise


# Function: set_run_attribute
# Purpose: Add a non-sensitive attribute to the current workflow trace.
def set_run_attribute(name: str, value: str | float | int | bool | None) -> None:
    """Add a non-sensitive attribute to the current workflow trace."""
    if value is not None:
        trace.get_current_span().set_attribute(f"llmops.{name}", value)


# Function: current_trace_id
# Purpose: Return the active trace ID, or ``None`` when tracing is not recording.
def current_trace_id() -> str | None:
    """Return the active trace ID, or ``None`` when tracing is not recording."""
    span_context = trace.get_current_span().get_span_context()
    return f"{span_context.trace_id:032x}" if span_context.is_valid else None


# Function: rag_answer_context_overlap
# Purpose: Return a lexical overlap heuristic, not a semantic faithfulness score.
def rag_answer_context_overlap(answer: str, contexts: list[str]) -> float:
    """Return a lexical overlap heuristic, not a semantic faithfulness score."""
    answer_terms = set(re.findall(r"\b[a-z0-9]{3,}\b", answer.lower()))
    context_terms = set(
        re.findall(r"\b[a-z0-9]{3,}\b", "\n".join(contexts).lower())
    )
    if not answer_terms:
        return 0.0
    return len(answer_terms & context_terms) / len(answer_terms)


# Function: site_report_dimensions
# Purpose: Check that the report mentions the required site-selection dimensions.
def site_report_dimensions(report: str) -> dict[str, bool]:
    """Check that the report mentions the required site-selection dimensions."""
    text = report.lower()
    terms = {
        "zoning": ("zoning", "environmental"),
        "power": ("power", "electricity", "grid"),
        "water": ("water", "cooling"),
        "financials": ("financial", "tax", "incentive"),
    }
    return {
        dimension: any(term in text for term in alternatives)
        for dimension, alternatives in terms.items()
    }


# Function: crag_fallback_required
# Purpose: Return whether the configured raw reranker score selects CRAG fallback.
def crag_fallback_required(score: float, threshold: float) -> bool:
    """Return whether the configured raw reranker score selects CRAG fallback."""
    return score < threshold
