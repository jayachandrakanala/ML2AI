import sqlite3

from langchain_core.documents import Document
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.llmops_eval import evaluate_golden_cases, load_golden_cases
from src.hybrid_search import HybridRetriever
from src.metrics_logger import (
    get_metrics_summary,
    log_execution,
    log_workflow_failure,
)
from src.mlops import (
    rag_answer_context_overlap,
    set_run_attribute,
    site_report_dimensions,
    workflow_span,
)
import src.metrics_logger as metrics_logger


def test_golden_set_passes_configured_offline_regression_gates():
    result = evaluate_golden_cases()

    assert len(load_golden_cases()) == 20
    assert result["passed"], result
    assert result["sample_counts"] == {
        "intent": 20,
        "bm25_source": 8,
        "site_region": 6,
        "crag_route": 8,
    }


def test_rag_overlap_is_explicitly_a_lexical_heuristic():
    assert rag_answer_context_overlap(
        "The assistant uses hybrid retrieval.", ["Hybrid retrieval uses BM25."]
    ) == 0.6
    assert rag_answer_context_overlap("!!!", ["Some context"]) == 0.0


def test_workflow_span_records_versioned_run_lineage():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    state = {"selected_model": "test-model", "thread_id": "thread-42"}
    config = {
        "configurable": {"thread_id": "thread-42"},
        "metadata": {
            "model_name": "test-model",
            "prompt_version": "1.0.0",
            "graph_topology": "intent-routed-v1",
        },
    }

    with workflow_span(state, config, "default"):
        set_run_attribute("intent", "RAG")

    span = exporter.get_finished_spans()[0]
    assert span.attributes["llmops.model_name"] == "test-model"
    assert span.attributes["llmops.prompt_version"] == "1.0.0"
    assert span.attributes["llmops.graph_topology"] == "intent-routed-v1"
    assert span.attributes["llmops.thread_id"] == "thread-42"
    assert span.attributes["llmops.intent"] == "RAG"
    assert state["trace_id"] == config["metadata"]["trace_id"]


def test_site_report_dimension_check_covers_required_categories():
    result = site_report_dimensions(
        "Zoning/environment, power grid, water cooling, and financial incentives."
    )

    assert all(result.values())
    assert not site_report_dimensions("No site evidence was found.")["power"]


def test_bm25_returns_matching_documents_with_negative_idf_scores():
    documents = [
        Document(page_content="architecture assistant workflow", metadata={"source": "a"}),
        Document(page_content="architecture assistant retrieval", metadata={"source": "b"}),
    ]
    retriever = HybridRetriever(documents)

    results = retriever.bm25_search("architecture assistant", k=3)

    assert results
    assert all(score < 0 for _, score in results)
    assert {document.metadata["source"] for document, _ in results} == {"a", "b"}


def test_metrics_migrate_legacy_table_and_store_lineage(tmp_path, monkeypatch):
    database_path = tmp_path / "metrics.sqlite"
    monkeypatch.setattr(metrics_logger, "METRICS_DB_PATH", str(database_path))
    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            CREATE TABLE execution_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                timestamp REAL,
                intent TEXT,
                latency_seconds REAL,
                token_cost REAL,
                citation_mode TEXT,
                status TEXT
            )
        """)

    log_execution(
        thread_id="thread-1",
        intent="RAG",
        latency_seconds=1.2,
        token_cost=0.01,
        citation_mode="[RAG]",
        model_name="test-model",
        prompt_version="1.0.0",
        workflow_mode="default",
        graph_topology="intent-routed-v1",
        trace_id="abc123",
    )

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT model_name, prompt_version, workflow_mode, graph_topology, trace_id "
            "FROM execution_logs"
        ).fetchone()
    assert row == (
        "test-model",
        "1.0.0",
        "default",
        "intent-routed-v1",
        "abc123",
    )
    assert get_metrics_summary()["total_requests"] == 1


def test_workflow_failures_are_counted_and_retain_trace_lineage(tmp_path, monkeypatch):
    database_path = tmp_path / "metrics.sqlite"
    monkeypatch.setattr(metrics_logger, "METRICS_DB_PATH", str(database_path))

    log_workflow_failure(
        thread_id="thread-2",
        query="Find an international site",
        failed_node="evaluator",
        error_type="RuntimeError",
        model_name="test-model",
        prompt_version="1.0.0",
        workflow_mode="autonomous",
        graph_topology="autonomous-supervisor-v1",
        trace_id="deadbeef",
        intent="SITE_SELECTION",
    )

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT query, failed_node, error_type, trace_id FROM workflow_failures"
        ).fetchone()
    assert row == (
        "Find an international site",
        "evaluator",
        "RuntimeError",
        "deadbeef",
    )
    assert get_metrics_summary()["failure_rate"] == 100.0
