# src/metrics_logger.py
import sqlite3
import time
from pathlib import Path

METRICS_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "metrics.sqlite")
Path(METRICS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

def init_metrics_db():
    """Create the metrics database directory and execution-log table if needed."""
    conn = sqlite3.connect(METRICS_DB_PATH, check_same_thread=False)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                timestamp REAL,
                intent TEXT,
                latency_seconds REAL,
                token_cost REAL,
                citation_mode TEXT,
                status TEXT,
                model_name TEXT,
                prompt_version TEXT,
                workflow_mode TEXT,
                graph_topology TEXT,
                trace_id TEXT
            )
        """)
        existing_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(execution_logs)")
        }
        for column in (
            "model_name",
            "prompt_version",
            "workflow_mode",
            "graph_topology",
            "trace_id",
        ):
            if column not in existing_columns:
                cursor.execute(f"ALTER TABLE execution_logs ADD COLUMN {column} TEXT")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                timestamp REAL,
                query TEXT,
                failed_node TEXT,
                error_type TEXT,
                model_name TEXT,
                prompt_version TEXT,
                workflow_mode TEXT,
                graph_topology TEXT,
                trace_id TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def log_execution(
    thread_id: str,
    intent: str,
    latency_seconds: float,
    token_cost: float,
    citation_mode: str,
    status: str = "SUCCESS",
    model_name: str | None = None,
    prompt_version: str | None = None,
    workflow_mode: str | None = None,
    graph_topology: str | None = None,
    trace_id: str | None = None,
):
    """Append one completed workflow turn and its operational measurements to SQLite.

    ``token_cost`` is the application's estimate; ``citation_mode`` stores the
    response badge/path used for the turn. Lineage fields are optional for
    compatibility with non-UI callers.
    """
    init_metrics_db()
    conn = sqlite3.connect(METRICS_DB_PATH, check_same_thread=False)
    try:
        conn.execute("""
            INSERT INTO execution_logs (
                thread_id, timestamp, intent, latency_seconds, token_cost,
                citation_mode, status, model_name, prompt_version,
                workflow_mode, graph_topology, trace_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            thread_id,
            time.time(),
            intent,
            latency_seconds,
            token_cost,
            citation_mode,
            status,
            model_name,
            prompt_version,
            workflow_mode,
            graph_topology,
            trace_id,
        ))
        conn.commit()
    finally:
        conn.close()


def log_workflow_failure(
    thread_id: str,
    query: str,
    failed_node: str,
    error_type: str,
    model_name: str,
    prompt_version: str,
    workflow_mode: str,
    graph_topology: str,
    trace_id: str | None,
    intent: str = "UNKNOWN",
) -> None:
    """Persist a PII-redacted failure candidate and count it in reliability metrics."""
    init_metrics_db()
    conn = sqlite3.connect(METRICS_DB_PATH, check_same_thread=False)
    timestamp = time.time()
    try:
        conn.execute("""
            INSERT INTO workflow_failures (
                thread_id, timestamp, query, failed_node, error_type,
                model_name, prompt_version, workflow_mode, graph_topology, trace_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            thread_id,
            timestamp,
            query,
            failed_node,
            error_type,
            model_name,
            prompt_version,
            workflow_mode,
            graph_topology,
            trace_id,
        ))
        conn.execute("""
            INSERT INTO execution_logs (
                thread_id, timestamp, intent, latency_seconds, token_cost,
                citation_mode, status, model_name, prompt_version,
                workflow_mode, graph_topology, trace_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            thread_id,
            timestamp,
            intent,
            0.0,
            0.0,
            "FAILED",
            "FAILED",
            model_name,
            prompt_version,
            workflow_mode,
            graph_topology,
            trace_id,
        ))
        conn.commit()
    finally:
        conn.close()

def get_metrics_summary() -> dict:
    """Aggregate all stored executions into latency, estimated cost, failure, and badge metrics.

    Latency percentiles are selected from sorted successful-turn samples; cost,
    request count, and failure rate include all logged rows. An empty database
    returns zero values and an empty citation-count mapping.
    """
    init_metrics_db()
    conn = sqlite3.connect(METRICS_DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("SELECT latency_seconds FROM execution_logs WHERE status='SUCCESS'")
    latencies = sorted([row[0] for row in cursor.fetchall()])
    
    cursor.execute("SELECT SUM(token_cost), COUNT(*) FROM execution_logs")
    total_cost_row = cursor.fetchone()
    total_cost = total_cost_row[0] or 0.0
    total_requests = total_cost_row[1] or 0
    
    cursor.execute("SELECT COUNT(*) FROM execution_logs WHERE status='FAILED'")
    failed_count = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT citation_mode, COUNT(*) FROM execution_logs GROUP BY citation_mode")
    citations = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    # Calculate p50 and p95 latency percentile thresholds
    p50, p95 = 0.0, 0.0
    if latencies:
        n = len(latencies)
        p50 = latencies[int(n * 0.5)]
        p95 = latencies[int(n * 0.95)] if n > 1 else latencies[-1]
        
    failure_rate = (failed_count / total_requests * 100) if total_requests > 0 else 0.0
    
    return {
        "p50_latency": round(p50, 3),
        "p95_latency": round(p95, 3),
        "total_cost": round(total_cost, 4),
        "total_requests": total_requests,
        "failure_rate": round(failure_rate, 2),
        "citations": citations
    }