# src/metrics_logger.py
import sqlite3
import time
from pathlib import Path

METRICS_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "metrics.sqlite")
Path(METRICS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

def init_metrics_db():
    """Initializes the SQLite metrics telemetry database."""
    conn = sqlite3.connect(METRICS_DB_PATH, check_same_thread=False)
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
            status TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_execution(thread_id: str, intent: str, latency_seconds: float, token_cost: float, citation_mode: str, status: str = "SUCCESS"):
    """Logs a completed graph execution turn with latency, cost, and citation telemetry."""
    init_metrics_db()
    conn = sqlite3.connect(METRICS_DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO execution_logs (thread_id, timestamp, intent, latency_seconds, token_cost, citation_mode, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (thread_id, time.time(), intent, latency_seconds, token_cost, citation_mode, status))
    conn.commit()
    conn.close()

def get_metrics_summary() -> dict:
    """Computes p50/p95 latency, total cost, failure rates, and citation distributions."""
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