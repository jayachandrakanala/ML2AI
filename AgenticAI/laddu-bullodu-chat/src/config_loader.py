# src/config_loader.py
import json
from pathlib import Path

CONFIG_PATH = Path("./config/prompts_v1.json")

def load_version_config() -> dict:
    """Loads versioned prompts and thresholds from external JSON config."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "version": "1.0.0",
        "thresholds": {"crag_relevance_threshold": 0.4, "top_k_retrieval": 20, "top_k_rerank": 3},
        "prompts": {
            "rag_system_prompt": "You are a helpful, professional AI assistant. Answer accurately using ONLY the verified re-ranked local context below.",
            "crag_fallback_prompt": "Local document retrieval yielded low confidence. Answer using live web search results below.",
            "web_system_prompt": "Answer accurately using the live web search results below."
        }
    }