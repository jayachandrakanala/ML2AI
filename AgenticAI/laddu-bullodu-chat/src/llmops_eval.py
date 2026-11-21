"""Model-free golden-set regression checks for local LLM workflow behavior."""

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.hybrid_search import HybridRetriever
from src.intent_router import infer_site_selection_defaults, route_intent
from src.mlops import crag_fallback_required

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SET_PATH = PROJECT_ROOT / "evals" / "golden_cases.jsonl"


def load_golden_cases(path: Path = GOLDEN_SET_PATH) -> list[dict[str, Any]]:
    """Load and minimally validate the JSONL regression dataset."""
    cases = []
    seen_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(case, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        required = {"id", "category", "query", "expected_intent"}
        missing = required - case.keys()
        if missing:
            raise ValueError(
                f"{path}:{line_number}: missing fields: {', '.join(sorted(missing))}"
            )
        if case["id"] in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate case ID {case['id']!r}")
        if not isinstance(case["query"], str) or not case["query"].strip():
            raise ValueError(f"{path}:{line_number}: query must be non-empty text")
        if "crag_score" in case and "expected_crag" not in case:
            raise ValueError(
                f"{path}:{line_number}: crag_score requires expected_crag"
            )
        seen_ids.add(case["id"])
        cases.append(case)
    if not cases:
        raise ValueError(f"Golden set is empty: {path}")
    return cases


def _load_document_corpus() -> list[Document]:
    documents = []
    for path in sorted((PROJECT_ROOT / "documents").glob("*.md")):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name},
            )
        )
    if not documents:
        raise FileNotFoundError(f"No Markdown evaluation documents under {PROJECT_ROOT / 'documents'}")
    return documents


def _load_eval_config() -> dict[str, Any]:
    path = PROJECT_ROOT / "config" / "prompts_v1.json"
    configuration = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return configuration


def evaluate_golden_cases(
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score deterministic routing, BM25 source hits, regions, and CRAG decisions.

    These checks do not call a model. They are regression signals for system
    behavior, not substitutes for RAGAS or human semantic-quality evaluation.
    """
    cases = cases if cases is not None else load_golden_cases()
    configuration = _load_eval_config()
    expected_sources = [
        case
        for case in cases
        if case.get("expected_source") or case.get("expected_sources")
    ]
    if not expected_sources:
        raise ValueError("Golden set has no expected_source retrieval cases.")

    intent_matches = []
    region_matches = []
    crag_matches = []
    failures = []

    for case in cases:
        actual_intent = route_intent(case["query"])
        matched = actual_intent == case["expected_intent"]
        intent_matches.append(matched)
        if not matched:
            failures.append({
                "id": case["id"],
                "check": "intent",
                "expected": case["expected_intent"],
                "actual": actual_intent,
            })

        if case.get("expected_region"):
            actual_region, _ = infer_site_selection_defaults(case["query"])
            matched = actual_region == case["expected_region"]
            region_matches.append(matched)
            if not matched:
                failures.append({
                    "id": case["id"],
                    "check": "site_region",
                    "expected": case["expected_region"],
                    "actual": actual_region,
                })

        if "crag_score" in case:
            threshold = configuration["thresholds"]["crag_relevance_threshold"]
            actual_crag = crag_fallback_required(case["crag_score"], threshold)
            matched = actual_crag == case["expected_crag"]
            crag_matches.append(matched)
            if not matched:
                failures.append({
                    "id": case["id"],
                    "check": "crag_route",
                    "expected": case["expected_crag"],
                    "actual": actual_crag,
                })

    retrieval_matches = []
    if expected_sources:
        retriever = HybridRetriever(_load_document_corpus())
        top_k = configuration.get("llmops", {}).get("retrieval_eval_top_k", 3)
        for case in expected_sources:
            results = retriever.bm25_search(case["query"], k=top_k)
            actual_sources = [
                document.metadata.get("source") for document, _ in results
            ]
            expected = case.get("expected_sources", [case.get("expected_source")])
            matched = bool(set(expected) & set(actual_sources))
            retrieval_matches.append(matched)
            if not matched:
                failures.append({
                    "id": case["id"],
                    "check": "bm25_source",
                    "expected": expected,
                    "actual": actual_sources,
                })

    def ratio(matches: list[bool]) -> float:
        return sum(matches) / len(matches) if matches else 0.0

    scores = {
        "intent_accuracy": ratio(intent_matches),
        "bm25_source_hit_rate": ratio(retrieval_matches),
        "site_region_accuracy": ratio(region_matches),
        "crag_route_accuracy": ratio(crag_matches),
    }
    gates = configuration.get("llmops", {}).get("regression_gates", {})
    missing_gates = set(scores) - gates.keys()
    if missing_gates:
        raise ValueError(
            f"Missing regression gates: {', '.join(sorted(missing_gates))}"
        )
    invalid_gates = {
        metric: threshold
        for metric, threshold in gates.items()
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1
    }
    if invalid_gates:
        raise ValueError(f"Regression gates must be between 0 and 1: {invalid_gates}")
    failed_gates = {
        metric: {"score": score, "minimum": gates[metric]}
        for metric, score in scores.items()
        if score < gates[metric]
    }
    return {
        "case_count": len(cases),
        "sample_counts": {
            "intent": len(intent_matches),
            "bm25_source": len(retrieval_matches),
            "site_region": len(region_matches),
            "crag_route": len(crag_matches),
        },
        "scores": scores,
        "gates": gates,
        "passed": not failed_gates,
        "failed_gates": failed_gates,
        "failures": failures,
    }


def send_scores_to_phoenix(result: dict[str, Any]) -> None:
    """Export aggregate golden-set scores to the local Phoenix project."""
    from opentelemetry import trace
    from phoenix.otel import register

    configuration = _load_eval_config()
    provider = register(
        endpoint="http://localhost:6006/v1/traces",
        project_name="laddu-bullodu-chat",
        batch=True,
        verbose=False,
    )
    with trace.get_tracer("laddu-bullodu-chat.llmops").start_as_current_span(
        "llmops.golden_set_evaluation"
    ) as span:
        span.set_attribute("llmops.evaluation.case_count", result["case_count"])
        span.set_attribute(
            "llmops.evaluation.config_version",
            str(configuration.get("version", "unknown")),
        )
        span.set_attribute("llmops.evaluation.passed", result["passed"])
        for metric, score in result["scores"].items():
            span.set_attribute(f"llmops.evaluation.{metric}", score)
        for metric, minimum in result["gates"].items():
            span.set_attribute(f"llmops.evaluation.gate.{metric}", minimum)
    if not provider.force_flush(timeout_millis=10_000):
        raise RuntimeError("Phoenix did not confirm export of evaluation scores.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when any configured regression threshold fails.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optionally write the result JSON to this path.",
    )
    parser.add_argument(
        "--phoenix",
        action="store_true",
        help="Export aggregate evaluation scores to Phoenix at localhost:6006.",
    )
    args = parser.parse_args()

    result = evaluate_golden_cases()
    if args.phoenix:
        send_scores_to_phoenix(result)
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.json_out:
        args.json_out.write_text(output + "\n", encoding="utf-8")
    return 1 if args.check and not result["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
