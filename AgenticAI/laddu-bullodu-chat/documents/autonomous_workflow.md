# Autonomous multi-agent workflow

The assistant's optional autonomous mode is implemented as a separate LangGraph
workflow in `src/autonomous_graph.py`. The default intent-routed graph remains
separate and unchanged.

## Routing and execution

An initialization step stores the latest user request as the goal and resets the
worker findings and run counters. A supervisor uses the selected tool-calling
chat model to choose one of the available actions: delegate to internal-document
RAG, current web search, the Neo4j supply-chain specialist, or commercial/site
selection; it can also choose to finish. Explicit tasks such as current weather,
site selection, questions about implemented assistant features, and questions
about this project's autonomous workflow require their matching specialist
rather than relying only on the model's choice. Project-document questions read
checked-in feature/workflow documents directly so answers do not depend on
embedding similarity or a web fallback.

Each worker reuses the corresponding existing project workflow. The RAG worker
combines dense vector retrieval with BM25, reranks results with BGE, and uses
the configured relevance threshold to choose local-context generation or
web-backed Corrective RAG (CRAG). Web search is used for current public facts;
project architecture explanations should be grounded in the local project
documents.

## Review and completion

Worker results are recorded with the specialist name, assigned query, and
sanitized output. After a worker runs, the critic checks the full set of
findings against every part of the original goal. A worker may cover only one
subtask; the review should not reject a multi-part answer merely because the
latest worker did not answer a different subtask already covered by an earlier
finding. If the combined evidence is incomplete, critic feedback is sent to
the supervisor, which can delegate another specialist or refine a query. The
site-selection specialist is a fixed research-and-report pipeline, so it runs
at most once per request; the final answer calls out any site-selection evidence
gaps rather than rerunning the same pipeline and producing conflicting reports.

The workflow is bounded to four specialist executions and five supervisor
turns per user request. The finalizer synthesizes the available findings into a
single answer, then the shared output guardrail sanitizes the response and
records execution metrics. Autonomous checkpoints are stored separately from
the default workflow's checkpoints.
