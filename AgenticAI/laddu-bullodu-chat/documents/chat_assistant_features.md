# Implemented chat assistant features

This note describes capabilities implemented in the Laddu Bullodu chat
assistant. It intentionally excludes common chatbot capabilities that are not
implemented in this project, such as voice interaction or image generation.

## User-facing capabilities

- Streamlit chat interface with a selectable chat model.
- Answers grounded in local Markdown/text documents through hybrid dense-vector
  and BM25 retrieval, BGE reranking, and a web-backed CRAG fallback when local
  relevance is low.
- Live web research for current questions.
- Supply-chain analysis using Neo4j and graph routing.
- Commercial real-estate and data-center site-selection research, including
  power, zoning/environment, incentives, and water-supply considerations.
- Optional autonomous multi-agent mode that delegates tasks to specialists,
  reviews combined findings with a critic, and synthesizes a final answer.
- PII sanitization, workflow progress display, execution metrics, and
  checkpoint-backed conversation/workflow resumption.

## Model choices

The UI supports Ollama, Groq, OpenAI, Google Gemini, and OpenRouter. Cloud
providers require their configured API keys; Ollama requires the local model
service. Model/provider availability and tool-calling support can vary by
provider.
