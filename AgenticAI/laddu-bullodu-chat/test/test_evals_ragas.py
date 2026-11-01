import os
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.llm_provider import GROQ_MODEL, MODEL_LABELS, get_chat_model
from src.local_embeddings import LocalEmbeddingGemmaEmbeddings

# Import your compiled LangGraph application
from src.agent_graph import app_graph, load_embeddings_and_vector_store

# 10 Diverse Evaluation Questions covering all system functionalities
EVAL_DATASET = [
    {
        "user_input": "Find exact records for order ORD-8801 and Port of Los Angeles throughput capacity.",
        "reference": "Order ORD-8801 is processed via the Port of Los Angeles with a constrained throughput capacity of 1,200 TEUs daily."
    },
    {
        "user_input": "What is our total revenue at risk and which customer orders are impacted by the port congestion at the Port of Los Angeles affecting our microcontrollers?",
        "reference": "Revenue at risk totals $1,100,000 impacting customer orders ORD-8801 and ORD-8802 via A16 Microcontrollers flowing through the Port of Los Angeles and Texas Assembly Plant."
    },
    {
        "user_input": "Trace the complete supply chain path from our Tier 2 suppliers through stalled components and assembly facilities to impacted customer deliverables.",
        "reference": "Neo4j graph traversal maps Tier 2 suppliers through component assembly facilities to warehouse distribution hubs and customer orders."
    },
    {
        "user_input": "Evaluate the top commercial real estate and data center site options in Northern Virginia for a new high-density cloud infrastructure deployment.",
        "reference": "Northern Virginia provides premier data center infrastructure with robust power grid capacity, high fiber density, and tax incentives."
    },
    {
        "user_input": "What are the key target locations and market conditions for commercial property investments in London and the United Kingdom?",
        "reference": "Key target markets include London, Slough, and Dublin, driven by high enterprise cloud demand and strong connectivity hubs."
    },
    {
        "user_input": "What are the core architecture and multi-agent workflow features of the Laddu Bullodu project?",
        "reference": "Laddu Bullodu is a Python Streamlit and LangGraph application featuring modular stateful nodes, ChromaDB vector search, and BGE cross-encoder re-ranking."
    },
    {
        "user_input": "Explain how the BGE Cross-Encoder re-ranker and Corrective RAG (CRAG) thresholding operate within our workflow.",
        "reference": "The system pulls 20 chunks, re-ranks them using BAAI/bge-reranker-large, and triggers a web search fallback if the relevance score drops below 0.4."
    },
    {
        "user_input": "What are the latest developments and news in artificial intelligence agents this week?",
        "reference": "Live web search results via Serper.dev provide real-time updates regarding agentic architecture updates and enterprise implementations."
    },
    {
        "user_input": "Search for current weather forecasts and recent economic market updates globally.",
        "reference": "Live web search pulls real-time weather and financial market intelligence using the Google Serper API wrapper."
    },
    {
        "user_input": "Explain the architectural differences between SAML 2.0 and OIDC authentication flows in zero-trust enterprise networks.",
        "reference": "SAML 2.0 relies on XML assertions exchanged via browser redirects, whereas OIDC is built on top of OAuth 2.0 using JSON Web Tokens (JWT)."
    },
    {
        "user_input": "Describe the security compliance benchmarks and Center for Internet Security (CIS) hardening standards used in enterprise systems.",
        "reference": "CIS benchmarks provide prescriptive configuration guidelines to secure operating systems, cloud environments, and middleware layers against vulnerabilities."
    }
]

def run_assistant_pipeline(query: str):
    """Invokes the LangGraph app_graph and extracts output content and retrieved contexts."""
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "selected_model": GROQ_MODEL
    }
    
    # Invoke the compiled state graph
    result = app_graph.invoke(initial_state)
    last_message = result["messages"][-1]
    answer_text = getattr(last_message, "content", str(last_message))
    
    # Extract contexts if stored in state (fallback to empty list if handled via web/LLM/graph)
    retrieved_contexts = [doc.page_content for doc in result.get("top_docs", [])]
    if not retrieved_contexts and "graph_context" in result:
        retrieved_contexts = [result["graph_context"]]
    elif not retrieved_contexts:
        retrieved_contexts = ["Context retrieved dynamically via tool execution or direct reasoning."]
        
    return answer_text, retrieved_contexts

def evaluate_offline():
    print("🚀 Starting Offline RAGAS Evaluation Suite (10 Scenarios)...")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in EVAL_DATASET:
        q = item["user_input"]
        gt = item["reference"]
        print(f"\nEvaluating Query: '{q}'")
        
        ans, ctx = run_assistant_pipeline(q)
        
        questions.append(q)
        answers.append(ans)
        contexts.append(ctx)
        ground_truths.append(gt)

    # Format into a Hugging Face Dataset expected by RAGAS standard schema
    eval_data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }
    dataset = Dataset.from_dict(eval_data)
    
    # Wrap Groq LLM and Local Embeddings for RAGAS evaluation compatibility
    evaluator_llm = LangchainLLMWrapper(get_chat_model(GROQ_MODEL))
    evaluator_embeddings = LangchainEmbeddingsWrapper(LocalEmbeddingGemmaEmbeddings())
    
    # Initialize metric objects explicitly with judge LLM and Embeddings
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]
    
    print("\nRunning RAGAS evaluation metrics calculation...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )
    
    df = result.to_pandas()
    print("\n📊 RAGAS Evaluation Summary Results:")
    print(df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])
    
    # Save score report locally
    df.to_csv("ragas_evaluation_report.csv", index=False)
    print("\n📁 Full evaluation report saved to: ragas_evaluation_report.csv")

if __name__ == "__main__":
    evaluate_offline()