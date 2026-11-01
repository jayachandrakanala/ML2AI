import os
import pytest
from dotenv import load_dotenv

load_dotenv()

from src.agent_graph import route_intent, app_graph
from src.supply_chain import supply_chain_graph

# ------------------------------------------------------------------------------
# 20 Complex Supply Chain Test Scenarios
# ------------------------------------------------------------------------------
TEST_SCENARIOS = [
    # Category 1: Revenue at Risk & Financial Disruption
    {
        "id": "TC-01",
        "category": "Financial Impact",
        "query": "What is our revenue at risk due to the current shipping delays at the Port of Los Angeles?",
        "expected_keywords": ["1,100,000", "ORD-8801", "ORD-8802", "Global Defense Systems", "Acme Retailers"]
    },
    {
        "id": "TC-02",
        "category": "Financial Impact",
        "query": "Which high-value customer accounts and order IDs are impacted by stalled A16 Microcontroller components?",
        "expected_keywords": ["ORD-8802", "ORD-8801", "A16 Microcontroller"]
    },
    {
        "id": "TC-03",
        "category": "Financial Impact",
        "query": "Calculate total value of pending orders tied to Tier 2 suppliers routing through congested ports.",
        "expected_keywords": ["Silicon Core Microchips", "Precision Lens Corp"]
    },

    # Category 2: Multi-Tier Bill of Materials (BOM) & Lineage
    {
        "id": "TC-04",
        "category": "BOM & Lineage",
        "query": "Show me the multi-tier bill of materials lineage for our enterprise security camera.",
        "expected_keywords": ["Enterprise Security Camera", "A16 Microcontroller", "4K Camera Sensor"]
    },
    {
        "id": "TC-05",
        "category": "BOM & Lineage",
        "query": "Trace the upstream Tier 1 and Tier 2 suppliers supplying components to the Texas Assembly Plant.",
        "expected_keywords": ["Texas Assembly Plant", "Supplier", "Tier"]
    },
    {
        "id": "TC-06",
        "category": "BOM & Lineage",
        "query": "Which specific finished products will experience production stoppage if 4K Camera Sensor Module inventory drops to zero?",
        "expected_keywords": ["4K Camera Sensor Module", "Enterprise Security Camera"]
    },

    # Category 3: Single Point of Failure (SPOF) & Bottlenecks
    {
        "id": "TC-07",
        "category": "Bottleneck Analysis",
        "query": "Identify single point of failure suppliers in our supply chain network supplying single-source components.",
        "expected_keywords": ["Supplier", "Component", "Single Point of Failure"]
    },
    {
        "id": "TC-08",
        "category": "Bottleneck Analysis",
        "query": "Which warehouse fulfillment centers are currently storing products at risk due to port congestion?",
        "expected_keywords": ["Warehouse", "STORED_AT", "FULFILLS"]
    },
    {
        "id": "TC-09",
        "category": "Bottleneck Analysis",
        "query": "Identify all ports currently flagged as CONGESTED and their connected downstream facilities.",
        "expected_keywords": ["Port of Los Angeles", "CONGESTED", "Facility"]
    },

    # Category 4: GDS Pathfinding & Intermodal Rerouting
    {
        "id": "TC-10",
        "category": "GDS Routing",
        "query": "Run a GDS Dijkstra shortest path calculation from Port of Shenzhen to Texas Assembly Plant using transit and cost weights.",
        "expected_keywords": ["GDS", "Dijkstra", "Port of Shenzhen", "Texas Assembly Plant"]
    },
    {
        "id": "TC-11",
        "category": "GDS Routing",
        "query": "What is the optimal alternative shipping route to bypass the Port of Los Angeles bottleneck?",
        "expected_keywords": ["Optimal Path", "Routing", "Bypass", "Feasibility"]
    },
    {
        "id": "TC-12",
        "category": "GDS Routing",
        "query": "Compare lead time scores between default sea routing versus alternate intermodal routes.",
        "expected_keywords": ["Lead Time", "Score", "Transit"]
    },

    # Category 5: Mitigation Strategies & Feasibility Trade-offs
    {
        "id": "TC-13",
        "category": "Mitigation Strategy",
        "query": "What are the cost variances and operational feasibility scores for emergency air freight versus secondary regional suppliers?",
        "expected_keywords": ["Air Freight", "Regional Supplier", "Cost Variance", "Feasibility"]
    },
    {
        "id": "TC-14",
        "category": "Mitigation Strategy",
        "query": "Give me an executive recommendation on which shipping strategy to execute to protect high-value orders.",
        "expected_keywords": ["Recommendation", "Execute", "Strategy"]
    },
    {
        "id": "TC-15",
        "category": "Mitigation Strategy",
        "query": "Evaluate carbon footprint and lead time trade-offs across all available rerouting options.",
        "expected_keywords": ["Lead Time", "Carbon", "Trade-offs"]
    },

    # Category 6: Order SLA Breaches & Due Date Risk
    {
        "id": "TC-16",
        "category": "SLA & Orders",
        "query": "Which customer orders have delivery due dates in November 2026 that are at risk of SLA breach?",
        "expected_keywords": ["2026-11", "Order", "DueDate"]
    },
    {
        "id": "TC-17",
        "category": "SLA & Orders",
        "query": "Check order status and delivery timeline for customer Acme Retailers under order ORD-8801.",
        "expected_keywords": ["ORD-8801", "Acme Retailers", "250,000"]
    },

    # Category 7: Domain Explanations & Conceptual Reasoning
    {
        "id": "TC-18",
        "category": "Conceptual Reasoning",
        "query": "Explain how our multi-tier supply chain graph models relationships between ports, suppliers, components, and orders.",
        "expected_keywords": ["Graph", "Supplier", "Port", "Facility", "Order"]
    },
    {
        "id": "TC-19",
        "category": "Conceptual Reasoning",
        "query": "How does Neo4j GDS Dijkstra algorithm help mitigate supply chain disruptions compared to static reports?",
        "expected_keywords": ["Dijkstra", "Algorithm", "Pathfinding", "Weighted"]
    },
    {
        "id": "TC-20",
        "category": "Conceptual Reasoning",
        "query": "Draft an executive summary memo outlining our supply chain risk profile and recommended actions.",
        "expected_keywords": ["Executive", "Summary", "Risk", "Mitigation"]
    }
]


# ------------------------------------------------------------------------------
# Test 1: Intent Classifier Validation across all 20 questions
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", TEST_SCENARIOS, ids=[s["id"] for s in TEST_SCENARIOS])
def test_intent_routing_classification(scenario):
    """Verify that all 20 complex supply chain prompts are routed to SUPPLY_CHAIN intent."""
    query = scenario["query"]
    intent = route_intent(query)
    assert intent == "SUPPLY_CHAIN", (
        f"[{scenario['id']}] Failed intent routing for query: '{query}'. "
        f"Got '{intent}' instead of 'SUPPLY_CHAIN'."
    )


# ------------------------------------------------------------------------------
# Test 2: Sub-Graph Graph Intelligence & LLM Reasoning Execution
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", TEST_SCENARIOS, ids=[s["id"] for s in TEST_SCENARIOS])
def test_supply_chain_subgraph_reasoning(scenario):
    """Verify that the LangGraph sub-graph executes Cypher generation, GDS pathfinding, and LLM reasoning."""
    query = scenario["query"]
    
    result = supply_chain_graph.invoke({
        "query": query,
        "selected_model": "llama3.2"
    })
    
    memo = result.get("final_mitigation_memo", "")
    assert memo, f"[{scenario['id']}] Supply chain graph returned an empty response."
    
    # Assert that at least one key domain term is present in the LLM output
    matched_keywords = [kw for kw in scenario["expected_keywords"] if kw.lower() in memo.lower()]
    assert len(matched_keywords) > 0, (
        f"[{scenario['id']}] Response failed reasoning assertion.\n"
        f"Expected at least one of {scenario['expected_keywords']}\n"
        f"Received Output:\n{memo}"
    )


# ------------------------------------------------------------------------------
# Test 3: End-to-End Agent Workflow & Guardrail Check
# ------------------------------------------------------------------------------
def test_end_to_end_financial_impact_workflow():
    """Perform a full end-to-end integration test through app_graph for the flagship revenue prompt."""
    flagship_prompt = "What is our revenue at risk due to the current shipping delays at the Port of Los Angeles?"
    
    result = app_graph.invoke({
        "messages": [("user", flagship_prompt)],
        "selected_model": "llama3.2"
    })
    
    final_response = result["messages"][-1].content
    
    # Assert intent badge formatting
    assert "[SUPPLY_CHAIN]" in final_response, "Response missing required [SUPPLY_CHAIN] badge prefix."
    
    # Assert exact financial synthesis figure
    assert "1,100,000" in final_response or "1100000" in final_response, (
        "End-to-end response missing expected total revenue at risk ($1,100,000)."
    )
    assert "ORD-8802" in final_response, "Response missing impacted order ORD-8802."
    assert "ORD-8801" in final_response, "Response missing impacted order ORD-8801."