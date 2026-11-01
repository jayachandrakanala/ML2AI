import os
import json
from typing import TypedDict, List, Dict, Any
try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

from langchain_neo4j import Neo4jGraph
from langgraph.graph import StateGraph, START, END
from src.logger import log_status
from src.llm_provider import get_chat_model, OLLAMA_MODEL

# Initialize Neo4j Connection using environment variables
neo4j_graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
    refresh_schema=False
)

class SupplyChainState(TypedDict):
    query: str
    selected_model: NotRequired[str]
    generated_cypher: str
    graph_context: str
    gds_routing_data: List[Dict[str, Any]]
    final_mitigation_memo: str

def text_to_cypher_node(state: SupplyChainState) -> Dict[str, Any]:
    """Uses the selected LLM to dynamically generate accurate Cypher based on user query and schema examples."""
    user_query = state["query"]
    selected_model = state.get("selected_model", OLLAMA_MODEL)
    log_status(f"LLM generating dynamic Cypher query for user prompt: '{user_query}'...")

    schema = """
    Nodes & Properties:
    - Port {id: 'PORT-LA', name: 'Port of Los Angeles', status: 'CONGESTED'}
    - Supplier {id, name, tier}
    - Component {id, name, stockLevel}
    - Product {id, name, retailPrice}
    - Facility {id, name, city, state}
    - Warehouse {id, name, city, state}
    - Order {id, customer, orderValue, status, dueDate}

    Exact Relationship Directions:
    - (Supplier)-[:SHIPS_VIA]->(Port)
    - (Port)-[:ROUTES_TO]->(Port)
    - (Port)-[:ALTERNATE_ROUTE]->(Port)
    - (Port)-[:DELIVERS_TO]->(Facility)
    - (Supplier)-[:SUPPLIES]->(Component)
    - (Component)-[:REQUIRED_FOR]->(Product)
    - (Facility)-[:MANUFACTURES]->(Product)
    - (Product)-[:STORED_AT]->(Warehouse)
    - (Warehouse)-[:FULFILLS]->(Order)
    """

    prompt = f"""You are an expert Cypher database query generator for a supply chain graph database.
Given the schema and examples below, generate a valid Cypher query to answer the user's question.
Return ONLY the raw Cypher query inside a markdown code block (e.g., ```cypher ... ```). Do not include any explanations.

Rules:
1. Always use `toLower(c.name) CONTAINS 'keyword'` when matching components (e.g., 'microcontroller', 'sensor', 'lens').
2. Do NOT try to connect Component directly to Port. Traverse through Facility and Product: (Port)-[:DELIVERS_TO]->(Facility)-[:MANUFACTURES]->(Product)<-[:REQUIRED_FOR]-(Component).

Schema & Relationships:
{schema}

Example 1 (Port congestion and revenue at risk):
Question: "What is our revenue at risk due to shipping delays at the Port of Los Angeles?"
Cypher:
MATCH (p:Port) WHERE p.id = 'PORT-LA' OR toUpper(p.status) = 'CONGESTED'
MATCH (p)-[:DELIVERS_TO*1..2]->(f:Facility)-[:MANUFACTURES]->(prod:Product)-[:STORED_AT]->(dc:Warehouse)-[:FULFILLS]->(o:Order)
RETURN p.name AS Port, o.id AS OrderID, o.customer AS Customer, o.orderValue AS RevenueAtRisk

Example 2 (Trace Tier 2 suppliers through components, products, and orders):
Question: "Trace the complete supply chain path from our Tier 2 suppliers through stalled components and assembly facilities to impacted customer deliverables."
Cypher:
MATCH (s:Supplier)-[:SUPPLIES]->(c:Component)-[:REQUIRED_FOR]->(prod:Product)<-[:MANUFACTURES]-(f:Facility)
MATCH (prod)-[:STORED_AT]->(dc:Warehouse)-[:FULFILLS]->(o:Order)
RETURN s.name AS Tier2Supplier, s.tier AS SupplierTier, c.name AS StalledComponent, f.name AS Facility, prod.name AS AffectedProduct, o.id AS ImpactedOrder, o.customer AS Customer, o.orderValue AS RevenueAtRisk
ORDER BY RevenueAtRisk DESC

Example 3 (Specific component impact at a congested port):
Question: "How do we handle the port congestion at the Port of Los Angeles affecting our microcontrollers?"
Cypher:
MATCH (c:Component) WHERE toLower(c.name) CONTAINS 'microcontroller'
MATCH (c)-[:REQUIRED_FOR]->(prod:Product)<-[:MANUFACTURES]-(f:Facility)<-[:DELIVERS_TO]-(p:Port)
WHERE p.id = 'PORT-LA' OR toUpper(p.status) = 'CONGESTED'
MATCH (prod)-[:STORED_AT]->(dc:Warehouse)-[:FULFILLS]->(o:Order)
RETURN p.name AS Port, c.name AS Component, prod.name AS Product, o.id AS OrderID, o.customer AS Customer, o.orderValue AS RevenueAtRisk

User Question: {user_query}
"""

    try:
        chat_model = get_chat_model(selected_model)
        response = chat_model.invoke(prompt)
        content = response.content.strip()
        
        if "```cypher" in content:
            cypher = content.split("```cypher")[1].split("```")[0].strip()
        elif "```" in content:
            cypher = content.split("```")[1].split("```")[0].strip()
        else:
            cypher = content.strip()
    except Exception as e:
        log_status(f"LLM Cypher generation warning: {e}. Using fallback query.")
        cypher = """
        MATCH (s:Supplier)-[:SUPPLIES]->(c:Component)-[:REQUIRED_FOR]->(prod:Product)<-[:MANUFACTURES]-(f:Facility)<-[:DELIVERS_TO]-(p:Port)
        MATCH (prod)-[:STORED_AT]->(dc:Warehouse)-[:FULFILLS]->(o:Order)
        RETURN p.name AS Port, s.name AS Supplier, c.name AS Component, prod.name AS Product, o.id AS OrderID, o.customer AS Customer, o.orderValue AS RevenueAtRisk
        """

    log_status(f"Generated Cypher: {cypher}")
    
    try:
        results = neo4j_graph.query(cypher)
        context_str = json.dumps(results, indent=2) if results else "No matching data found in graph."
    except Exception as e:
        log_status(f"Cypher execution failed: {e}. Executing fallback graph query.")
        fallback_cypher = """
        MATCH (s:Supplier)-[:SUPPLIES]->(c:Component)-[:REQUIRED_FOR]->(prod:Product)<-[:MANUFACTURES]-(f:Facility)<-[:DELIVERS_TO]-(p:Port)
        MATCH (prod)-[:STORED_AT]->(dc:Warehouse)-[:FULFILLS]->(o:Order)
        RETURN p.name AS Port, s.name AS Supplier, c.name AS Component, prod.name AS Product, o.id AS OrderID, o.customer AS Customer, o.orderValue AS RevenueAtRisk
        """
        results = neo4j_graph.query(fallback_cypher)
        context_str = json.dumps(results, indent=2)

    log_status("Neo4j graph intelligence data retrieved successfully.")
    return {"generated_cypher": cypher, "graph_context": context_str}

def calculate_gds_optimal_route(state: SupplyChainState) -> Dict[str, Any]:
    """Dedicated Node: Executes Neo4j GDS Dijkstra shortest path calculation using composite weights."""
    log_status("Running GDS Dijkstra weighted pathfinding on in-memory graph projection...")
    
    gds_routes = []
    try:
        # Setup composite weight & ensure projection
        neo4j_graph.query("""
        MATCH ()-[r]->()
        WHERE r.transitDays IS NOT NULL
        SET r.compositeWeight = toFloat(r.transitDays) + (coalesce(r.costMultiplier, 1.0) * 4.0)
        """)

        # Drop existing projection if it exists to refresh properties
        neo4j_graph.query("CALL gds.graph.drop('supplyChainNetwork', false) YIELD graphName")

        # Project fresh graph including compositeWeight
        neo4j_graph.query("""
        CALL gds.graph.project(
            'supplyChainNetwork',
            { 
              Port: {label: 'Port'}, 
              Supplier: {label: 'Supplier'}, 
              Facility: {label: 'Facility'}, 
              Warehouse: {label: 'Warehouse'} 
            },
            {
              ROUTES_TO: {type: 'ROUTES_TO', properties: ['compositeWeight', 'transitDays'], orientation: 'NATURAL'},
              DELIVERS_TO: {type: 'DELIVERS_TO', properties: ['compositeWeight', 'transitDays'], orientation: 'NATURAL'},
              ALTERNATE_ROUTE: {type: 'ALTERNATE_ROUTE', properties: ['compositeWeight', 'transitDays'], orientation: 'NATURAL'},
              SHIPS_VIA: {type: 'SHIPS_VIA', properties: ['compositeWeight', 'transitDays'], orientation: 'NATURAL'}
            }
        )
        """)

        dijkstra_cypher = """
        MATCH (source:Port {id: 'PORT-SZ'}), (target:Facility {id: 'FAC-US'})
        CALL gds.shortestPath.dijkstra.stream('supplyChainNetwork', {
            sourceNode: source,
            targetNode: target,
            relationshipWeightProperty: 'compositeWeight'
        })
        YIELD totalCost, nodeIds
        RETURN 
            totalCost AS optimizedScore,
            [nodeId IN nodeIds | gds.util.asNode(nodeId).name] AS optimalPathNodes
        """
        results = neo4j_graph.query(dijkstra_cypher)
        if results:
            res = results[0]
            gds_routes.append({
                "strategy": "GDS Optimized Routing (Dijkstra: Transit + Cost)",
                "optimized_score": res.get("optimizedScore"),
                "path": res.get("optimalPathNodes"),
                "feasibility": "High"
            })
    except Exception as e:
        log_status(f"GDS execution warning: {e}")
        gds_routes.append({"strategy": "Fallback Intermodal Rerouting", "feasibility": "Medium"})

    return {"gds_routing_data": gds_routes}


def llm_reasoning_synthesizer(state: SupplyChainState) -> Dict[str, Any]:
    """Synthesizer Node: Uses the LLM to reason over graph context and write a tailored response."""
    log_status("LLM synthesizing tailored response based on retrieved graph intelligence...")
    
    user_query = state["query"]
    selected_model = state.get("selected_model", OLLAMA_MODEL)
    graph_context = state.get("graph_context", "[]")
    gds_data = state.get("gds_routing_data", [])

    prompt = f"""You are an expert Supply Chain & Graph AI Architect assistant.
Answer the user's specific question directly, thoroughly, and professionally using ONLY the retrieved Neo4j Graph Data and GDS Route Analytics provided below.

User Question: "{user_query}"

Retrieved Neo4j Graph Data:
{graph_context}

GDS Dijkstra Route Analytics:
{json.dumps(gds_data, indent=2)}

Instructions:
1. Address the user's prompt directly in your opening response.
2. If the user asks about revenue at risk, calculate and summarize the total financial exposure from the order records.
3. If the user asks about general concepts or explanations, explain how the multi-tier supply chain graph connects suppliers, ports, facilities, and orders.
4. Format your findings clearly using Markdown headers and bullet points. Do not output raw JSON blobs blindly; integrate the values naturally into your professional reasoning.
"""

    try:
        chat_model = get_chat_model(selected_model)
        response = chat_model.invoke(prompt)
        memo = response.content
    except Exception as e:
        log_status(f"LLM synthesis warning: {e}")
        memo = f"Supply chain analysis completed successfully for query: {user_query}\n\nRetrieved Context:\n{graph_context}"

    return {"final_mitigation_memo": memo}


# Build 4-Node StateGraph Workflow with LLM Reasoning
sc_builder = StateGraph(SupplyChainState)
sc_builder.add_node("text_to_cypher_node", text_to_cypher_node)
sc_builder.add_node("calculate_gds_optimal_route", calculate_gds_optimal_route)
sc_builder.add_node("llm_reasoning_synthesizer", llm_reasoning_synthesizer)

sc_builder.add_edge(START, "text_to_cypher_node")
sc_builder.add_edge("text_to_cypher_node", "calculate_gds_optimal_route")
sc_builder.add_edge("calculate_gds_optimal_route", "llm_reasoning_synthesizer")
sc_builder.add_edge("llm_reasoning_synthesizer", END)

supply_chain_graph = sc_builder.compile()