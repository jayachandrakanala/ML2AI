# src/intent_router.py
import re
import json
from src.logger import log_status
from src.llm_provider import GROQ_MODEL, get_chat_model

def route_intent(query: str) -> str:
    """Choose a processing path by checking the query against keyword groups."""
    query_lower = query.lower()

    sc_keywords = [
        "supply chain", "logistics", "port congestion", "shipping delay",
        "supplier disruption", "carrier bankruptcy", "freight rerouting",
        "inventory bottleneck", "customs delay", "part shortage", "warehouse disruption",
        "stalled", "component", "microcontroller", "port", "transit", "shipping"
    ]
    graph_entity_keywords = [
        "tier 1", "tier 2", "tier 3", "bill of materials", "bom",
        "component shortage", "stock level", "assembly plant", "fulfillment hub",
        "single point of failure", "spof", "lineage", "multi-tier", "upstream supplier",
        "facility", "warehouse", "supplier", "product"
    ]
    impact_keywords = [
        "revenue at risk", "impacted order", "order value", "delivery due date",
        "order delay", "customer impact", "sla breach", "order fulfillment",
        "impacted", "customer accounts", "order ids", "financial exposure"
    ]
    mitigation_keywords = [
        "reroute", "re-route", "mitigation strategy", "alternative route",
        "intermodal", "air freight", "secondary supplier", "lead time variance",
        "gds", "dijkstra", "shortest path", "routing", "bypass"
    ]

    all_sc_patterns = sc_keywords + graph_entity_keywords + impact_keywords + mitigation_keywords
    if any(keyword in query_lower for keyword in all_sc_patterns):
        return "SUPPLY_CHAIN"

    if re.search(r"\bord-\d+\b", query_lower):
        return "SUPPLY_CHAIN"
    
    site_keywords = [
        "site selection", "data center", "commercial real estate", "investment location",
        "shopping complex", "shopping mall", "shopping center", "retail space",
        "commercial property", "real estate investment", "industrial park", "warehouse site",
        "real estate business", "real estate market", "property development", "buying property",
        "real estate in", "property in", "invest in real estate", "starting real estate",
        "real estate venture", "property business"
    ]
    if any(keyword in query_lower for keyword in site_keywords):
        return "SITE_SELECTION"
        
    web_keywords = ["latest", "news", "current", "weather", "today", "search", "price", "stock", "who is", "what is happening", "google", "online", "recent", "days", "week", "month"]
    if any(keyword in query_lower for keyword in web_keywords):
        return "WEB"
        
    local_keywords = ["laddu", "bullodu", "roadmap", "feature", "architecture", "assistant", "chat", "project", "doc", "document", "author", "phase", "stack", "implementation", "kanala"]
    if any(keyword in query_lower for keyword in local_keywords):
        return "RAG"
        
    return "LLM"

def extract_site_parameters(query: str, selected_model: str = GROQ_MODEL) -> tuple:
    """Extract the requested country and locations for a site-selection run dynamically with regional safeguards."""
    log_status("Analyzing user query for site selection parameters...")
    extraction_prompt = f"""
    You are a precise data extraction assistant for commercial real estate and data center site selection.
    Analyze the user query below and extract:
    1. "country": The target country or macro region (e.g., "Asia-Pacific", "Canada", "United States").
    2. "locations": A list of 3 specific major cities or data center hubs matching the user's requested region. Do NOT default to US hubs if an international region is requested.
    
    Return ONLY a valid JSON object with keys "country" (string) and "locations" (list of strings). No extra text.
    User Query: "{query}"
    """
    
    query_lower = query.lower()
    
    # 1. Determine intelligent defaults based on query content
    if "canada" in query_lower or "canadian" in query_lower:
        default_country = "Canada"
        default_locations = ["Montreal QC", "Toronto ON", "Calgary AB"]
    elif any(w in query_lower for w in ["asia", "asian", "singapore", "japan", "tokyo", "india", "mumbai", "seoul", "hong kong", "apac"]):
        default_country = "Asia-Pacific"
        default_locations = ["Singapore", "Tokyo Japan", "Mumbai India"]
    elif any(w in query_lower for w in ["europe", "uk", "germany", "london", "frankfurt", "dublin", "eu"]):
        default_country = "Europe"
        default_locations = ["Frankfurt Germany", "London UK", "Dublin Ireland"]
    else:
        default_country = "United States"
        default_locations = ["Northern Virginia", "Phoenix AZ", "Columbus OH"]

    try:
        response = get_chat_model(selected_model).invoke(extraction_prompt)
        content = response.content.strip()
        json_match = re.search(r'\{.*\}', content, raw:=True) if False else re.search(r'\{.*\}', content, re.DOTALL)
        
        if json_match:
            data = json.loads(json_match.group(0))
            country = data.get("country", default_country)
            locations = data.get("locations", default_locations)
            
            # 2. Safeguard Override: If the user asked for an international region but LLM hallucinated US hubs, override!
            us_defaults = {"Northern Virginia", "Phoenix AZ", "Columbus OH"}
            if set(locations).intersection(us_defaults) and default_country != "United States":
                log_status(f"⚠️ LLM returned U.S. defaults for non-U.S. query. Overriding with {default_country} hubs.")
                locations = default_locations
                country = default_country
                
            if not locations:
                locations = default_locations
            return country, locations
            
    except Exception as e:
        log_status(f"Extraction warning: {e}. Using intelligent regional fallback.")

    return (default_country, default_locations)