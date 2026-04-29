import logging
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from src.agents.crawler import perform_recursive_extraction
from src.agents.strategist import query_graph, generate_report
from src.llm import ASI1LLM

logger = logging.getLogger(__name__)
routing_llm = ASI1LLM(model="asi1", temperature=0.1)

class ResearchState(TypedDict):
    user_input: str
    seed_id: Optional[str]        # Clean ArXiv ID extracted by Architect
    oa_seed_id: Optional[str]     # Resolved OpenAlex ID (e.g. oa:W1234)
    intent: Optional[str]
    seed_query: Optional[str]
    question: Optional[str]
    output: Optional[str]
    format: str
    search_count: int
    cache_hit: bool               # True when paper was already in Memgraph
    paper_history: List[str]      # ArXiv IDs seen in this conversation
    history: List[str]

def extract_paper_id(text: str) -> Optional[str]:
    """
    Normalise any paper reference into a canonical ID (ArXiv or DOI).
    Handles:
      - ArXiv URLs and IDs (e.g. 2401.00001)
      - Standard DOIs (e.g. 10.1145/1234567.1234567)
      - DOI URLs (e.g. https://doi.org/10.1145/...)
    """
    import re
    patterns = [
        r'arxiv\.org/(?:abs|pdf)/([\d]{4}\.[\d]{4,5})',    # ArXiv URLs
        r'arxiv:([\d]{4}\.[\d]{4,5})',                       # arxiv: prefix
        r'\b([\d]{4}\.[\d]{4,5})(?:v\d+)?\b',               # bare ArXiv ID
        r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)',                  # Standard DOI
        r'doi\.org/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)',         # DOI URL
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace('.pdf', '')
            # Clean up trailing punctuation often caught in regex
            if val.endswith('.'): val = val[:-1]
            return val
    return None

# ─────────────────────────────────────────────────────────────
# Node definitions FIRST — build_workflow references them below
# ─────────────────────────────────────────────────────────────

def architect_node(state: ResearchState) -> ResearchState:
    """Routes the user request to the right pipeline."""
    from src.database import db
    user_input = state["user_input"]
    paper_history = state.get("paper_history", [])

    # Strip @mention prefix injected by the front-end
    if "@" in user_input:
        parts = user_input.split(None, 1)
        user_input = parts[1] if len(parts) > 1 else user_input
    state["user_input"] = user_input

    # ── Intelligent LLM Routing ─────────────────────────────────────
    prompt = f"""
    You are the R2E2 Architect. Analyze the user input and session history to decide the next action.
    
    Current Research Seed: {paper_history[-1] if paper_history else "None"}
    User Input: "{user_input}"
    
    CLASSIFICATION RULES:
    - GREETING: User is saying hello, hi, or asking who you are/what you do.
    - NEW_RESEARCH: User provided a NEW link, ArXiv ID, DOI, or a completely new technical topic to explore.
    - FOLLOW_UP_QA: User is asking a question about the current research, specific references, methodology, or conceptual gaps in the existing graph.
    - GENERATE_REPORT: User specifically asked for a summary, roadmap, or report of what we found.
    - PURGE: User wants to reset the system.
    
    Respond ONLY with the classification name.
    """
    intent = routing_llm.invoke(prompt).strip().upper()
    
    # ── Resolve IDs based on AI Intent ──────────────────────────────
    if "GREETING" in intent:
        state["intent"] = "GREETING"
        state["output"] = "Hello! I am your Deep Research Architect. I specialize in high-density technical analysis of academic papers from ArXiv and ScienceDirect. I can map out citation networks, perform deep-reads of methodologies, and identify systemic gaps in research fields.\n\nHow can I assist your research today? You can provide a paper URL (ArXiv/DOI) or ask a question about your current graph."
        return state

    if "NEW_RESEARCH" in intent:
        paper_id = extract_paper_id(user_input)
        if paper_id:
            # Check if we already have it
            existing_oa_id = db.find_paper_by_arxiv_id(paper_id)
            if existing_oa_id:
                logger.info(f"New research requested for {paper_id}, but we have it. Generating report.")
                state["intent"] = "GENERATE_REPORT"
                state["oa_seed_id"] = existing_oa_id
            else:
                state["intent"] = "TRIGGER_CRAWLER"
                state["seed_id"] = paper_id
            
            if paper_id not in paper_history:
                paper_history.append(paper_id)
        else:
            # Topic-based search
            state["intent"] = "TRIGGER_CRAWLER"
            state["seed_id"] = user_input
            
    elif "FOLLOW_UP_QA" in intent:
        state["intent"] = "TRIGGER_STRATEGIST"
        if paper_history:
            last_paper = paper_history[-1]
            state["oa_seed_id"] = db.find_paper_by_arxiv_id(last_paper) or f"oa:{last_paper}"
            logger.info(f"AI classified as Follow-up. Scoping to: {state['oa_seed_id']}")

    elif "GENERATE_REPORT" in intent:
        state["intent"] = "GENERATE_REPORT"
        if paper_history:
            last_paper = paper_history[-1]
            state["oa_seed_id"] = db.find_paper_by_arxiv_id(last_paper) or f"oa:{last_paper}"

    elif "PURGE" in intent:
        state["intent"] = "PURGE"

    state["paper_history"] = paper_history
    state["format"] = "pdf" if "pdf" in user_input.lower() else "markdown"
    return state


def crawler_node(state: ResearchState) -> ResearchState:
    # Use the pre-processed clean ID; fall back to raw input if not available
    seed = state.get("seed_id") or state["user_input"]
    logger.info(f"Crawler Node — seed ID: {seed}")
    state["search_count"] = state.get("search_count", 0) + 1
 
    oa_id, res = perform_recursive_extraction(seed)
    state["oa_seed_id"] = oa_id
    state["output"] = res
    state["intent"] = "GENERATE_REPORT"
    return state


def strategist_node(state: ResearchState) -> ResearchState:
    user_input = state["user_input"]
    # Prioritize canonical OA ID for cluster scoping
    seed_id = state.get("oa_seed_id") or state.get("seed_id")
    logger.info(f"Strategist: Graph-First Q&A... Scoped seed: {seed_id}")

    response = query_graph(user_input, seed_id=seed_id)
    if "SIGNAL:MISSING_DATA_FOR_SEARCH" in response:
        logger.warning("Missing data — triggering targeted crawl.")
        state["intent"] = "TARGETED_SEARCH"
    else:
        state["output"] = response
        state["intent"] = ""   # clear intent so route → END
    return state


def report_node(state: ResearchState) -> ResearchState:
    fmt = state.get("format", "markdown")
    # Prioritize canonical OA ID for cluster scoping
    seed_id = state.get("oa_seed_id") or state.get("seed_id")
    logger.info(f"Report Node: generating {fmt} report... Scoped seed: {seed_id}")
    state["output"] = generate_report(format=fmt, seed_id=seed_id)
    return state


def purge_node(state: ResearchState) -> ResearchState:
    from src.database import db
    db.manual_purge()
    state["output"] = "Database purged successfully."
    return state


# ─────────────────────────────────────────────────────────────
# Router — called after every node to decide next step
# ─────────────────────────────────────────────────────────────

def route_workflow(state: ResearchState) -> str:
    intent = state.get("intent", "")
    if "PURGE" in intent:
        return "purge_node"
    if "TRIGGER_CRAWLER" in intent or intent == "TARGETED_SEARCH":
        if state.get("search_count", 0) > 2:
            return "report_node"
        return "crawler_node"
    if "GENERATE_REPORT" in intent:
        return "report_node"
    if "TRIGGER_STRATEGIST" in intent:
        return "strategist_node"
    return END


# ─────────────────────────────────────────────────────────────
# Graph assembly — all nodes are defined above this point
# ─────────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    wf = StateGraph(ResearchState)
    wf.add_node("architect_node",  architect_node)
    wf.add_node("crawler_node",    crawler_node)
    wf.add_node("strategist_node", strategist_node)
    wf.add_node("report_node",     report_node)
    wf.add_node("purge_node",      purge_node)

    wf.set_entry_point("architect_node")

    wf.add_conditional_edges("architect_node", route_workflow, {
        "crawler_node":    "crawler_node",
        "strategist_node": "strategist_node",
        "report_node":     "report_node",
        "purge_node":      "purge_node",
        END:               END,
    })
    wf.add_conditional_edges("crawler_node", route_workflow, {
        "report_node":     "report_node",
        "crawler_node":    "crawler_node",
        END:               END,
    })
    wf.add_conditional_edges("strategist_node", route_workflow, {
        "crawler_node":    "crawler_node",
        "report_node":     "report_node",
        END:               END,
    })

    wf.add_edge("report_node", END)
    wf.add_edge("purge_node",  END)
    return wf.compile()
