import os
import logging
import sys

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.llm import ASI1LLM
from src.database import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# ASI1 configuration for Strategist
strategist_llm = ASI1LLM(model="asi1", temperature=0.3, max_tokens=4096)

def evaluate_gaps(seed_id: str = None) -> str:
    """Strategist evaluates the graph for open gaps. If seed_id is provided, it scopes to that cluster."""
    logger.info(f"Strategist performing worthiness analysis. Scoped seed: {seed_id}")
    
    if seed_id:
        nodes = db.get_cluster(seed_id)
    else:
        nodes = db.get_all_nodes()
    
    # Graph Compression: Extract only essential metadata
    compressed_data = []
    for record in nodes:
        node = record.get('n', {})
        compressed_data.append({
            "id": node.get('id'),
            "title": node.get('title'),
            "results": str(node.get('empirical_results', ""))[:200]
        })

    prompt = f"""
    [Identity Context]
    You are the R2E2 Strategist. Analyze this research graph (Compressed): {compressed_data}
    
    [Task Context]
    Identify 3 major research gaps and calculate a 'Worthiness Score' (0.0 to 1.0) for each. 
    Explain why these gaps matter based on the cited papers.
    """
    try:
        analysis = strategist_llm.invoke(prompt)
        return analysis
    except Exception as e:
        logger.error(f"Worthiness Analysis Error: {e}")
        return f"Worthiness analysis unavailable due to API timeout. Local cluster contains {len(compressed_data)} nodes."

def query_graph(question: str, seed_id: str = None) -> str:
    """Graph-First Q&A. Scopes to cluster if seed_id provided. Uses Perplexity-style citations."""
    # ── Step 0: Direct Keyword Search ───────────────────────────────
    # Perform a fast string-match search for keywords from the question
    keywords = [w.strip() for w in question.split() if len(w.strip()) > 3]
    direct_matches = []
    for kw in keywords:
        matches = db.search_papers(kw, limit=10)
        direct_matches.extend(matches)
    
    # ── Step 1: Semantic Retrieval ──────────────────────────────────
    # Fetch titles for a broader semantic scouting pass
    all_nodes_raw = db.get_all_nodes()
    all_titles = [{"id": r['n']['id'], "title": r['n']['title']} for r in all_nodes_raw if r.get('n')]
    
    if not all_titles:
        return "I don't have any research data in my graph yet. Please provide a paper URL to start."

    scout_prompt = f"""
    You are the Deep Research Scout. Evaluate if our current research library contains enough technical information to answer the user's question.
    
    Question: {question}
    Library Titles: {all_titles[:200]}
    
    EVALUATION RULES:
    1. If we have the PRIMARY paper or SEMANTICALLY RELATED papers that likely discuss this topic (even if the title isn't a 1:1 match), the answer is likely in our graph.
    2. Score the 'Information Coverage' from 0-100. 
    3. If Coverage >= 60, we will answer from the graph. If < 60, we will trigger a new search.
    
    Return JSON: {{"coverage_score": 85, "relevant_ids": ["id1", "id2"], "reasoning": "We have the foundational paper [id1] which covers the architecture requested."}}
    """
    try:
        import json
        scout_res = strategist_llm.invoke(scout_prompt).strip()
        if "```json" in scout_res:
            scout_res = scout_res.split("```json")[1].split("```")[0].strip()
        
        scout_data = json.loads(scout_res)
        coverage_score = scout_data.get("coverage_score", 0)
        relevant_ids = scout_data.get("relevant_ids", [])
        
        # We no longer trigger autonomous crawls. 
        # Even if the score is low, we provide the best possible answer from existing data.
        logger.info(f"Information coverage score: {coverage_score}%. Reasoning: {scout_data.get('reasoning')}. Proceeding with best-effort graph answer.")
    except Exception as e:
        logger.error(f"Scouting error: {e}")
        relevant_ids = [t['id'] for t in all_titles[:15]]

    # ── Step 2: Context Loading ─────────────────────────────────────
    # Fetch full technical data for the selected papers
    context = []
    source_map = {}
    for i, rid in enumerate(relevant_ids[:15]):
        # Match from our pre-fetched list
        node_match = next((r['n'] for r in all_nodes_raw if r['n']['id'] == rid), None)
        if node_match:
            sid = i + 1
            res_text = node_match.get('results') or node_match.get('empirical_results') or ""
            source_map[sid] = f"[{sid}] {node_match.get('title')} ({node_match.get('id')})"
            context.append(f"Source [{sid}]: ID {node_match.get('id')} - {node_match.get('title')}. Methodology: {node_match.get('methodology', '')[:200]}. Results: {str(res_text)[:300]}")

    prompt = f"""
    [Task]
    Answer the question using the provided sources. 
    Use inline citations like [1], [2] whenever you reference a source.
    
    [Context]
    {context}
    
    [Question]
    {question}
    
    [Instruction]
    Provide a detailed technical answer based ONLY on the provided sources.
    - NEVER state that you don't have enough information or that data is missing.
    - Synthesize the most helpful and positive answer possible using the available technical details.
    - If a specific detail is not present, provide the most relevant related technical context from the papers.
    - Ensure the tone is helpful and proactive.
    """
    try:
        response = strategist_llm.invoke(prompt).strip()
        if "SIGNAL:MISSING_DATA_FOR_SEARCH" in response:
            return response
            
        # Manually append the formatted source list to ensure accuracy
        sources_footer = "\n\n**Sources:**\n"
        used_indices = []
        import re
        # Find all [n] citations in the response
        matches = re.findall(r'\[(\d+)\]', response)
        for m in matches:
            idx = int(m)
            if idx in source_map and idx not in used_indices:
                sources_footer += f"{source_map[idx]}\n"
                used_indices.append(idx)
        
        if used_indices:
            return response + sources_footer
        return response
    except Exception as e:
        return f"Query error: {e}. Try a more specific question."

def generate_report(format: str = "markdown", seed_id: str = None) -> str:
    """
    Generates a structured 'Research Roadmap' report.
    1. Verified Frontier
    2. Research Gap Analysis
    3. Comparative Benchmark Tables
    4. Worthiness Scorecard
    5. Implementation Roadmap
    """
    if seed_id:
        logger.info(f"Generating Research Roadmap for cluster: {seed_id}")
        nodes_raw = db.get_cluster(seed_id)
    else:
        nodes_raw = db.get_all_nodes()
    
    # Process graph into a readable format for the LLM
    nodes = []
    edges = []
    for r in nodes_raw:
        n = r.get('n', {})
        rel = r.get('r', {})
        m = r.get('m', {})
        
        node_data = {
            "id": n.get('id'),
            "title": n.get('title'),
            "url": n.get('url', ''),
            "results": n.get('results', n.get('empirical_results', '')),
            "methodology": n.get('methodology', ''),
            "implementation": n.get('implementation', '')
        }
        if node_data not in nodes:
            nodes.append(node_data)
        
        if rel and m:
            # record.data() returns relationships as dicts or objects depending on driver
            # Let's be safe and try to get the type correctly
            rel_type = "RELATES_TO"
            if isinstance(rel, dict):
                rel_type = rel.get("type", "RELATES_TO")
            elif hasattr(rel, "type"):
                rel_type = rel.type
            
            edge = f"{n.get('id')} --({rel_type})--> {m.get('id')}"
            if edge not in edges:
                edges.append(edge)

    # Contextual string for LLM
    knowledge_str = "\n".join([
        f"[{i+1}] {n['id']}: {n['title']} (Link: {n['url']})\n"
        f"   - Analysis Depth: {'FULL TEXT' if n.get('methodology') and n.get('methodology') != 'Not specified' else 'ABSTRACT ONLY'}\n"
        f"   - Methodology: {n['methodology']}\n"
        f"   - Key Results: {n['results']}"
        for i, n in enumerate(nodes[:25])
    ])
    graph_str = "\n".join(edges[:30])

    prompt = f"""
    [Identity]
    You are the R2E2 Simplification Agent. Your goal is to take complex research data and turn it into a clear, 3-minute executive brief for a human user.
    
    [Input Data]
    Ingested Papers:
    {knowledge_str}
    
    [Formatting Rules]
    - Use plain English. Avoid overly academic jargon.
    - If you don't have data for a section, HIDE it entirely.
    - Use simple Markdown tables and bold bullet points.

    [Task: Generate The 3-Minute Report]
    Please provide the following four sections:

    1. 🚀 THE MAIN TAKEAWAY
       - A single, bold sentence summarizing the state of this research.

    2. 📊 TECHNICAL SNAPSHOT
       - A simple table comparing the top 3-5 papers.
       - Include columns for: Title, Methodology, and Key Results.
       - Include the specific 'Link' for each paper in the Title column.

    3. 🔍 SYSTEMIC RESEARCH GAPS (The Unsolved Frontier)
       - Identify 2-3 **systemic gaps** that are NOT addressed by ANY of the papers in this collection.
       - Use citations (e.g., "While [1] and [2] address X, no paper covers Y") to prove these gaps.
       - Focus on "blind spots" shared by the entire field.
       - These must be conceptual research holes, not technical data missing from the abstract.

    4. 💡 STRATEGIC RESEARCH DIRECTIONS
       - Instead of just listing papers, provide 2-3 **actionable research ideas** for the user.
       - Use citations to show the foundation of your ideas (e.g., "Based on [3], one could implement...").

    5. 📜 REFERENCES
       - List ALL papers mentioned or cited in the report.
       - Use the format: [Index] Title (Link: URL)
       - Ensure the Index matches the numbers used in your citations.
    """
    try:
        report_md = strategist_llm.invoke(prompt)
        if format == "pdf":
            return generate_pdf_report(report_md)
        return report_md
    except Exception as e:
        logger.error(f"Roadmap Generation Error: {e}")
        return f"# Research Roadmap (Local Fallback)\n\nError: {e}\n\nNodes: {len(nodes)}"

def generate_pdf_report(markdown_content: str) -> str:
    """Converts Markdown report to PDF."""
    pdf_path = os.path.abspath("research_report.pdf")
    logger.info(f"Generating PDF report at {pdf_path}...")
    
    try:
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Simple Markdown-to-PDF parser (simplified)
        for line in markdown_content.split('\n'):
            if line.startswith('# '):
                elements.append(Paragraph(line[2:], styles['Heading1']))
            elif line.startswith('## '):
                elements.append(Paragraph(line[3:], styles['Heading2']))
            elif line.strip():
                elements.append(Paragraph(line, styles['Normal']))
            elements.append(Spacer(1, 12))
            
        doc.build(elements)
        return f"PDF Report generated successfully: {pdf_path}"
    except Exception as e:
        return f"PDF generation error: {e}"
