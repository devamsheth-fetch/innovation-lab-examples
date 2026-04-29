import os
import re
import logging
import requests
from typing import Dict, Any, List, Optional

# Ensure src can be imported
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.database import db
from src.llm import ASI1LLM

# ASI1 configuration for Crawler
crawler_llm = ASI1LLM(model="asi1", temperature=0.5)

# --- Dependency Resilience Layer ---
# EvoAgentX aggressively imports all tools. Mock unused ones to prevent init errors.
from unittest.mock import MagicMock

_mock_modules = [
    "googleapiclient", "googleapiclient.discovery", "google_auth_oauthlib",
    "google_auth_oauthlib.flow", "telethon", "docker", "browser_use",
    "arxiv",
]
for _mod in _mock_modules:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

try:
    from evoagentx.actions import Action, ActionInput, ActionOutput
    from evoagentx.agents import AgentManager, Agent
    from evoagentx.workflow import WorkFlow, WorkFlowGenerator, Task
    EVO_AVAILABLE = True
except ImportError as e:
    EVO_AVAILABLE = False
    logging.warning(f"EvoAgentX initialization partially failed: {e}. Using built-in crawler.")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# OpenAlex helpers
# ─────────────────────────────────────────────────────────────
# ArXiv API Helper (Metadata Translator)
# ─────────────────────────────────────────────────────────────

def fetch_arxiv_metadata(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch canonical metadata from ArXiv API.
    Used as a translator to find papers in OpenAlex via Title/DOI.
    """
    import xml.etree.ElementTree as ET
    endpoint = "http://export.arxiv.org/api/query"
    params = {"id_list": arxiv_id}
    
    try:
        logger.info(f"Translating ArXiv ID {arxiv_id} via Cornell API...")
        resp = requests.get(endpoint, params=params, timeout=10)
        resp.raise_for_status()
        
        root = ET.fromstring(resp.content)
        # ArXiv API uses Atom namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        
        if entry is None or entry.find('atom:id', ns) is None:
            return None
            
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        doi = entry.find('{http://arxiv.org/schemas/atom}doi')
        
        return {
            "title": title,
            "summary": summary,
            "doi": doi.text if doi is not None else None,
            "arxiv_id": arxiv_id
        }
    except Exception as e:
        logger.error(f"ArXiv API Translation Error: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# Semantic Scholar API Helper (Graph Fallback)
# ─────────────────────────────────────────────────────────────

# Local cache to prevent redundant 429-triggering calls in a single session
_ss_cache: Dict[str, Any] = {}

def fetch_semantic_scholar_data(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch citation and reference data from Semantic Scholar.
    Includes caching and exponential backoff to handle strict 429 rate limits.
    """
    import time
    import random

    # Normalize ID for Semantic Scholar
    if "arxiv" in paper_id.lower() or re.match(r'^\d{4}\.\d{4,5}', paper_id):
        clean_id = paper_id.lower().replace("arxiv:", "")
        ss_id = f"ARXIV:{clean_id}"
    elif "10." in paper_id:
        ss_id = f"DOI:{paper_id}"
    else:
        ss_id = paper_id

    if ss_id in _ss_cache:
        logger.info(f"Using cached Semantic Scholar data for {ss_id}")
        return _ss_cache[ss_id]

    endpoint = f"https://api.semanticscholar.org/graph/v1/paper/{ss_id}"
    params = {
        "fields": "title,abstract,publicationDate,citations,references,citations.title,citations.abstract,references.title,references.abstract"
    }
    
    # Professional headers to avoid being flagged as a basic bot
    headers = {
        "User-Agent": "R2E2-Research-Agent/1.0 (https://github.com/fetchai/innovation-lab-examples; mailto:developer@fetch.ai)",
        "Accept": "application/json"
    }
    
    ss_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if ss_api_key and "your_semantic" not in ss_api_key:
        headers["x-api-key"] = ss_api_key
    
    # Exponential backoff retry logic (more aggressive)
    for attempt in range(3):
        try:
            logger.info(f"Checking Semantic Scholar for graph data: {ss_id} (Attempt {attempt+1})...")
            resp = requests.get(endpoint, params=params, headers=headers, timeout=15)
            
            if resp.status_code == 429:
                # Longer wait for public tier: 5s, 10s, 20s
                wait = (5 * (2 ** attempt)) + random.random()
                logger.warning(f"Semantic Scholar Rate Limit (429). Backing off for {wait:.2f}s...")
                time.sleep(wait)
                continue
                
            if resp.status_code == 404:
                return None
                
            resp.raise_for_status()
            data = resp.json()
            _ss_cache[ss_id] = data  # Save to cache
            return data
            
        except Exception as e:
            logger.error(f"Semantic Scholar API Error: {e}")
            if attempt < 2:
                time.sleep(2)
                continue
            return None
    return None

# ─────────────────────────────────────────────────────────────
# Deep Read (ArXiv HTML Extraction)
# ─────────────────────────────────────────────────────────────

def deep_read_arxiv_paper(arxiv_id: str) -> Dict[str, str]:
    """
    Downloads the HTML version of an ArXiv paper (Ar5iv) and extracts
    high-density sections for Methodology, Implementation, and Results.
    """
    # Directly use Ar5iv Labs (much more reliable for older papers)
    html_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        logger.info(f"Deep Reading ArXiv paper (Ar5iv): {html_url}...")
        resp = requests.get(html_url, timeout=15)
        resp.raise_for_status()
        html_content = resp.text
        
        # Simple text extraction (stripping HTML tags)
        # In a real environment, we'd use BeautifulSoup, but we'll use regex for zero-dep resilience
        text_content = re.sub(r'<[^>]+>', ' ', html_content)
        text_content = re.sub(r'\s+', ' ', text_content)[:15000] # Limit to ~3000 tokens
        
        prompt = f"""You are a technical research analyst. Analyze the following full-text excerpt from paper {arxiv_id}.
        Extract and summarize exactly three sections:
        1. Methodology: The core mathematical or theoretical approach.
        2. Implementation: Hardware, software, dataset, or training details.
        3. Results: Key metrics, benchmarks, and quantitative findings.
        
        Text: {text_content}
        
        Format as JSON: {{"methodology": "...", "implementation": "...", "results": "..."}}
        Return ONLY the JSON.
        """
        
        import json
        llm_response = crawler_llm.invoke(prompt).strip()
        # Handle potential LLM markdown wrapping
        if "```json" in llm_response:
            llm_response = llm_response.split("```json")[1].split("```")[0].strip()
        
        return json.loads(llm_response)
    except Exception as e:
        logger.error(f"Deep Read Error for {arxiv_id}: {e}")
        return {"methodology": "N/A", "implementation": "N/A", "results": "N/A"}

# ─────────────────────────────────────────────────────────────
# ScienceDirect / Elsevier Deep Read
# ─────────────────────────────────────────────────────────────

def fetch_elsevier_deep_read(doi: str) -> Dict[str, str]:
    """
    Uses the ScienceDirect API to fetch full-text content for a given DOI.
    Extracts Methodology and Results sections using LLM.
    """
    api_key = os.getenv("ELSEVIER_API_KEY")
    if not api_key or "your_elsevier" in api_key:
        return {"methodology": "", "implementation": "", "results": ""}

    # DOI needs to be clean (strip https://doi.org/)
    clean_doi = doi.replace("https://doi.org/", "")
    endpoint = f"https://api.elsevier.com/content/article/doi/{clean_doi}"
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json"
    }

    try:
        logger.info(f"Deep Reading ScienceDirect paper: {doi}...")
        resp = requests.get(endpoint, headers=headers, timeout=20)
        if resp.status_code == 403:
            logger.warning(f"Elsevier Access Denied for {doi} (API key may not have full-text permissions).")
            return {"methodology": "", "implementation": "", "results": ""}
        
        resp.raise_for_status()
        data = resp.json()
        
        # Elsevier returns a nested 'originalText' or 'body' depending on the journal
        full_text = str(data.get("full-text-retrieval-response", {}).get("originalText", ""))
        if not full_text:
            full_text = str(data)

        text_excerpt = full_text[:15000] # Limit to ~3000 tokens
        
        prompt = f"Analyze this Elsevier paper excerpt. Extract Methodology and Key Results. JSON format: {{\"methodology\": \"...\", \"results\": \"...\"}}\n\nText: {text_excerpt}"
        
        import json
        llm_response = crawler_llm.invoke(prompt).strip()
        if "```json" in llm_response:
            llm_response = llm_response.split("```json")[1].split("```")[0].strip()
        
        return json.loads(llm_response)
    except Exception as e:
        logger.error(f"Elsevier Deep Read Error for {doi}: {e}")
        return {"methodology": "", "implementation": "", "results": ""}

# ─────────────────────────────────────────────────────────────

def _reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    if not inverted_index:
        return ""
    try:
        max_pos = max(pos for positions in inverted_index.values() for pos in positions)
        words = [""] * (max_pos + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                if pos < len(words):
                    words[pos] = word
        return " ".join(words)
    except Exception:
        return "Abstract reconstruction failed."

def fetch_openalex_papers(
    query: str,
    max_results: int = 5,
    citing_id: str = None,
    arxiv_id: str = None,
    oa_id: str = None,
) -> List[Dict[str, Any]]:
    """
    Fetch papers from OpenAlex.
    Priority: oa_id > arxiv_id > citing_id > keyword search.
    """
    endpoint = os.getenv("OPENALEX_URL", "https://api.openalex.org")
    params: Dict[str, Any] = {"per_page": max_results}

    if oa_id:
        # Direct lookup by OpenAlex Work ID
        params["filter"] = f"openalex:{oa_id}"
    elif arxiv_id:
        # ArXiv specific sequence (DOI then Landing Page)
        params["filter"] = f"doi:10.48550/arxiv.{arxiv_id}"
    elif query.startswith("10."):
        # Standard DOI lookup
        params["filter"] = f"doi:{query}"
    elif query.startswith("http"):
        # Generic URL lookup
        params["filter"] = f"locations.landing_page_url:{query}"
    elif citing_id:
        # OpenAlex cites filter requires the full URL form
        oa_url = citing_id if citing_id.startswith("https://") else f"https://openalex.org/{citing_id}"
        params["filter"] = f"cites:{oa_url}"
    else:
        # Keyword search
        params["search"] = query

    papers = []
    try:
        response = requests.get(f"{endpoint}/works", params=params, timeout=15)
        response.raise_for_status()
        results = response.json().get("results", [])

        # Sequential Fallback for ArXiv IDs only
        if not results and arxiv_id:
            logger.info("DOI filter returned nothing — trying ArXiv landing page fallback...")
            lp_filter = f"locations.landing_page_url:http://arxiv.org/abs/{arxiv_id}|https://arxiv.org/abs/{arxiv_id}"
            params["filter"] = lp_filter
            response = requests.get(f"{endpoint}/works", params=params, timeout=15)
            results = response.json().get("results", [])

        # Fallback: if filter returned nothing, try keyword search 
        # (UNLESS it was a specific ArXiv ID or Citing ID search)
        if not results and "filter" in params and not (arxiv_id or citing_id):
            logger.info("OpenAlex filter returned nothing — falling back to keyword search.")
            params = {"per_page": max_results, "search": query}
            response = requests.get(f"{endpoint}/works", params=params, timeout=15)
            results = response.json().get("results", [])

        for item in results:
            abstract = _reconstruct_abstract(item.get("abstract_inverted_index", {}))
            oa_work_id = item.get("id", "").split("/")[-1]
            doi = item.get("doi")
            lp_url = item.get("primary_location", {}).get("landing_page_url") if item.get("primary_location") else None
            
            # Detect ArXiv ID from URLs
            found_arxiv_id = None
            locs = item.get("locations", []) or []
            all_urls = ([lp_url] if lp_url else []) + [loc.get("landing_page_url") for loc in locs if loc and loc.get("landing_page_url")]
            for u in all_urls:
                if u and "arxiv.org/abs/" in u:
                    found_arxiv_id = u.split("arxiv.org/abs/")[-1].split("v")[0] # strip version
                    break

            # Synthesize ArXiv URL if found
            arxiv_url = f"https://arxiv.org/abs/{found_arxiv_id}" if found_arxiv_id else None
            
            # Prefer DOI, then ArXiv URL, then Landing Page, then OA ID
            clean_id = doi if doi else (arxiv_url if arxiv_url else (lp_url if lp_url else f"oa:{oa_work_id}"))

            papers.append({
                "id": clean_id,
                "oa_id": oa_work_id,
                "arxiv_id": found_arxiv_id,
                "title": item.get("display_name", "Unknown"),
                "summary": abstract or "No abstract available.",
                "publication_date": item.get("publication_date", ""),
                "url": clean_id
            })
    except Exception as e:
        logger.error(f"OpenAlex API Error: {e}")
    return papers


def extract_empirical_results(summary: str) -> str:
    prompt = f"""Extract ONLY empirical results, quantitative benchmarks, and technical metrics.
Abstract: {summary}"""
    try:
        return crawler_llm.invoke(prompt).strip()
    except Exception as e:
        return f"Extraction error: {e}"


# ─────────────────────────────────────────────────────────────
# URL / seed helpers
# ─────────────────────────────────────────────────────────────

def _extract_url(raw: str) -> str:
    """Return the first URL found in a chat message."""
    urls = re.findall(r'https?://\S+', raw)
    return urls[0].rstrip(".,;)") if urls else raw.strip()

def _arxiv_id_from_url(url: str) -> Optional[str]:
    m = re.search(r'arxiv\.org/abs/([^\s/?]+)', url)
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────
# 3-Layer Crawler (OpenAlex only)
# ─────────────────────────────────────────────────────────────

def fetch_referenced_works(paper_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch papers that the given work itself REFERENCES (its bibliography).
    Handles both OpenAlex IDs and Semantic Scholar IDs.
    """
    papers = []
    
    # ── CASE 1: Semantic Scholar ──────────────────────────────
    if paper_id.startswith("s2:"):
        ss_id = paper_id.replace("s2:", "")
        ss_data = fetch_semantic_scholar_data(ss_id)
        if ss_data and ss_data.get("references"):
            for r in ss_data["references"][:10]:
                ss_ref_id = r.get('paperId')
                papers.append({
                    "id": f"s2:{ss_ref_id}",
                    "oa_id": None,
                    "title": r.get("title", "Unknown"),
                    "summary": r.get("abstract") or "No abstract available.",
                    "publication_date": "",
                    "url": f"https://www.semanticscholar.org/paper/{ss_ref_id}"
                })
        return papers

    # ── CASE 2: OpenAlex / DOI ────────────────────────────────
    endpoint = os.getenv("OPENALEX_URL", "https://api.openalex.org")
    try:
        # If it's a DOI or full URL, we need to find it in OpenAlex first
        if paper_id.startswith("http"):
            oa_results = fetch_openalex_papers(paper_id, max_results=1)
            if not oa_results or not oa_results[0].get("oa_id"):
                return []
            oa_work_id = oa_results[0]["oa_id"]
        else:
            oa_work_id = paper_id.replace("oa:", "")

        # Step 1: Get the work's own referenced_works list
        resp = requests.get(f"{endpoint}/works/{oa_work_id}", timeout=15)
        resp.raise_for_status()
        work_data = resp.json()
        ref_ids = work_data.get("referenced_works", [])[:max_results]

        if not ref_ids:
            logger.info(f"No referenced_works found for {oa_work_id}")
            return []

        # Step 2: Bulk fetch metadata for each referenced work
        for ref_url in ref_ids:
            ref_oa_id = ref_url.split("/")[-1]
            try:
                r = requests.get(f"{endpoint}/works/{ref_oa_id}", timeout=10)
                if r.ok:
                    item = r.json()
                    abstract = _reconstruct_abstract(item.get("abstract_inverted_index", {}))
                    doi = item.get("doi")
                    lp_url = item.get("primary_location", {}).get("landing_page_url") if item.get("primary_location") else None
                    # Detect ArXiv ID
                    found_arxiv_id = None
                    locs = item.get("locations", []) or []
                    all_urls = ([lp_url] if lp_url else []) + [loc.get("landing_page_url") for loc in locs if loc and loc.get("landing_page_url")]
                    for u in all_urls:
                        if u and "arxiv.org/abs/" in u:
                            found_arxiv_id = u.split("arxiv.org/abs/")[-1].split("v")[0]
                            break
                    
                    arxiv_url = f"https://arxiv.org/abs/{found_arxiv_id}" if found_arxiv_id else None
                    clean_id = doi if doi else (arxiv_url if arxiv_url else (lp_url if lp_url else f"oa:{ref_oa_id}"))

                    papers.append({
                        "id": clean_id,
                        "oa_id": ref_oa_id,
                        "arxiv_id": found_arxiv_id,
                        "title": item.get("display_name", "Unknown"),
                        "summary": abstract or "No abstract available.",
                        "publication_date": item.get("publication_date", ""),
                        "url": clean_id
                    })
            except Exception as e:
                logger.warning(f"Could not fetch referenced work {ref_oa_id}: {e}")
    except Exception as e:
        logger.error(f"fetch_referenced_works error for {paper_id}: {e}")
    return papers


def perform_recursive_extraction(seed: str) -> str:
    """
    Accepts a clean ArXiv ID (e.g. '2401.00001') or a keyword query.
    Level 0 — Seed paper
    Level 1 — Papers CITING the seed
    Level 2 — References of Level-1 papers
    Level 3 — Papers REFERENCED BY the seed (bibliography)
    """
    # Determine if input is an ArXiv ID or a keyword
    import re
    # Import locally to avoid circular dependencies
    from src.workflow import extract_paper_id
    arxiv_id = extract_paper_id(seed) or (seed if re.match(r'^\d{4}\.\d{4,5}', seed) else None)
    
    logger.info(f"Crawler start | seed={seed} | resolved_arxiv_id={arxiv_id}")

    stored_count = 0

    # ── Level 0: Seed paper ───────────────────────────────────
    logger.info("Level 0 — fetching seed paper from OpenAlex...")
    if arxiv_id:
        seed_papers = fetch_openalex_papers(seed, max_results=1, arxiv_id=arxiv_id)
        
        # TRANSITION: If OpenAlex ID lookup fails, use ArXiv API as a translator
        if not seed_papers:
            logger.info(f"OpenAlex direct lookup failed for {arxiv_id}. Triggering ArXiv Translator...")
            arxiv_meta = fetch_arxiv_metadata(arxiv_id)
            if arxiv_meta:
                # Try finding by DOI if ArXiv provided it
                if arxiv_meta.get("doi"):
                    seed_papers = fetch_openalex_papers(arxiv_meta["doi"], max_results=1)
                
                # If still nothing, try finding by Canonical Title (very high precision)
                if not seed_papers:
                    logger.info(f"Searching OpenAlex by canonical title: {arxiv_meta['title']}")
                    seed_papers = fetch_openalex_papers(arxiv_meta["title"], max_results=1)
                
                # If STILL nothing, use ArXiv data as a standalone node (Level 0 only)
                if not seed_papers:
                    logger.warning(f"Paper {arxiv_id} not in OpenAlex graph. Using ArXiv metadata standalone.")
                    seed_papers = [{
                        "id": f"arxiv:{arxiv_id}",
                        "title": arxiv_meta["title"],
                        "summary": arxiv_meta["summary"],
                        "publication_date": ""
                    }]
    else:
        seed_papers = fetch_openalex_papers(seed, max_results=1)

    if not seed_papers:
        return f"OpenAlex could not find any papers for: {seed}"

    sp = seed_papers[0]
    seed_id = sp["id"]
    raw_oa_id = sp.get("oa_id")
    cluster_seed = seed_id  # every paper in this crawl is tagged with seed_id

    # Level 0 Deep Read
    deep_data = {"methodology": "", "implementation": "", "results": ""}
    if arxiv_id:
        logger.info(f"Triggering Deep Read for seed paper {arxiv_id}...")
        deep_data = deep_read_arxiv_paper(arxiv_id)
    
    # Try Elsevier if DOI is available and ArXiv failed
    if (not deep_data.get("methodology") or deep_data["methodology"] == "N/A") and sp.get("id", "").startswith("10."):
        elsevier_res = fetch_elsevier_deep_read(sp["id"])
        if elsevier_res.get("methodology"):
            deep_data.update(elsevier_res)

    # Store original user query (ArXiv ID or DOI) for cache resolution later
    original_id_to_store = arxiv_id if arxiv_id else seed
    db.add_paper_node(
        seed_id, 
        sp["title"], 
        extract_empirical_results(sp["summary"]), 
        cluster_seed=cluster_seed, 
        original_id=original_id_to_store,
        url=sp.get("url", ""),
        oa_id=raw_oa_id,
        methodology=deep_data.get("methodology", ""),
        implementation=deep_data.get("implementation", ""),
        results=deep_data.get("results", "")
    )
    stored_count += 1
    logger.info(f"Level 0 stored: {sp['title']}")

    # ── Level 1: Papers citing the seed (up to 10) ─────────────
    logger.info(f"Level 1 — papers citing {seed_id}...")
    citing = fetch_openalex_papers(sp["title"], max_results=10, citing_id=raw_oa_id)
    
    # FALLBACK: If OpenAlex has 0 citations, try Semantic Scholar
    if not citing:
        logger.info("OpenAlex found 0 citations — falling back to Semantic Scholar...")
        ss_data = fetch_semantic_scholar_data(arxiv_id if arxiv_id else raw_oa_id)
        if ss_data and ss_data.get("citations"):
            for c in ss_data["citations"][:10]:
                ss_paper_id = c.get('paperId')
                ext_ids = c.get('externalIds') or {}
                citing.append({
                    "id": f"s2:{ss_paper_id}",
                    "arxiv_id": ext_ids.get('ArXiv'),
                    "title": c.get("title", "Unknown"),
                    "summary": c.get("abstract") or "No abstract available.",
                    "publication_date": "",
                    "url": f"https://www.semanticscholar.org/paper/{ss_paper_id}"
                })

    level1_ids = []
    for p in citing:
        # Smart Extraction: Deep Read (ArXiv) or Abstract Analysis (Other)
        d_data = {"methodology": "", "implementation": "", "results": ""}
        if p.get("arxiv_id"):
            logger.info(f"  Recursive Deep Read for L1 citation: {p['arxiv_id']}...")
            d_data = deep_read_arxiv_paper(p["arxiv_id"])
        
        # Try Elsevier if DOI is available and ArXiv failed
        if (not d_data.get("methodology")) and p.get("id", "").startswith("10."):
            elsevier_res = fetch_elsevier_deep_read(p["id"])
            if elsevier_res.get("methodology"):
                d_data.update(elsevier_res)
        
        # Fallback to high-density abstract analysis if deep read failed or wasn't applicable
        if not d_data.get("methodology") or d_data["methodology"] == "Not specified":
            summary_text = p.get("summary", "")
            if summary_text and len(summary_text) > 50:
                logger.info(f"  Performing High-Density Abstract Analysis for: {p['title']}...")
                fallback_prompt = f"""Extract high-density technical info from this abstract:
                1. Methodology: The mathematical/theoretical approach.
                2. Key Findings: Main results or improvements.
                3. Conceptual Gaps: What does this paper NOT address?
                
                Abstract: {summary_text}
                
                Return JSON: {{"methodology": "...", "results": "...", "gaps": "..."}}"""
                try:
                    fb_res = json.loads(crawler_llm.invoke(fallback_prompt).strip())
                    d_data["methodology"] = fb_res.get("methodology", "Not specified")
                    d_data["results"] = fb_res.get("results", "Not specified")
                    d_data["gaps"] = fb_res.get("gaps", "Not specified")
                except:
                    pass

        db.add_paper_node(
            p["id"], 
            p["title"], 
            extract_empirical_results(p["summary"]), 
            cluster_seed=cluster_seed, 
            url=p.get("url", ""), 
            oa_id=p.get("oa_id"),
            methodology=d_data.get("methodology", ""),
            implementation=d_data.get("implementation", ""),
            results=d_data.get("results", "")
        )
        db.link_nodes(p["id"], seed_id, "CITES")
        level1_ids.append(p["id"])
        stored_count += 1
        logger.info(f"  L1 stored: {p['title']}")

    # ── Level 2: References OF each Level-1 paper (their bibliography) ─
    logger.info("Level 2 — fetching references of Level-1 papers...")
    for l1_id in level1_ids[:5]:
        raw_l1 = l1_id.replace("oa:", "")
        refs = fetch_referenced_works(raw_l1, max_results=5)
        for p in refs:
            # Smart Extraction for Level 2 (Secondary References)
            d_data = {"methodology": "", "implementation": "", "results": ""}
            if p.get("arxiv_id"):
                logger.info(f"  Recursive Deep Read for L2 reference: {p['arxiv_id']}...")
                d_data = deep_read_arxiv_paper(p["arxiv_id"])
            
            # Try Elsevier if DOI is available and ArXiv failed
            if (not d_data.get("methodology")) and p.get("id", "").startswith("10."):
                elsevier_res = fetch_elsevier_deep_read(p["id"])
                if elsevier_res.get("methodology"):
                    d_data.update(elsevier_res)
            
            if not d_data.get("methodology") or not d_data.get("results"):
                summary_text = p.get("summary", "")
                if summary_text and len(summary_text) > 50:
                    fallback_prompt = f"From this abstract, extract: 1. Methodology 2. Key Results. Return JSON: {{\"methodology\": \"...\", \"results\": \"...\"}}\n\nAbstract: {summary_text}"
                    try:
                        fb_res = json.loads(crawler_llm.invoke(fallback_prompt).strip())
                        d_data["methodology"] = d_data.get("methodology") or fb_res.get("methodology", "Not specified")
                        d_data["results"] = d_data.get("results") or fb_res.get("results", "Not specified")
                    except:
                        pass

            db.add_paper_node(
                p["id"], 
                p["title"], 
                p["summary"], 
                cluster_seed=cluster_seed, 
                url=p.get("url", ""), 
                oa_id=p.get("oa_id"),
                methodology=d_data.get("methodology", ""),
                implementation=d_data.get("implementation", ""),
                results=d_data.get("results", "")
            )
            db.link_nodes(l1_id, p["id"], "REFERENCES")
            stored_count += 1
            logger.info(f"  L2 stored: {p['title']}")

    # ── Level 3: Papers referenced IN the seed (bibliography) ─
    logger.info(f"Level 3 — papers referenced BY seed ({raw_oa_id})...")
    seed_refs = fetch_referenced_works(raw_oa_id, max_results=10)
    
    # FALLBACK: If OpenAlex has 0 references, try Semantic Scholar
    if not seed_refs:
        logger.info("OpenAlex found 0 references — falling back to Semantic Scholar...")
        ss_data = fetch_semantic_scholar_data(arxiv_id if arxiv_id else raw_oa_id)
        if ss_data and ss_data.get("references"):
            for r in ss_data["references"][:10]:
                ss_ref_id = r.get('paperId')
                ext_ids = r.get('externalIds') or {}
                seed_refs.append({
                    "id": f"s2:{ss_ref_id}",
                    "arxiv_id": ext_ids.get('ArXiv'),
                    "title": r.get("title", "Unknown"),
                    "summary": r.get("abstract") or "No abstract available.",
                    "publication_date": "",
                    "url": f"https://www.semanticscholar.org/paper/{ss_ref_id}"
                })

    for p in seed_refs:
        # Smart Extraction: Deep Read (ArXiv) or Abstract Analysis (Other)
        d_data = {"methodology": "", "implementation": "", "results": ""}
        if p.get("arxiv_id"):
            logger.info(f"  Recursive Deep Read for L3 reference: {p['arxiv_id']}...")
            d_data = deep_read_arxiv_paper(p["arxiv_id"])

        # Try Elsevier if DOI is available and ArXiv failed
        if (not d_data.get("methodology")) and p.get("id", "").startswith("10."):
            elsevier_res = fetch_elsevier_deep_read(p["id"])
            if elsevier_res.get("methodology"):
                d_data.update(elsevier_res)

        # Fallback to abstract if deep read failed or wasn't applicable
        if not d_data.get("methodology") or not d_data.get("results"):
            summary_text = p.get("summary", "")
            if summary_text and len(summary_text) > 50:
                fallback_prompt = f"From this abstract, extract: 1. Methodology 2. Key Results. Return JSON: {{\"methodology\": \"...\", \"results\": \"...\"}}\n\nAbstract: {summary_text}"
                try:
                    fb_res = json.loads(crawler_llm.invoke(fallback_prompt).strip())
                    d_data["methodology"] = d_data.get("methodology") or fb_res.get("methodology", "Not specified")
                    d_data["results"] = d_data.get("results") or fb_res.get("results", "Not specified")
                except:
                    pass

        db.add_paper_node(
            p["id"], 
            p["title"], 
            p["summary"], 
            cluster_seed=cluster_seed, 
            url=p.get("url", ""), 
            oa_id=p.get("oa_id"),
            methodology=d_data.get("methodology", ""),
            implementation=d_data.get("implementation", ""),
            results=d_data.get("results", "")
        )
        db.link_nodes(seed_id, p["id"], "REFERENCES")
        stored_count += 1
        logger.info(f"  L3 stored: {p['title']}")

    msg = f"Crawl complete: {stored_count} papers ingested across 4 levels for '{sp['title']}'."
    logger.info(msg)
    return seed_id, msg
