import os
import time
import logging
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase, Driver
from datetime import datetime, UTC, timedelta
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemgraphConnection:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("MEMGRAPH_USER", "")
        self.password = password or os.getenv("MEMGRAPH_PASSWORD", "")
        
        try:
            self.driver: Driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password) if self.user else None)
            logger.info("Connected to Memgraph via Cypher/Bolt.")
        except Exception as e:
            logger.error(f"Failed to connect to Memgraph: {e}")
            self.driver = None
        
    def close(self):
        if self.driver:
            self.driver.close()

    def execute_query(self, query: str, parameters: dict = None):
        """Execute a query and return results. Handles both read and write transactions."""
        with self.driver.session() as session:
            # If query contains write keywords, use execute_write for immediate commit
            write_keywords = ["MERGE", "CREATE", "DELETE", "SET", "REMOVE"]
            is_write = any(k in query.upper() for k in write_keywords)
            
            if is_write:
                return session.execute_write(lambda tx: [r.data() for r in tx.run(query, parameters or {})])
            else:
                return session.execute_read(lambda tx: [r.data() for r in tx.run(query, parameters or {})])

    def add_paper_node(self, paper_id: str, title: str, empirical_results: str = "", cluster_seed: str = "", original_id: str = "", url: str = "", oa_id: str = "", methodology: str = "", implementation: str = "", results: str = ""):
        query = """
        MERGE (p:Paper {id: $paper_id})
        SET p.title = $title, 
            p.empirical_results = $empirical_results,
            p.cluster_seed = coalesce(p.cluster_seed, $cluster_seed),
            p.original_id = coalesce(p.original_id, $original_id),
            p.url = coalesce(p.url, $url),
            p.oa_id = coalesce(p.oa_id, $oa_id),
            p.methodology = $methodology,
            p.implementation = $implementation,
            p.results = $results,
            p.created_at = coalesce(p.created_at, $created_at)
        RETURN p
        """
        return self.execute_query(query, {
            "paper_id": paper_id, 
            "title": title,
            "empirical_results": empirical_results,
            "cluster_seed": cluster_seed,
            "original_id": original_id,
            "url": url,
            "oa_id": oa_id,
            "methodology": methodology,
            "implementation": implementation,
            "results": results,
            "created_at": datetime.now(UTC).isoformat()
        })

    def add_gap_node(self, gap_id: str, description: str, worthiness_score: float = 0.0):
        query = """
        MERGE (g:Gap {id: $gap_id})
        SET g.description = $description, 
            g.worthiness_score = $worthiness_score,
            g.created_at = coalesce(g.created_at, $created_at)
        RETURN g
        """
        return self.execute_query(query, {
            "gap_id": gap_id, 
            "description": description, 
            "worthiness_score": worthiness_score,
            "created_at": datetime.now(UTC).isoformat()
        })

    def link_nodes(self, from_id: str, to_id: str, relationship_type: str = "CITES"):
        query = f"""
        MATCH (a) WHERE a.id = $from_id
        MATCH (b) WHERE b.id = $to_id
        MERGE (a)-[r:{relationship_type}]->(b)
        RETURN r
        """
        return self.execute_query(query, {"from_id": from_id, "to_id": to_id})

    def manual_purge(self):
        """High-priority administrative command processed by the Architect to wipe the database and local caches."""
        logger.warning("Initiating manual purge of Memgraph database...")
        # Adding RETURN count to provide immediate confirmation to the driver
        return self.execute_query("MATCH (n) DETACH DELETE n RETURN count(n) as deleted_count")

    def auto_purge_old_nodes(self, days: int = 10):
        """Deletes all nodes and associated metadata older than specified days based on 'Created At'."""
        cutoff_date = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        query = """
        MATCH (n)
        WHERE n.created_at < $cutoff_date
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """
        result = self.execute_query(query, {"cutoff_date": cutoff_date})
        if result and len(result) > 0:
            logger.info(f"Auto-purge completed. Deleted {result[0].get('deleted_count', 0)} old nodes.")
        return result

    def get_all_nodes(self):
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        """
        return self.execute_query(query)

    def paper_exists(self, paper_id: str) -> bool:
        """Check if a paper is already in the graph."""
        query = "MATCH (p:Paper {id: $paper_id}) RETURN p LIMIT 1"
        result = self.execute_query(query, {"paper_id": paper_id})
        return len(result) > 0

    def find_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[str]:
        """Find the canonical OpenAlex ID for a given ArXiv ID or DOI."""
        query = "MATCH (p:Paper) WHERE p.original_id = $arxiv_id OR p.id CONTAINS $arxiv_id RETURN p.id AS id LIMIT 1"
        result = self.execute_query(query, {"arxiv_id": arxiv_id})
        return result[0]['id'] if result else None

    def search_papers(self, keywords: str, limit: int = 10) -> list:
        """Search papers by keywords in title, methodology or results."""
        # Split keywords and search for each in a case-insensitive way
        query = """
        MATCH (p:Paper)
        WHERE (toLower(p.title) CONTAINS toLower($k) OR toLower(p.methodology) CONTAINS toLower($k) OR toLower(p.results) CONTAINS toLower($k))
        RETURN p
        LIMIT $limit
        """
        return self.execute_query(query, {"k": keywords, "limit": limit})

    def get_cluster(self, seed_id: str) -> list:
        """Return all papers belonging to a crawl cluster seeded by seed_id."""
        query = """
        MATCH (n:Paper)
        WHERE n.cluster_seed = $seed_id OR n.id = $seed_id
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        """
        return self.execute_query(query, {"seed_id": seed_id})

db = MemgraphConnection()
