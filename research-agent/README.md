# R2E2 Research Agent

R2E2 is an autonomous multi-agent knowledge graph construction system. It traverses academic literature to extract empirical results and perform automated gap analysis.

## Architecture
- **Architect (Routing)**: Acts as the primary gateway, analyzing user intent and rapidly routing tasks. Uses an isolated ASI1 context configured for fast routing.
- **Crawler (Recursive Auditor)**: Runs via EvoAgentX framework to recursively traverse citation trees across ArXiv and OpenAlex. Uses its own ASI1 instance for extracting empirical benchmarks.
- **Strategist (Final Auditor)**: Audits the knowledge graph to assign "worthiness scores" to research gaps. Uses an independent ASI1 instance configured for deep reasoning and Q&A.

## Infrastructure
- **Memgraph**: In-memory graph database accessed via native Cypher (Bolt). Includes automated 10-day purging.
- **LangGraph**: Orchestrates the core workflows between agents.
- **ReportLab**: Generates PDF outputs with benchmark tables and citation trees.

## Setup
1. Clone the repository and navigate to this directory.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your keys (e.g., `ASI_LLM_KEY`).
4. Start the Memgraph container:
   ```bash
   docker-compose up -d
   ```
5. Run the agents or the workflow manually.
