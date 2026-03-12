# AG2 Research Synthesis Team

A multi-agent research pipeline using [AG2](https://github.com/ag2ai/ag2) (formerly AutoGen)
integrated with the Fetch.ai uAgents ecosystem via the A2A protocol.

## Architecture

Four specialists collaborate under GroupChat with LLM-driven speaker selection, wrapped as
an A2A executor and exposed as a discoverable agent on Agentverse.

```
GroupChat (AG2)
├── web_researcher   — searches and gathers information
├── financial_analyst — analyses market and economic aspects
├── tech_analyst     — evaluates technical feasibility
└── synthesizer      — produces the final report
         ↓
SingleA2AAdapter (Fetch.ai uagents-adapter)
         ↓
Agentverse (discoverable at port 8008)
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY and AGENTVERSE_API_KEY
python main.py
```

## AG2 Features Demonstrated

- **`GroupChat` with `speaker_selection_method="auto"`** — LLM-driven dynamic speaker selection
- **Native MCP client** — optional connection to Fetch.ai's MCP gateway for web search tools
- **Pattern B (A2A Outbound)** — same integration pattern as LangChain and Google ADK examples
