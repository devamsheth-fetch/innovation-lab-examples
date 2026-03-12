"""
Fetch.ai uAgent exposing the AG2 research team via A2A protocol (Pattern B).
Discoverable on Agentverse; callable from ASI:One or other uAgents.
"""
import os
from dotenv import load_dotenv
from uagents_adapter import SingleA2AAdapter
from autogen import LLMConfig

from agent_executor import AG2ResearchExecutor

load_dotenv()

llm_config = LLMConfig(
    config_list=[{
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    }],
    temperature=0.3,
    cache_seed=None,
)

executor = AG2ResearchExecutor(
    llm_config=llm_config,
    mcp_url=os.getenv("MCP_SERVER_URL"),  # optional: Fetch.ai MCP gateway
)

adapter = SingleA2AAdapter(
    agent_executor=executor,
    name="AG2 Research Synthesis Team",
    description=(
        "Multi-agent research team using AG2 (formerly AutoGen). "
        "Four specialists (web researcher, financial analyst, tech analyst, synthesizer) "
        "collaborate to produce comprehensive research reports on any topic."
    ),
    port=int(os.getenv("AGENT_PORT", "8008")),
    agentverse_url=os.getenv("AGENTVERSE_URL", "https://agentverse.ai"),
    mailbox_api_key=os.getenv("AGENTVERSE_API_KEY", ""),
)

if __name__ == "__main__":
    adapter.run()
