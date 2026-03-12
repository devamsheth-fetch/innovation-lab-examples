"""
Wraps the AG2 GroupChat workflow as a LangChain-compatible AgentExecutor
for use with SingleA2AAdapter (Pattern B — matches LangChain/ADK examples).
"""
import asyncio
from autogen import LLMConfig
from workflow import run_research


class AG2ResearchExecutor:
    """Drop-in AgentExecutor interface for SingleA2AAdapter."""

    def __init__(self, llm_config: LLMConfig, mcp_url: str | None = None):
        self.llm_config = llm_config
        self.mcp_url = mcp_url

    def invoke(self, inputs: dict) -> dict:
        topic = inputs.get("input", "")
        result = asyncio.run(run_research(topic, self.llm_config, self.mcp_url))
        return {"output": result}
