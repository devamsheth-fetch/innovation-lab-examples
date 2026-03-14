"""Unit tests — no LLM calls, no network, no uAgents runtime."""
import pytest
import os
os.environ.setdefault("OPENAI_API_KEY", "test-key")
from autogen import LLMConfig

TEST_LLM_CONFIG = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": "test", "base_url": "https://api.openai.com/v1"}
)

def test_agents_instantiate():
    from agents import build_agents
    agents = build_agents(TEST_LLM_CONFIG)
    assert len(agents) == 4
    names = [a.name for a in agents]
    assert "web_researcher" in names
    assert "financial_analyst" in names
    assert "tech_analyst" in names
    assert "synthesizer" in names

def test_synthesizer_termination():
    from agents import build_agents
    agents = build_agents(TEST_LLM_CONFIG)
    synthesizer = next(a for a in agents if a.name == "synthesizer")
    assert synthesizer._is_termination_msg({"content": "Report done. TERMINATE"}) is True
    assert synthesizer._is_termination_msg({"content": "Still analysing..."}) is False

def test_executor_instantiates():
    from autogen import UserProxyAgent
    executor = UserProxyAgent(
        name="executor", human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
    )
    assert executor.name == "executor"


def test_llmconfig_positional_dict_construction():
    """Validate the positional-dict LLMConfig used in main.py works with AG2 0.11+."""
    cfg = LLMConfig(
        {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "https://api.openai.com/v1"},
        temperature=0.3,
        cache_seed=None,
    )
    from autogen import AssistantAgent
    agent = AssistantAgent(
        name="llm_config_test",
        system_message="test",
        llm_config=cfg,
    )
    assert agent.name == "llm_config_test"
