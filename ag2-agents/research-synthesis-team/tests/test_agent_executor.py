from unittest.mock import patch, MagicMock
from autogen import LLMConfig

TEST_LLM = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": "test", "base_url": "https://api.openai.com/v1"}
)

def test_executor_invoke_calls_run_research():
    from agent_executor import AG2ResearchExecutor
    executor = AG2ResearchExecutor(llm_config=TEST_LLM)
    with patch("agent_executor.asyncio.run", return_value="Mock research report") as mock_run:
        result = executor.invoke({"input": "quantum computing"})
    mock_run.assert_called_once()
    assert result == {"output": "Mock research report"}
