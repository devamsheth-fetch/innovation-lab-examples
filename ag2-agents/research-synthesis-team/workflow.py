"""GroupChat orchestration with AG2 native MCP client for tool access."""
from autogen import GroupChat, GroupChatManager, LLMConfig, UserProxyAgent
from autogen.mcp import create_toolkit
from mcp import ClientSession
from mcp.client.sse import sse_client

from agents import build_agents


async def run_research(topic: str, llm_config: LLMConfig, mcp_url: str | None = None) -> str:
    agents = build_agents(llm_config)
    executor = UserProxyAgent(
        name="executor",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
        default_auto_reply="",
    )

    # Optionally connect to an MCP server (e.g. Fetch.ai's Brave/DuckDuckGo MCP)
    if mcp_url:
        async with (
            sse_client(mcp_url, timeout=60) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            toolkit = await create_toolkit(session=session)
            toolkit.register_for_llm(agents[0])   # web_researcher gets the tools
            toolkit.register_for_execution(agents[0])
            result = await _run_groupchat(agents, executor, llm_config, topic)
    else:
        result = await _run_groupchat(agents, executor, llm_config, topic)

    return result


async def _run_groupchat(agents, executor, llm_config, topic):
    gc = GroupChat(
        agents=[executor] + agents,
        messages=[],
        max_round=16,
        speaker_selection_method="auto",
    )
    manager = GroupChatManager(groupchat=gc, llm_config=llm_config)
    await executor.a_initiate_chat(
        manager,
        message=f"Research and synthesise a comprehensive report on: {topic}",
    )
    reports = [m["content"] for m in gc.messages if m.get("name") == "synthesizer" and m.get("content")]
    return reports[-1] if reports else "Research did not complete."
