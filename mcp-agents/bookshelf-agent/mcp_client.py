import json
import logging
import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

logger = logging.getLogger("mcp_client")

OPEN_LIBRARY_PARAMS = StdioServerParameters(
    command="npx",
    args=["-y", "@8ensmith/mcp-open-library"] # Using the correct npx package name format
)

GOOGLE_MAPS_PARAMS = StdioServerParameters(
    command="npx",
    args=["-y", "@google/mcp-google-maps"]
)

# Store mapping of tool name to server params
_tool_to_server = {}

def mcp_to_openai_tool(mcp_tool):
    """Maps an MCP tool definition to the OpenAI function-calling schema."""
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
        },
    }

async def _fetch_tools_for_server(params: StdioServerParameters, server_id: str):
    logger.info(f"[{server_id}] Fetching tools via npx...")
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                openai_tools = []
                for t in tools_result.tools:
                    openai_tools.append(mcp_to_openai_tool(t))
                    _tool_to_server[t.name] = params
                logger.info(f"[{server_id}] Loaded {len(tools_result.tools)} tools.")
                return openai_tools
    except Exception as e:
        logger.error(f"[{server_id}] Failed to fetch tools: {e}")
        return []

async def fetch_mcp_tools(retries: int = 2, backoff: float = 1.0):
    """Connects to MCP servers via stdio and aggregates their tools."""
    _tool_to_server.clear()
    
    # We run them sequentially or concurrently
    tasks = [
        _fetch_tools_for_server(OPEN_LIBRARY_PARAMS, "OpenLibrary"),
        _fetch_tools_for_server(GOOGLE_MAPS_PARAMS, "GoogleMaps")
    ]
    
    results = await asyncio.gather(*tasks)
    all_tools = []
    for t_list in results:
        all_tools.extend(t_list)
        
    return all_tools

async def _call_single_tool(tool_call):
    """Executes a single tool call against the appropriate MCP server."""
    t_name = tool_call.function.name
    t_args = json.loads(tool_call.function.arguments)
    
    params = _tool_to_server.get(t_name)
    if not params:
        return (tool_call.id, f"Error: Tool '{t_name}' is not mapped to any MCP server.")

    logger.info(f"[MCP] Executing '{t_name}' via npx...")
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tool_result = await session.call_tool(t_name, t_args)
                
                raw_content = ""
                for part in tool_result.content:
                    if hasattr(part, "text"):
                        raw_content += part.text
                        
                return (tool_call.id, raw_content)
    except Exception as e:
        logger.error(f"[MCP] Tool call '{t_name}' failed: {e}")
        return (tool_call.id, f"Error executing {t_name}: {str(e)}")

async def execute_mcp_tools(tool_calls):
    """Executes multiple tool calls."""
    # stdio clients can be spawned per-request or we could maintain a pool.
    # For simplicity and isolation, we spawn a process per tool call.
    tasks = [
        _call_single_tool(tc) for tc in tool_calls
    ]
    results = await asyncio.gather(*tasks)
    return results
