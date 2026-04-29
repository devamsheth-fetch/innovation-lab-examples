import logging
import os
from dotenv import load_dotenv
from uagents import Agent, Context
from mcp_client import fetch_mcp_tools
from chat_protocol import chat_proto
from db import db_manager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-24s │ %(message)s",
    datefmt="%H:%M:%S",
)

logging.getLogger("mcp_client").setLevel(logging.INFO)
logging.getLogger("chat_protocol").setLevel(logging.INFO)
logging.getLogger("db").setLevel(logging.INFO)

agent = Agent(
    name="bookshelf_agent",
    port=8001,
    seed=os.getenv("AGENT_SEED_PHRASE", "bookshelf_agent_seed_phrase"),
    mailbox=True,
    enable_agent_inspector=True,
)

@agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info("Bootstrapping: Connecting to database...")
    try:
        db_manager.connect()
    except Exception as e:
        ctx.logger.error(f"Failed to connect to database: {e}")
        
    ctx.logger.info("Bootstrapping: Fetching tools from local MCP servers...")
    try:
        tools_metadata = await fetch_mcp_tools(retries=2, backoff=1.0)
        ctx.storage.set("tools_metadata", tools_metadata)
        ctx.logger.info(f"Bootstrap complete. {len(tools_metadata)} MCP tool(s) loaded.")
    except Exception as e:
        ctx.logger.warning(f"Bootstrap failed to load MCP tools: {e}")
        ctx.storage.set("tools_metadata", [])

agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
