from datetime import datetime
from uuid import uuid4
import os
import sys
from uagents import Agent, Context, Model, Protocol
from pydantic import Field
from dotenv import load_dotenv

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.llm import ASI1LLM
from src.database import db
from src.workflow import build_workflow
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)
from uagents_core.utils.registration import (
    register_chat_agent,
    RegistrationRequestCredentials,
)

load_dotenv()

# Compile the LangGraph Orchestration
research_graph = build_workflow()

# Monkey-patch uagents to fix Pydantic v2 compatibility issue with AgentInfo
try:
    from uagents.models import AgentInfo
    if "agent_type" in AgentInfo.model_fields:
        AgentInfo.model_fields["agent_type"].default = "agent"
        AgentInfo.model_rebuild(force=True)
except Exception:
    pass

# Architect Configuration
AGENT_NAME = "Deep_Research_Architect"
SEED_PHRASE = os.getenv("AGENT_SEED_PHRASE", "r2e2_architect_seed_phrase")
AGENTVERSE_KEY = os.getenv("AGENTVERSE_API_KEY")

architect_agent = Agent(
    name=AGENT_NAME,
    seed=SEED_PHRASE,
    port=8001,
    mailbox=True,
)

# ASI1 configuration for Architect
architect_llm = ASI1LLM(model="asi1", temperature=0.1)

# Protocol Setup
protocol = Protocol(spec=chat_protocol_spec)

# Agent Addresses (Can be configured via .env or hardcoded for local test)
CRAWLER_ADDRESS = os.getenv("CRAWLER_ADDRESS", "agent1q...") # Placeholder
STRATEGIST_ADDRESS = os.getenv("STRATEGIST_ADDRESS", "agent1q...") # Placeholder

def route_intent(user_input: str) -> str:
    """Uses ASI1 to analyze incoming messages including text, URLs, PDF references, or Code snippets."""
    prompt = f"""
    [Identity Context]
    You are the R2E2 Architect, the primary gateway and traffic controller for a multi-agent research system.
    
    [World Context]
    The system consists of:
    1. Crawler: Handles data ingestion (ArXiv/OpenAlex URLs, PDF summaries, paper IDs, or research code analysis).
    2. Strategist: Handles Q&A, graph gap analysis, and final report generation based on existing data.
    All research state is stored in a Memgraph database.
    
    [Task Context]
    Analyze the user input and orchestrate the corresponding agent:
    - TRIGGER_CRAWLER: Orchestrate the Crawler Agent for data ingestion (ArXiv/URLs/PDFs/Code).
    - TRIGGER_STRATEGIST: Orchestrate the Strategist Agent for graph-based Q&A and gap analysis.
    - PURGE: Reset the internal research state.
    
    [Example Context]
    User: "Ingest this paper: https://arxiv.org/abs/2401.00001" -> TRIGGER_CRAWLER
    User: "Run the Strategist on our current graph." -> TRIGGER_STRATEGIST
    
    [Constraint Context]
    User Input: "{user_input}"
    Respond ONLY with the intent name (e.g., TRIGGER_CRAWLER).
    """
    try:
        response = architect_llm.invoke(prompt)
        return response.strip().upper()
    except Exception as e:
        return f"ERROR: {e}"

@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )
    
    # Extract text content
    user_input = ""
    for item in msg.content:
        if isinstance(item, TextContent):
            user_input += item.text
    
    # Orchestrate the entire research pipeline via LangGraph
    # Architect is the entry uAgent, and also the first node in the graph
    # Orchestrate the entire research pipeline via LangGraph
    
    # Maintain paper history in ctx to persist across messages for this session
    if not ctx.storage.has("paper_history"):
        ctx.storage.set("paper_history", [])
    
    state = {
        "user_input": user_input,
        "seed_id": None,
        "oa_seed_id": None,
        "intent": None,
        "seed_query": None,
        "question": None,
        "output": None,
        "format": "markdown",
        "search_count": 0,
        "cache_hit": False,
        "paper_history": ctx.storage.get("paper_history") or [],
        "history": []
    }
    
    try:
        result = research_graph.invoke(state)
        # Update the persistent history with any new paper IDs found by the Architect
        ctx.storage.set("paper_history", result.get("paper_history", ctx.storage.get("paper_history")))
        
        response_text = result.get("output", "Pipeline completed with no output.")
    except Exception as e:
        ctx.logger.error(f"R2E2 Graph execution error: {e}")
        response_text = f"An error occurred during pipeline execution: {e}"

    await ctx.send(sender, ChatMessage(
        timestamp=datetime.now(),
        msg_id=uuid4(),
        content=[
            TextContent(type="text", text=response_text),
            EndSessionContent(type="end-session"),
        ]
    ))

@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass

architect_agent.include(protocol, publish_manifest=True)

README = """# R2E2 Architect Agent
Autonomous research gateway using ASI1 LLM.
"""

@architect_agent.on_event("startup")
async def startup_handler(ctx: Context):
    ctx.logger.info(f"🚀 Architect Agent starting at {ctx.agent.address}")
    if AGENTVERSE_KEY:
        try:
            register_chat_agent(
                AGENT_NAME,
                architect_agent._endpoints[0].url if architect_agent._endpoints else "http://localhost:8001/submit",
                active=True,
                credentials=RegistrationRequestCredentials(
                    agentverse_api_key=AGENTVERSE_KEY,
                    agent_seed_phrase=SEED_PHRASE,
                ),
                readme=README,
                description="Gateway agent for R2E2 research team. Routes tasks to Crawler and Strategist."
            )
            ctx.logger.info("✅ Registered with Agentverse")
        except Exception as e:
            ctx.logger.error(f"Failed to register with Agentverse: {e}")

if __name__ == "__main__":
    architect_agent.run()
