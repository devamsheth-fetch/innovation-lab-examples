import os
import json
import time
from datetime import datetime, timezone
from uuid import uuid4
from dotenv import load_dotenv
from uagents import Context, Protocol
from openai import AsyncOpenAI

from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

from mcp_client import execute_mcp_tools
from db import db_manager

load_dotenv()

_openai_client = AsyncOpenAI(
    api_key=os.getenv("ASI1_API_KEY"),
    base_url="https://api.asi1.ai/v1",
)

chat_proto = Protocol(spec=chat_protocol_spec)

SYSTEM_PROMPT = """You are the Bookshelf Agent, a Personal Literary Concierge.

## Your Mission
Bridge the physical and digital literary worlds. You help users identify books, manage their persistent semantic memory (Read and Wishlist), and find local library availability.

## Core Workflows
1. **Ingestion & Identification:**
   - If the user provides a book title, author, ISBN, or an image URL of a book cover, identify the book.
   - You MUST extract the Title, Author, and ISBN.
   - If an image URL is provided, use your multimodal vision capability to read the cover/spine.

2. **Enrichment & Database Management:**
   - Once identified, use Open Library MCP tools to fetch the summary and metadata.
   - Ask the user if they want to add it to their "Read List" or "Wishlist".
   - Once they confirm, use the `db_manager_insert_book` tool to save it to their persistent PostgreSQL database.

3. **Personalization (Vibe Checks):**
   - If the user asks for a recommendation or "Vibe Check", use the `db_manager_search_similar` tool to query their database.
   - Provide a "3-point summary" based on their past Read List.

4. **Local Logistics:**
   - If the user asks where to find a book physically, use the Google Maps MCP to search for libraries near them (default to San Jose / Fremont area if they don't specify).

5. **PDF Report (Book Snapshot):**
   - If the user requests a PDF or Book Snapshot, use the `generate_pdf_snapshot` tool.

6. **Manual Wipe:**
   - If the user explicitly asks to "Manual Wipe" their data, use the `db_manager_manual_wipe` tool.

## Tool Execution Rules
- Execute tools when you need to interact with Open Library, Google Maps, the local Postgres DB, or generate a PDF.
- Summarize tool results concisely for the user.
- DO NOT invent book data. Rely on the Open Library tools.
"""

def _build_send_response(text: str, end_session: bool = False):
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=content,
    )

# Internal tools exposed to the LLM (bypassing MCP for local DB/PDF actions)
LOCAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "db_manager_insert_book",
            "description": "Inserts a book into the user's persistent PostgreSQL database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "isbn": {"type": "string"},
                    "title": {"type": "string"},
                    "author": {"type": "string"},
                    "summary": {"type": "string"},
                    "status": {"type": "string", "enum": ["read", "wishlist"]}
                },
                "required": ["isbn", "title", "author", "summary", "status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "db_manager_search_similar",
            "description": "Searches the user's persistent database for similar books using semantic vector search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The vibe, genre, or description to search for."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "db_manager_manual_wipe",
            "description": "Wipes the user's persistent database (TRUNCATE). Only use if explicitly requested.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_pdf_snapshot",
            "description": "Generates a PDF snapshot of a book and its library locations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "author": {"type": "string"},
                    "summary": {"type": "string"},
                    "library_locations": {"type": "string", "description": "Text describing local availability."}
                },
                "required": ["title", "author", "summary", "library_locations"]
            }
        }
    }
]

async def _execute_local_tool(ctx: Context, tool_call):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    if name == "db_manager_insert_book":
        await db_manager.insert_book(args['isbn'], args['title'], args['author'], args['summary'], args['status'])
        return "Book inserted successfully."
        
    elif name == "db_manager_search_similar":
        results = await db_manager.search_similar_books(args['query'])
        return json.dumps(results)
        
    elif name == "db_manager_manual_wipe":
        db_manager.manual_wipe()
        return "Database wiped successfully."
        
    elif name == "generate_pdf_snapshot":
        # Generate simple PDF
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        import os
        
        pdf_path = f"snapshot_{uuid4().hex[:8]}.pdf"
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.drawString(100, 750, f"Book Snapshot: {args['title']}")
        c.drawString(100, 730, f"Author: {args['author']}")
        c.drawString(100, 700, "Summary:")
        
        # Simple text wrapping for summary
        textobject = c.beginText(100, 680)
        textobject.setFont("Helvetica", 10)
        import textwrap
        for line in textwrap.wrap(args['summary'], width=80):
            textobject.textLine(line)
        c.drawText(textobject)
        
        c.drawString(100, 400, "Local Availability:")
        textobject = c.beginText(100, 380)
        for line in textwrap.wrap(args['library_locations'], width=80):
            textobject.textLine(line)
        c.drawText(textobject)
        
        c.save()
        return f"PDF snapshot generated at {os.path.abspath(pdf_path)}"
    
    return "Local tool executed."

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id))

    user_input = ""
    is_start_session = False
    
    # Simple extraction of text. To support images natively in chat via URL, we'll parse URLs from text.
    for item in msg.content:
        if isinstance(item, TextContent):
            user_input += item.text
        elif isinstance(item, StartSessionContent):
            is_start_session = True

    if is_start_session and not user_input:
        ctx.logger.info(f"[{sender[:16]}...] Session started.")
        return

    # Check for image URL in user_input
    # If a URL ending in jpg/png is found, we can format it as a multimodal message
    import re
    img_urls = re.findall(r'(https?://[^\s]+(?:jpg|jpeg|png|webp))', user_input, re.IGNORECASE)

    history_key = f"history_{sender}"
    session_data = ctx.storage.get(history_key)
    current_time = time.time()

    if session_data:
        messages = session_data.get("messages", [])
        last_active = session_data.get("last_active", current_time)
        if current_time - last_active > 7200:
            conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        else:
            conversation_history = messages
    else:
        conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    if len(conversation_history) > 15:
        conversation_history = [conversation_history[0]] + conversation_history[-14:]

    # Construct user message with or without image
    if img_urls:
        content_array = [{"type": "text", "text": user_input}]
        for url in img_urls:
            content_array.append({"type": "image_url", "image_url": {"url": url}})
        conversation_history.append({"role": "user", "content": content_array})
    else:
        conversation_history.append({"role": "user", "content": user_input})

    mcp_tools_metadata = ctx.storage.get("tools_metadata") or []
    
    # Combine MCP tools and Local tools
    all_tools = mcp_tools_metadata + LOCAL_TOOLS

    iteration = 0
    try:
        while True:
            if iteration >= 5:
                break
            iteration += 1

            response = await _openai_client.chat.completions.create(
                model=os.getenv("ASI1_MODEL", "asi1"),
                messages=conversation_history,
                tools=all_tools if all_tools else None,
            )
            assistant_msg = response.choices[0].message

            assistant_dict = {"role": "assistant", "content": assistant_msg.content}
            if assistant_msg.tool_calls:
                assistant_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    } for tc in assistant_msg.tool_calls
                ]
            conversation_history.append(assistant_dict)

            if not assistant_msg.tool_calls:
                break

            # Execute tool calls
            mcp_tool_calls = []
            for tc in assistant_msg.tool_calls:
                # Check if it's a local tool
                if any(lt["function"]["name"] == tc.function.name for lt in LOCAL_TOOLS):
                    ctx.logger.info(f"Executing local tool: {tc.function.name}")
                    res = await _execute_local_tool(ctx, tc)
                    conversation_history.append({"role": "tool", "tool_call_id": tc.id, "content": res})
                else:
                    mcp_tool_calls.append(tc)
                    
            if mcp_tool_calls:
                mcp_results = await execute_mcp_tools(mcp_tool_calls)
                for tool_call_id, content in mcp_results:
                    conversation_history.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

        # Persist history
        ctx.storage.set(history_key, {"messages": conversation_history, "last_active": time.time()})

        final_answer = assistant_msg.content or "I have finished processing your request."
        await ctx.send(sender, _build_send_response(final_answer, end_session=False))

    except Exception as e:
        ctx.logger.error(f"Reasoning loop error for {sender[:16]}...: {e}")
        await ctx.send(sender, _build_send_response("Sorry, an internal error occurred. Please try again.", end_session=False))

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass
