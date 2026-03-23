"""
Agentic RAG chat: LangGraph agent with SQL, vector search, and web search tools.

Replaces the old fixed-pipeline hybrid_rag_answer with a fully agentic loop
where the LLM autonomously decides which tools to call.
"""
from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.rag.embeddings import embed_query

# ---------------------------------------------------------------------------
# DB schema reference for the LLM
# ---------------------------------------------------------------------------
DB_SCHEMA = """\
## Database Schema

### zoning_regulations
Core table with one row per zone per municipality.
Columns: id, municipality, province, zone_code, zone_name, zone_category (residential/commercial/industrial/mixed_use/institutional/open_space/agricultural/other),
min_lot_size_sqm, max_building_height_m, max_stories, max_lot_coverage_pct,
min_front_setback_m, min_rear_setback_m, min_side_setback_m,
parking_spaces_per_unit, permitted_dwelling_types (JSONB array), max_units_per_lot,
density_units_per_hectare, min_lot_frontage_m, max_floor_area_ratio,
min_landscaped_area_pct, bylaw_number, bylaw_effective_date,
permitted_commercial_uses (JSONB array), prohibited_uses (JSONB array),
home_occupation_permitted (bool), secondary_suite_permitted (bool),
short_term_rental_permitted (bool), inclusionary_zoning_pct,
overlay_district, min_unit_size_sqm, additional_regulations (JSONB),
source_url, source_document, extraction_date, raw_text_excerpt,
confidence_score, needs_review, restrictiveness_score

### official_plan_policies
One row per policy area per municipality.
Columns: id, municipality, province, policy_area, land_use_designation,
growth_targets, density_targets, transit_policy, affordable_housing_policy,
heritage_conservation, permitted_uses_summary,
source_url, source_document, extraction_date, raw_text_excerpt,
confidence_score, needs_review

### municipalities
One row per municipality.
Columns: id, name, province, population, latitude, longitude,
geometry (JSONB GeoJSON), data_sources (JSONB array), last_updated

### document_chunks
Raw text and multimodal chunks used for RAG.
Columns: id, municipality, source_url, source_document, chunk_text,
chunk_index, chunk_type (text/pdf_page/image), embedding (vector 3072)

### scrape_logs
Pipeline run history.
Columns: id, municipality, url, scraped_at, status, documents_found, error_message
"""

SYSTEM_PROMPT = f"""\
You are ZoneMap Canada's AI assistant, an expert in Canadian municipal zoning bylaws and land use regulations.

You have access to a PostgreSQL database with comprehensive zoning data. Use the tools provided to answer user questions accurately.

## Tool Selection Guide

- **query_database**: Use for structured data questions -- counts, averages, comparisons, rankings, filtering, aggregations. Write standard PostgreSQL SQL. Always use SELECT only.
- **search_knowledge_base**: Use for conceptual questions, policy intent, bylaw text details, or to find relevant document passages via vector similarity. Also use this to ENRICH answers from SQL with additional context from source documents.
- **search_web**: Use ONLY for general zoning concept questions (e.g. "what is inclusionary zoning?"), NOT for questions about specific municipalities.

## IMPORTANT: Use Multiple Tools for Rich Answers

For most municipality-specific questions, you should call BOTH query_database AND search_knowledge_base:
1. First use query_database to get the structured data (zone codes, numbers, regulations).
2. Then use search_knowledge_base to find relevant bylaw text, policy context, or document excerpts that add depth to your answer.

This gives the user both precise data AND the context/reasoning from the actual bylaw documents.

Example: "What residential zones does Toronto have?"
- Step 1: SQL to get zone codes and names from zoning_regulations
- Step 2: search_knowledge_base("Toronto residential zones permitted uses setbacks") to get bylaw text with details
- Step 3: Combine both into a comprehensive answer

## Critical Rule: Only Answer From Scraped Data

You must ONLY answer questions about municipalities that already exist in the database. Before answering any municipality-specific question:
1. First query the `municipalities` table to check if the city exists: `SELECT name FROM municipalities`
2. If the municipality is NOT in the database, do NOT attempt to answer using web search or speculation. Instead respond:
   "**[City name] hasn't been added to ZoneMap yet.** To get zoning data for this city, go to the **Admin** page and run the scraping pipeline. Once the data is loaded, I'll be able to answer your questions about it."
3. NEVER use search_web to look up zoning data for a specific municipality. Web search is only for general zoning concepts.

## Rules

- You may call multiple tools, or the same tool multiple times, before answering.
- For most questions, use at LEAST 2 tool calls to provide comprehensive answers.
- Always cite specific zone codes (e.g. R1, C2), municipalities, and numerical values.
- When comparing municipalities, use actual numbers from query results.
- If data is missing or insufficient, say so rather than guessing.
- Format your final response with markdown: **bold** for key metrics, bullet lists for comparisons, headers for sections.
- Keep answers focused, concise but thorough.
- For JSONB array columns (permitted_dwelling_types, permitted_commercial_uses, prohibited_uses), query them with PostgreSQL JSONB operators -- e.g. `jsonb_array_length(permitted_dwelling_types)` or `permitted_dwelling_types::text ILIKE '%townhouse%'`.

{DB_SCHEMA}
"""

_UNSAFE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|COPY)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------
class ChatAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    sources: list[dict]


# ---------------------------------------------------------------------------
# Tool implementations (session-bound via closure)
# ---------------------------------------------------------------------------
def _build_tools(db: AsyncSession):
    """Build tool functions with the DB session bound via closure."""

    @tool
    async def query_database(sql: str) -> str:
        """Execute a read-only SQL SELECT query against the zoning database.
        Use this for structured data questions: counts, averages, comparisons,
        rankings, filtering. Write standard PostgreSQL SQL.
        Only SELECT statements are allowed."""
        sql_stripped = sql.strip().rstrip(";")
        if _UNSAFE_PATTERN.search(sql_stripped):
            return "ERROR: Only SELECT queries are allowed. Your query contained a prohibited keyword."

        if not sql_stripped.upper().lstrip().startswith("SELECT"):
            return "ERROR: Query must start with SELECT."

        if "LIMIT" not in sql_stripped.upper():
            sql_stripped += " LIMIT 50"

        try:
            result = await db.execute(text(sql_stripped))
            rows = result.fetchall()
            columns = list(result.keys()) if result.keys() else []
        except Exception as e:
            err = str(e)
            if len(err) > 300:
                err = err[:300] + "..."
            return f"SQL ERROR: {err}\nPlease fix the query and try again."

        if not rows:
            return "Query returned 0 rows."

        header = " | ".join(columns)
        sep = " | ".join("---" for _ in columns)
        lines = [header, sep]
        for row in rows[:50]:
            cells = []
            for val in row:
                s = str(val) if val is not None else ""
                if len(s) > 120:
                    s = s[:120] + "..."
                cells.append(s)
            lines.append(" | ".join(cells))

        return f"Query returned {len(rows)} rows:\n\n" + "\n".join(lines)

    @tool
    async def search_knowledge_base(query: str, municipality: str = "") -> str:
        """Search the embedded document knowledge base using semantic similarity.
        Use this for conceptual questions about bylaw text, policy intent, or
        when you need to find relevant document passages.
        Optionally filter by municipality name."""
        muni = municipality if municipality else None
        query_embedding = await embed_query(query)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        if muni:
            sql_stmt = text(
                "SELECT id, municipality, source_url, source_document, "
                "chunk_text, chunk_index, chunk_type, "
                "embedding <=> cast(:emb as vector) AS distance "
                "FROM document_chunks "
                "WHERE embedding IS NOT NULL "
                "AND municipality ILIKE :municipality "
                "ORDER BY embedding <=> cast(:emb as vector) "
                "LIMIT :top_k"
            )
            params = {"emb": embedding_str, "municipality": f"%{muni}%", "top_k": 8}
        else:
            sql_stmt = text(
                "SELECT id, municipality, source_url, source_document, "
                "chunk_text, chunk_index, chunk_type, "
                "embedding <=> cast(:emb as vector) AS distance "
                "FROM document_chunks "
                "WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> cast(:emb as vector) "
                "LIMIT :top_k"
            )
            params = {"emb": embedding_str, "top_k": 8}

        try:
            result = await db.execute(sql_stmt, params)
            rows = result.fetchall()
        except Exception as e:
            return f"Vector search error: {e}"

        if not rows:
            return "No relevant documents found."

        parts = []
        for row in rows:
            chunk_type = row[6]
            source_url = row[2] or ""
            source_doc = row[3] or ""
            label = f"[{row[1]} - {source_doc}]"
            if chunk_type == "image":
                label += " (Zoning Map)"
            elif chunk_type == "pdf_page":
                label += " (PDF Page)"
            if source_url:
                label += f"\nSource: {source_url}"
            parts.append(f"{label}\n{row[4]}")

        return f"Found {len(rows)} relevant chunks:\n\n" + "\n\n---\n\n".join(parts)

    @tool
    async def search_web(query: str) -> str:
        """Search the web for zoning information not yet in the database.
        Use this only when database data is insufficient or the user asks
        about a municipality not yet scraped."""
        from tavily import TavilyClient

        try:
            settings = get_settings()
            client = TavilyClient(api_key=settings.tavily_api_key)
            response = client.search(
                query,
                max_results=5,
                search_depth="advanced",
            )
            results = []
            for r in response.get("results", []):
                results.append(
                    f"**{r['title']}**\n{r['url']}\n{r.get('content', '')[:400]}"
                )
            return "\n\n---\n\n".join(results) if results else "No web results found."
        except Exception as e:
            return f"Web search error: {e}"

    return [query_database, search_knowledge_base, search_web]


# ---------------------------------------------------------------------------
# Source extraction helpers
# ---------------------------------------------------------------------------
_URL_PATTERN = re.compile(r"https?://[^\s\)\]\",>]+")


def _extract_sources_from_messages(messages: list) -> list[dict]:
    """Extract source metadata from tool calls AND their results."""
    sources: list[dict] = []
    seen: set[str] = set()

    tool_call_map: dict[str, tuple[str, dict]] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id", "")
                if tc_id:
                    tool_call_map[tc_id] = (tc.get("name", ""), tc.get("args", {}))

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})

                if name == "query_database":
                    sql_text = args.get("sql", "")
                    tables_mentioned = []
                    for tbl in ("zoning_regulations", "official_plan_policies", "municipalities"):
                        if tbl in sql_text.lower():
                            tables_mentioned.append(tbl)
                    if tables_mentioned:
                        key = f"sql:{sql_text[:80]}"
                        if key not in seen:
                            seen.add(key)
                            sources.append({
                                "municipality": "",
                                "zone_code": "",
                                "source_url": "",
                                "source_document": f"SQL query on {', '.join(tables_mentioned)}",
                                "type": "sql",
                                "sql": sql_text,
                            })

        elif isinstance(msg, ToolMessage):
            tc_id = msg.tool_call_id if hasattr(msg, "tool_call_id") else ""
            tool_info = tool_call_map.get(tc_id)
            if not tool_info:
                continue
            name, args = tool_info
            output = msg.content if hasattr(msg, "content") else ""
            if isinstance(output, list):
                output = " ".join(str(p) for p in output)
            output = str(output) if output else ""

            if name == "search_knowledge_base":
                urls = _URL_PATTERN.findall(output)
                unique_urls: list[str] = []
                for u in urls:
                    u = u.rstrip(".,;:)")
                    if u not in unique_urls:
                        unique_urls.append(u)
                if unique_urls:
                    for url in unique_urls[:4]:
                        key = f"kb:{url}"
                        if key not in seen:
                            seen.add(key)
                            try:
                                from urllib.parse import urlparse
                                domain = urlparse(url).netloc
                            except Exception:
                                domain = url[:40]
                            sources.append({
                                "municipality": args.get("municipality", ""),
                                "zone_code": "",
                                "source_url": url,
                                "source_document": domain,
                                "type": "document_chunk",
                            })
                else:
                    muni = args.get("municipality", "")
                    key = f"kb:{muni}:{args.get('query', '')[:50]}"
                    if key not in seen:
                        seen.add(key)
                        sources.append({
                            "municipality": muni,
                            "zone_code": "",
                            "source_url": "",
                            "source_document": "Knowledge Base",
                            "type": "document_chunk",
                        })

            elif name == "search_web":
                urls = _URL_PATTERN.findall(output)
                added = 0
                for u in urls:
                    u = u.rstrip(".,;:)")
                    key = f"web:{u}"
                    if key not in seen and added < 3:
                        seen.add(key)
                        try:
                            from urllib.parse import urlparse
                            domain = urlparse(u).netloc
                        except Exception:
                            domain = u[:40]
                        sources.append({
                            "municipality": "",
                            "zone_code": "",
                            "source_url": u,
                            "source_document": domain,
                            "type": "web",
                        })
                        added += 1
                if added == 0:
                    key = f"web:{args.get('query', '')[:50]}"
                    if key not in seen:
                        seen.add(key)
                        sources.append({
                            "municipality": "",
                            "zone_code": "",
                            "source_url": "",
                            "source_document": f"Web: {args.get('query', '')}",
                            "type": "web",
                        })

    return sources


def _get_text_content(msg) -> str:
    """Extract text from a message, handling list-of-parts content."""
    content = msg.content if hasattr(msg, "content") else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p if isinstance(p, str) else p.get("text", "") if isinstance(p, dict) else ""
            for p in content
        )
    return str(content) if content else ""


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def _build_graph(tools: list):
    """Build the LangGraph agent for chat."""

    tool_node = ToolNode(tools)

    llm = ChatGoogleGenerativeAI(
        model=get_settings().agent_model,
        google_api_key=get_settings().google_api_key,
        temperature=0.3,
    ).bind_tools(tools)

    async def agent_node(state: ChatAgentState) -> dict:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: ChatAgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "done"

    graph = StateGraph(ChatAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "done": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


TOOL_DESCRIPTIONS = {
    "query_database": "Querying the database",
    "search_knowledge_base": "Searching knowledge base",
    "search_web": "Searching the web",
}


def _summarize_tool_result(name: str, output: str) -> str:
    """Create a brief human-readable summary of a tool's result."""
    if "returned 0 rows" in output or "No relevant documents" in output:
        return "No results found"
    if "ERROR" in output or "error" in output.lower()[:30]:
        return "Encountered an error, retrying..."
    if name == "query_database":
        match = re.search(r"returned (\d+) rows", output)
        if match:
            return f"Found {match.group(1)} database records"
    if name == "search_knowledge_base":
        match = re.search(r"Found (\d+) relevant chunks", output)
        if match:
            return f"Found {match.group(1)} relevant document passages"
    if name == "search_web":
        count = output.count("---")
        if count > 0:
            return f"Found {count + 1} web results"
    return "Processing results..."


def _extract_tool_detail(name: str, args: dict) -> str:
    """Extract a human-readable detail string from tool arguments."""
    if name == "query_database":
        sql = args.get("sql", "")
        if len(sql) > 200:
            sql = sql[:200] + "..."
        return sql
    if name == "search_knowledge_base":
        q = args.get("query", "")
        muni = args.get("municipality", "")
        detail = f'"{q}"'
        if muni:
            detail += f" in {muni}"
        return detail
    if name == "search_web":
        return f'"{args.get("query", "")}"'
    return ""


def _extract_result_preview(name: str, output: str) -> str:
    """Extract a short preview of the tool output for display."""
    if name == "query_database":
        lines = output.strip().split("\n")
        if len(lines) > 3:
            header = lines[1]
            first_rows = lines[3:6]
            return header + "\n" + "\n".join(first_rows)
    if name == "search_knowledge_base":
        chunks = output.split("---")
        previews = []
        for chunk in chunks[:3]:
            chunk = chunk.strip()
            if chunk.startswith("Found"):
                continue
            label_end = chunk.find("\n")
            if label_end > 0:
                label = chunk[:label_end].strip()
                previews.append(label)
        if previews:
            return "\n".join(previews)
    if name == "search_web":
        titles = re.findall(r"\*\*(.+?)\*\*", output)
        if titles:
            return "\n".join(titles[:3])
    return ""


def _build_history_messages(
    history: list[dict[str, str]] | None,
) -> list[HumanMessage | AIMessage]:
    """Convert frontend chat history into LangChain message objects.

    Only keeps the last 10 turns (20 messages) to stay within context limits.
    """
    if not history:
        return []
    msgs: list[HumanMessage | AIMessage] = []
    trimmed = history[-20:]
    for entry in trimmed:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
async def agentic_chat_answer(
    question: str,
    db: AsyncSession,
    municipality: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Answer a zoning question using the agentic RAG chat agent.

    The agent autonomously decides which tools to call (SQL, vector search,
    web search) and loops until it has enough information to answer.

    Returns {"answer": str, "sources": list[dict]} matching ChatResponse schema.
    """
    tools = _build_tools(db)
    graph = _build_graph(tools)

    user_msg = question
    if municipality:
        user_msg = f"[Context: the user is asking about {municipality}]\n\n{question}"

    history_msgs = _build_history_messages(history)

    initial_state: ChatAgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            *history_msgs,
            HumanMessage(content=user_msg),
        ],
        "sources": [],
    }

    try:
        result = await graph.ainvoke(initial_state, {"recursion_limit": 12})
    except Exception as e:
        return {
            "answer": f"I encountered an error while researching your question: {e}",
            "sources": [],
        }

    last_msg = result["messages"][-1]
    answer = _get_text_content(last_msg)

    sources = _extract_sources_from_messages(result["messages"])

    return {
        "answer": answer,
        "sources": sources[:15],
    }


# ---------------------------------------------------------------------------
# Streaming entry point (SSE)
# ---------------------------------------------------------------------------
async def agentic_chat_stream(
    question: str,
    db: AsyncSession,
    municipality: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream the agent's thought process and answer as SSE events.

    Yields newline-delimited JSON events:
      {"type": "tool_start", "name": "...", "description": "..."}
      {"type": "tool_end",   "name": "...", "summary": "..."}
      {"type": "token",      "content": "..."}
      {"type": "done",       "sources": [...]}
      {"type": "error",      "message": "..."}
    """
    tools = _build_tools(db)
    graph = _build_graph(tools)

    user_msg = question
    if municipality:
        user_msg = f"[Context: the user is asking about {municipality}]\n\n{question}"

    history_msgs = _build_history_messages(history)

    initial_state: ChatAgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            *history_msgs,
            HumanMessage(content=user_msg),
        ],
        "sources": [],
    }

    all_messages: list = []
    answer_text = ""

    try:
        async for event in graph.astream_events(
            initial_state,
            config={"recursion_limit": 12},
            version="v2",
        ):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_tool_start" and name in TOOL_DESCRIPTIONS:
                input_data = event.get("data", {}).get("input", {})
                args = input_data if isinstance(input_data, dict) else {}
                detail = _extract_tool_detail(name, args)
                yield f"data: {json.dumps({'type': 'tool_start', 'name': name, 'description': TOOL_DESCRIPTIONS[name], 'detail': detail})}\n\n"

            elif kind == "on_tool_end" and name in TOOL_DESCRIPTIONS:
                output = event.get("data", {}).get("output", "")
                if not isinstance(output, str):
                    output = str(output)
                summary = _summarize_tool_result(name, output)
                preview = _extract_result_preview(name, output)
                yield f"data: {json.dumps({'type': 'tool_end', 'name': name, 'summary': summary, 'preview': preview})}\n\n"

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content"):
                    text_content = _get_text_content(chunk)
                    if text_content and not (hasattr(chunk, "tool_calls") and chunk.tool_calls):
                        answer_text += text_content
                        yield f"data: {json.dumps({'type': 'token', 'content': text_content})}\n\n"

            elif kind == "on_chain_end" and name == "LangGraph":
                result = event.get("data", {}).get("output", {})
                if isinstance(result, dict) and "messages" in result:
                    all_messages = result["messages"]

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return

    sources = _extract_sources_from_messages(all_messages)
    yield f"data: {json.dumps({'type': 'done', 'sources': sources[:15]})}\n\n"
