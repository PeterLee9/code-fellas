"""
Hybrid RAG chain: combines vector search over document chunks
with smart structured queries over the zoning database.
"""
from __future__ import annotations

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.models.database import ZoningRegulationDB, OfficialPlanPolicyDB
from backend.rag.chunker import regulation_to_text, policy_to_text
from backend.rag.retriever import retrieve_similar_chunks


def _get_text_content(response) -> str:
    """Extract text from an LLM response, handling both string and list-of-parts formats."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


class QueryIntent:
    """Parsed user intent from query analysis."""

    def __init__(self, raw: dict[str, Any]):
        self.municipality: str | None = raw.get("municipality")
        self.zone_category: str | None = raw.get("zone_category")
        self.metric: str | None = raw.get("metric")
        self.intent: str = raw.get("intent", "general")
        self.search_query: str = raw.get("search_query", "")


async def analyze_query(question: str) -> QueryIntent:
    """Use a fast LLM call to extract structured intent from a user question."""
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.agent_model,
        google_api_key=settings.google_api_key,
        temperature=0.0,
    )

    prompt = f"""Extract the user's intent from this question about Canadian zoning regulations.
Return ONLY a JSON object with these fields (use null if not applicable):

- "municipality": string or null (the city being asked about, e.g. "Toronto", "Ottawa")
- "zone_category": string or null (one of: residential, commercial, industrial, mixed_use, institutional, open_space, agricultural)
- "metric": string or null (the specific metric being asked about; one of: min_lot_size_sqm, max_building_height_m, max_stories, max_lot_coverage_pct, parking_spaces_per_unit, density_units_per_hectare, restrictiveness_score, permitted_dwelling_types, min_lot_frontage_m, max_floor_area_ratio, min_landscaped_area_pct, bylaw_number, permitted_commercial_uses, prohibited_uses, home_occupation_permitted, secondary_suite_permitted, short_term_rental_permitted, inclusionary_zoning_pct, overlay_district, min_unit_size_sqm)
- "intent": string (one of: "lookup", "comparison", "ranking", "general")
- "search_query": string (a concise version of the question optimized for semantic search)

Question: {question}

JSON:"""

    response = await llm.ainvoke(prompt)
    content = _get_text_content(response).strip()

    import json
    try:
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return QueryIntent(json.loads(content))
    except (json.JSONDecodeError, IndexError):
        return QueryIntent({"intent": "general", "search_query": question})


METRIC_COLUMNS: dict[str, str] = {
    "min_lot_size_sqm": "min_lot_size_sqm",
    "max_building_height_m": "max_building_height_m",
    "max_stories": "max_stories",
    "max_lot_coverage_pct": "max_lot_coverage_pct",
    "parking_spaces_per_unit": "parking_spaces_per_unit",
    "density_units_per_hectare": "density_units_per_hectare",
    "restrictiveness_score": "restrictiveness_score",
    "min_lot_frontage_m": "min_lot_frontage_m",
    "max_floor_area_ratio": "max_floor_area_ratio",
    "min_landscaped_area_pct": "min_landscaped_area_pct",
    "inclusionary_zoning_pct": "inclusionary_zoning_pct",
    "min_unit_size_sqm": "min_unit_size_sqm",
}


async def get_structured_context(
    db: AsyncSession,
    intent: QueryIntent,
    limit: int = 30,
) -> tuple[list[str], list[dict]]:
    """Query the structured zoning_regulations table based on parsed intent."""
    stmt = select(ZoningRegulationDB)

    if intent.municipality:
        stmt = stmt.where(ZoningRegulationDB.municipality.ilike(f"%{intent.municipality}%"))
    if intent.zone_category:
        stmt = stmt.where(ZoningRegulationDB.zone_category == intent.zone_category)

    col_attr = METRIC_COLUMNS.get(intent.metric or "") if intent.metric else None
    if col_attr and hasattr(ZoningRegulationDB, col_attr):
        col = getattr(ZoningRegulationDB, col_attr)
        stmt = stmt.where(col.isnot(None))
        stmt = stmt.order_by(col.desc(), ZoningRegulationDB.municipality)
    else:
        stmt = stmt.order_by(ZoningRegulationDB.municipality, ZoningRegulationDB.zone_code)

    stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    context_parts = []
    sources = []

    for r in rows:
        context_parts.append(regulation_to_text(r))
        sources.append({
            "municipality": r.municipality,
            "zone_code": r.zone_code,
            "source_url": r.source_url,
            "source_document": r.source_document,
            "type": "structured",
        })

    # Also fetch official plan policies if relevant
    if intent.intent == "general" or intent.zone_category is None:
        pol_stmt = select(OfficialPlanPolicyDB)
        if intent.municipality:
            pol_stmt = pol_stmt.where(OfficialPlanPolicyDB.municipality.ilike(f"%{intent.municipality}%"))
        pol_stmt = pol_stmt.limit(10)
        pol_result = await db.execute(pol_stmt)
        pol_rows = pol_result.scalars().all()
        for p in pol_rows:
            context_parts.append(policy_to_text(p))
            sources.append({
                "municipality": p.municipality,
                "zone_code": f"Policy: {p.policy_area}",
                "source_url": p.source_url,
                "source_document": p.source_document,
                "type": "policy",
            })

    return context_parts, sources


async def hybrid_rag_answer(
    question: str,
    db: AsyncSession,
    municipality: str | None = None,
) -> dict[str, Any]:
    """
    Answer a question using hybrid retrieval:
    1. Analyze the query to extract intent
    2. Vector search for semantically relevant document chunks
    3. Structured SQL query for relevant zoning data
    4. Merge context and generate answer
    """
    # Step 1: Analyze the query
    intent = await analyze_query(question)
    if municipality:
        intent.municipality = municipality

    # Step 2: Vector search on document_chunks
    search_query = intent.search_query or question
    vector_chunks = await retrieve_similar_chunks(
        query=search_query,
        db=db,
        municipality=intent.municipality,
        top_k=8,
    )

    vector_context = []
    vector_sources = []
    for c in vector_chunks:
        vector_context.append(f"[{c['municipality']} - {c['source_document']}]\n{c['chunk_text']}")
        vector_sources.append({
            "municipality": c["municipality"],
            "zone_code": "",
            "source_url": c["source_url"],
            "source_document": c["source_document"],
            "type": "document_chunk",
        })

    # Step 3: Structured query on zoning_regulations
    structured_context, structured_sources = await get_structured_context(db, intent)

    # Step 4: Merge context
    all_context_parts = []
    if structured_context:
        all_context_parts.append("## Structured Zoning Data\n" + "\n\n".join(structured_context))
    if vector_context:
        all_context_parts.append("## Source Documents\n" + "\n\n---\n\n".join(vector_context))

    combined_context = "\n\n".join(all_context_parts)
    if not combined_context.strip():
        combined_context = "No relevant data found in the knowledge base for this query."

    # Step 5: Generate answer
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.agent_model,
        google_api_key=settings.google_api_key,
        temperature=0.3,
    )

    system_prompt = """You are ZoneMap Canada's AI assistant, an expert in Canadian municipal zoning bylaws and land use regulations.

Rules:
- Answer based ONLY on the provided context data. Do not make up information.
- Always cite specific zone codes (e.g., R1, C2), municipalities, and numerical values.
- When comparing, use actual numbers from the data.
- If data is missing or insufficient, explicitly say so rather than guessing.
- Format your response with markdown: use **bold** for key metrics, bullet lists for comparisons, and headers for sections.
- Keep answers focused and concise but thorough."""

    prompt = f"""{system_prompt}

CONTEXT DATA:
{combined_context}

USER QUESTION: {question}

Provide a clear, well-structured answer with specific data points and citations."""

    response = await llm.ainvoke(prompt)

    # Deduplicate sources
    seen = set()
    unique_sources = []
    for s in structured_sources + vector_sources:
        key = f"{s['municipality']}:{s.get('zone_code', '')}:{s.get('source_document', '')}"
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {
        "answer": _get_text_content(response),
        "sources": unique_sources[:15],
    }
