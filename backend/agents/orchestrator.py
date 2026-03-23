"""
LangGraph orchestrator: the main agentic pipeline for discovering,
scraping, extracting, and storing municipal zoning data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sqlalchemy import delete

from backend.agents.scraper_agent import (
    download_and_extract_pdf,
    download_image,
    scrape_webpage,
    search_web,
    get_image_cache,
    clear_image_cache,
)
from backend.agents.pdf_agent import get_pdf_cache, clear_pdf_cache
from backend.agents.extractor_agent import (
    extract_official_plan_from_text,
    extract_zoning_from_text,
)
from backend.models.database import (
    DocumentChunkDB,
    MunicipalityDB,
    OfficialPlanPolicyDB,
    ScrapeLogDB,
    ZoningRegulationDB,
    get_async_engine,
    get_async_session_factory,
)
from backend.config import get_settings
from backend.models.schemas import DocumentChunkCreate
from backend.rag.chunker import chunk_text, structured_to_chunks, pdf_to_page_chunks
from backend.rag.embeddings import chunk_embed_and_store
from backend.services.geocoder import geocode_municipality

TOOLS = [search_web, scrape_webpage, download_and_extract_pdf, download_image]

SYSTEM_PROMPT = """\
You are a municipal data research agent specialising in Canadian zoning bylaws and land use regulations.

Your goal is to find and extract comprehensive zoning regulation data for a given Canadian municipality.

## Workflow

1. **Search** for the municipality's official zoning bylaw page using search_web.
   - Try the municipality's official website first.
   - Also search civic.band (e.g. "civic.band {municipality} zoning") -- it aggregates Canadian municipal data.
   - Also check open data portals (e.g. open.toronto.ca, opendata.vancouver.ca).
2. **Scrape** the main zoning bylaw page using scrape_webpage. Look for:
   - Links to specific zone category pages (residential, commercial, industrial, mixed-use)
   - Links to PDF bylaw documents
   - Zoning tables, permitted-use tables, development-standard tables
3. **Follow links** systematically: scrape sub-pages for each zone category to get complete data.
4. **Download** any PDF zoning bylaws using download_and_extract_pdf.
5. **Download images** of zoning maps, land use maps, or setback diagrams using download_image.
   - When scraping a page, look at the "Images found on this page" section at the bottom.
   - Download images that appear to be zoning maps, land use maps, or technical diagrams.
   - Skip logos, icons, photos of buildings, or decorative images.
   - Look for URLs containing "map", "zoning", "land-use", "diagram", "schedule".
6. After gathering enough text (aim for at least 3-5 different pages/documents), respond with "EXTRACTION_READY" followed by all the zoning text you've collected.

## Known URL Patterns for Canadian Municipalities

Many Canadian municipal sites follow these patterns:
- `/{city}.ca/zoning` or `/{city}.ca/planning/zoning-bylaw`
- `/{city}.ca/city-services/planning-and-development`
- `/open.{city}.ca` for open data portals
- `/www.{city}.ca/bylaws` for bylaw libraries
- `civic.band/{province}/{city}` for aggregated data

## Important Rules

- Focus on OFFICIAL government websites (.ca domain preferred).
- Do NOT scrape the same URL twice -- keep track of which pages you've visited.
- Look for: zoning bylaws, zoning maps, permitted-use tables, development standards.
- Download PDFs of zoning bylaw sections, especially residential and commercial zones.
- Be efficient: don't scrape irrelevant pages.
- Always note the exact URLs you scraped.
- Record bylaw numbers (e.g. "By-law 569-2013") and effective dates when visible.
- Rate limit: wait between requests (the tools handle this).

## Data to Extract

For each zone, look for ALL of these:
- Zone code, zone name, zone category
- Lot size, lot frontage, building height, stories, lot coverage, floor area ratio (FAR)
- Setbacks (front, rear, side), landscaped area requirements
- Parking requirements, density limits, max units per lot
- Permitted dwelling types and commercial uses, prohibited uses
- Secondary suites / ADU permissions, home occupation rules
- Short-term rental rules, inclusionary zoning requirements
- Overlay districts (heritage, transit-oriented, waterfront)
- Minimum unit sizes

## Tables
- When scraping web pages, HTML tables with zoning data are automatically preserved.
- When downloading PDFs, the extractor automatically captures table structures.
- Pay attention to "Schedule" or "Appendix" sections in bylaws -- these often contain the key data tables.

## Data Priority
1. Municipal official websites
2. civic.band (aggregated Canadian municipal data)
3. Open data portals (CKAN, ArcGIS REST)
4. Planning department pages
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    municipality: str
    province: str
    collected_text: str
    source_urls: list[str]
    status: str


def get_agent_llm():
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.agent_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    ).bind_tools(TOOLS)


async def agent_node(state: AgentState) -> dict:
    """The agent decides what to do next."""
    llm = get_agent_llm()
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


def _msg_text(msg) -> str:
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


def _count_tool_rounds(messages: list) -> int:
    """Count how many tool call rounds the agent has completed."""
    return sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)


async def should_continue(state: AgentState) -> str:
    """Decide whether to continue tool calls or move to extraction.
    Requires at least 2 tool call rounds before allowing non-explicit extraction,
    preventing premature extraction on insufficient data."""
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            return "tools"
        content = _msg_text(last_message)
        if "EXTRACTION_READY" in content:
            return "extract"
        if _count_tool_rounds(state["messages"]) < 2:
            return "agent"
    return "extract"


def _extract_urls_from_messages(messages: list) -> list[str]:
    """Parse all URLs that were passed to scrape/download tools from the message history."""
    urls: list[str] = []
    for msg in messages:
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            if tc.get("name") in ("scrape_webpage", "download_and_extract_pdf", "download_image"):
                url = tc.get("args", {}).get("url", "")
                if url and url not in urls:
                    urls.append(url)
    return urls


async def extraction_node(state: AgentState) -> dict:
    """Collect all scraped text and tracked URLs from the message history."""
    collected_parts: list[str] = []

    for msg in state["messages"]:
        text = _msg_text(msg)
        if len(text) > 200:
            collected_parts.append(text)

    source_urls = _extract_urls_from_messages(state["messages"])
    collected_text = "\n\n---\n\n".join(collected_parts)

    return {
        "collected_text": collected_text,
        "source_urls": source_urls,
        "status": "extraction_complete",
    }


def build_agent_graph() -> StateGraph:
    """Build the LangGraph agent pipeline with a recursion limit."""
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("extract", extraction_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "extract": "extract", "agent": "agent"})
    graph.add_edge("tools", "agent")
    graph.add_edge("extract", END)

    return graph.compile()


ProgressCallback = Callable[[str, dict[str, Any]], None] | None


def _notify(callback: ProgressCallback, step: str, data: dict[str, Any] | None = None):
    """Send progress update via callback or print."""
    if callback:
        callback(step, data or {})
    else:
        print(f"  [{step}] {data or ''}")


async def run_pipeline_for_municipality(
    municipality: str,
    province: str,
    database_url: str,
    progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    """
    Run the full agentic pipeline for a single municipality:
    1. Agent discovers and scrapes zoning data
    2. LLM extracts structured regulations + official plan policies
    3. Geocode
    4. Upsert: delete old data, store new data
    5. Chunk + embed for RAG
    """
    _notify(progress_callback, "started", {"municipality": municipality})
    print(f"\n{'='*60}")
    print(f"Processing: {municipality}, {province}")
    print(f"{'='*60}")

    # Step 1: Discover and scrape
    _notify(progress_callback, "discovering", {"municipality": municipality})
    print("[1/6] Running agent to discover and scrape zoning data...")
    graph = build_agent_graph()

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Find and extract zoning bylaw data for {municipality}, {province}, Canada. "
                    f"Search for their official zoning bylaw page, also try civic.band for {municipality}. "
                    f"Scrape relevant pages, follow links to zone-specific sub-pages, "
                    f"and download any PDF zoning documents. "
                    f"If you find any zoning maps or land use diagrams on the pages, download them with download_image. "
                    f"Record the bylaw number and effective date if visible."
                )
            ),
        ],
        "municipality": municipality,
        "province": province,
        "collected_text": "",
        "source_urls": [],
        "status": "started",
    }

    try:
        result = await graph.ainvoke(initial_state, {"recursion_limit": 25})
    except Exception as e:
        err_msg = str(e)
        if "recursion" in err_msg.lower():
            print(f"  [WARN] Agent hit recursion limit -- proceeding with partial data")
            result = initial_state
        else:
            _notify(progress_callback, "failed", {"error": err_msg})
            print(f"  [ERROR] Agent failed: {err_msg}")
            return {"municipality": municipality, "regulations": 0, "status": "agent_error", "error": err_msg}

    collected_text = result.get("collected_text", "")
    source_urls = result.get("source_urls", [])

    if not source_urls:
        source_urls = _extract_urls_from_messages(result.get("messages", []))

    if not collected_text or len(collected_text) < 100:
        _notify(progress_callback, "failed", {"error": "insufficient_data"})
        print("  [WARN] Agent collected insufficient text, skipping extraction")
        return {"municipality": municipality, "regulations": 0, "status": "insufficient_data"}

    _notify(progress_callback, "scraping", {"chars_collected": len(collected_text)})
    print(f"  Collected {len(collected_text)} characters of text from {len(source_urls)} URLs")

    primary_source_url = source_urls[0] if source_urls else f"https://www.{municipality.lower().replace(' ', '')}.ca"

    # Step 2: Extract structured zoning data
    _notify(progress_callback, "extracting", {"municipality": municipality})
    print("[2/6] Extracting structured zoning data...")
    regulations = await extract_zoning_from_text(
        raw_text=collected_text,
        municipality=municipality,
        province=province,
        source_url=primary_source_url,
        source_document=f"{municipality} Zoning Bylaw",
    )
    print(f"  Extracted {len(regulations)} zoning regulations")

    # Step 3: Extract official plan policies
    print("[3/6] Extracting official plan policies...")
    policies = await extract_official_plan_from_text(
        raw_text=collected_text,
        municipality=municipality,
        province=province,
        source_url=primary_source_url,
        source_document=f"{municipality} Official Plan",
    )
    print(f"  Extracted {len(policies)} official plan policies")

    if not regulations and not policies:
        _notify(progress_callback, "failed", {"error": "extraction_failed"})
        return {"municipality": municipality, "regulations": 0, "policies": 0, "status": "extraction_failed"}

    # Step 4: Geocode
    _notify(progress_callback, "geocoding", {"municipality": municipality})
    print("[4/6] Geocoding municipality...")
    coords = await geocode_municipality(municipality, province)
    geocode_failed = coords is None
    if geocode_failed:
        print(f"  [WARN] Geocoding failed for {municipality}, {province} -- using Ontario centroid as fallback")
        lat, lng = 43.65, -79.38
    else:
        lat, lng = coords
    print(f"  Coordinates: {lat}, {lng}")

    # Step 5: Store in database (upsert -- delete old data first)
    _notify(progress_callback, "storing", {"regulations": len(regulations), "policies": len(policies)})
    print("[5/6] Storing in database (upsert)...")
    engine = get_async_engine(database_url)
    Session = get_async_session_factory(engine)

    async with Session() as session:
        await session.execute(
            delete(DocumentChunkDB).where(DocumentChunkDB.municipality == municipality)
        )
        await session.execute(
            delete(ZoningRegulationDB).where(ZoningRegulationDB.municipality == municipality)
        )
        await session.execute(
            delete(OfficialPlanPolicyDB).where(OfficialPlanPolicyDB.municipality == municipality)
        )
        await session.execute(
            delete(MunicipalityDB).where(MunicipalityDB.name == municipality)
        )
        print(f"  Deleted old data for {municipality}")

        muni_data_sources = source_urls or [primary_source_url]
        if geocode_failed:
            muni_data_sources = ["[geocode_fallback]"] + muni_data_sources
        muni = MunicipalityDB(
            name=municipality,
            province=province,
            latitude=lat,
            longitude=lng,
            data_sources=muni_data_sources,
            last_updated=datetime.now(timezone.utc),
        )
        session.add(muni)

        for reg in regulations:
            db_reg = ZoningRegulationDB(
                municipality=reg.municipality,
                province=reg.province,
                zone_code=reg.zone_code,
                zone_name=reg.zone_name,
                zone_category=reg.zone_category,
                min_lot_size_sqm=reg.min_lot_size_sqm,
                max_building_height_m=reg.max_building_height_m,
                max_stories=reg.max_stories,
                max_lot_coverage_pct=reg.max_lot_coverage_pct,
                min_front_setback_m=reg.min_front_setback_m,
                min_rear_setback_m=reg.min_rear_setback_m,
                min_side_setback_m=reg.min_side_setback_m,
                parking_spaces_per_unit=reg.parking_spaces_per_unit,
                permitted_dwelling_types=reg.permitted_dwelling_types,
                max_units_per_lot=reg.max_units_per_lot,
                density_units_per_hectare=reg.density_units_per_hectare,
                min_lot_frontage_m=reg.min_lot_frontage_m,
                max_floor_area_ratio=reg.max_floor_area_ratio,
                min_landscaped_area_pct=reg.min_landscaped_area_pct,
                bylaw_number=reg.bylaw_number,
                bylaw_effective_date=reg.bylaw_effective_date,
                permitted_commercial_uses=reg.permitted_commercial_uses,
                prohibited_uses=reg.prohibited_uses,
                home_occupation_permitted=reg.home_occupation_permitted,
                secondary_suite_permitted=reg.secondary_suite_permitted,
                short_term_rental_permitted=reg.short_term_rental_permitted,
                inclusionary_zoning_pct=reg.inclusionary_zoning_pct,
                overlay_district=reg.overlay_district,
                min_unit_size_sqm=reg.min_unit_size_sqm,
                additional_regulations=reg.additional_regulations,
                source_url=reg.source_url,
                source_document=reg.source_document,
                extraction_date=reg.extraction_date,
                raw_text_excerpt=reg.raw_text_excerpt,
                confidence_score=reg.confidence_score,
                needs_review=reg.needs_review,
                restrictiveness_score=reg.restrictiveness_score,
            )
            session.add(db_reg)

        for pol in policies:
            db_pol = OfficialPlanPolicyDB(
                municipality=pol.municipality,
                province=pol.province,
                policy_area=pol.policy_area,
                land_use_designation=pol.land_use_designation,
                growth_targets=pol.growth_targets,
                density_targets=pol.density_targets,
                transit_policy=pol.transit_policy,
                affordable_housing_policy=pol.affordable_housing_policy,
                heritage_conservation=pol.heritage_conservation,
                permitted_uses_summary=pol.permitted_uses_summary,
                source_url=pol.source_url,
                source_document=pol.source_document,
                extraction_date=pol.extraction_date,
                raw_text_excerpt=pol.raw_text_excerpt,
                confidence_score=pol.confidence_score,
                needs_review=pol.needs_review,
            )
            session.add(db_pol)

        log = ScrapeLogDB(
            municipality=municipality,
            url=primary_source_url,
            status="success",
            documents_found=len(regulations) + len(policies),
            scraped_at=datetime.now(timezone.utc),
        )
        session.add(log)

        await session.commit()

        # Step 6: Chunk + embed for RAG (text + PDF pages + images)
        _notify(progress_callback, "embedding", {"municipality": municipality})
        print("[6/6] Chunking and embedding for RAG...")

        text_chunks = chunk_text(
            text=collected_text,
            municipality=municipality,
            source_url=primary_source_url,
            source_document=f"{municipality} Zoning Bylaw",
        )
        structured_chunks = structured_to_chunks(
            regulations=regulations,
            policies=policies,
            source_url=primary_source_url,
            source_document=f"{municipality} Zoning Bylaw",
        )
        all_chunks = text_chunks + structured_chunks

        # Collect cached PDF bytes for multimodal page embedding
        pdf_cache = get_pdf_cache()
        pdf_bytes_list: list[tuple[bytes, str, str]] = []
        for pdf_url, pdf_bytes in pdf_cache.items():
            page_chunks, _ = pdf_to_page_chunks(
                pdf_bytes, municipality, pdf_url, f"{municipality} Zoning Bylaw"
            )
            all_chunks.extend(page_chunks)
            pdf_bytes_list.append((pdf_bytes, pdf_url, f"{municipality} Zoning Bylaw"))
        if pdf_bytes_list:
            print(f"  Found {len(pdf_bytes_list)} cached PDFs for multimodal embedding")

        # Collect cached images for multimodal embedding
        image_cache = get_image_cache()
        image_data_list: list[tuple[bytes, str]] = []
        for img_url, (img_bytes, mime_type) in image_cache.items():
            all_chunks.append(DocumentChunkCreate(
                municipality=municipality,
                source_url=img_url,
                source_document=f"{municipality} Zoning Map",
                chunk_text=f"Zoning map/diagram from {img_url}",
                chunk_index=len(all_chunks),
                chunk_type="image",
            ))
            image_data_list.append((img_bytes, mime_type))
        if image_data_list:
            print(f"  Found {len(image_data_list)} cached images for multimodal embedding")

        try:
            stored_count = await chunk_embed_and_store(
                all_chunks, session,
                pdf_bytes_list=pdf_bytes_list if pdf_bytes_list else None,
                image_data_list=image_data_list if image_data_list else None,
            )
            await session.commit()
            print(f"  Embedded and stored {stored_count} chunks for RAG")
        except Exception as e:
            print(f"  [WARN] Embedding failed (non-fatal): {e}")
        finally:
            clear_pdf_cache()
            clear_image_cache()

    await engine.dispose()
    _notify(progress_callback, "completed", {"regulations": len(regulations), "policies": len(policies)})
    print(f"[DONE] Stored {len(regulations)} regulations and {len(policies)} policies for {municipality}")

    return {
        "municipality": municipality,
        "regulations": len(regulations),
        "policies": len(policies),
        "status": "success",
    }
