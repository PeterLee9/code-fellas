"""LLM-based structured extraction for zoning data."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import get_settings
from backend.models.schemas import (
    MultiZoneExtractionResult,
    OfficialPlanExtractionResult,
    OfficialPlanPolicyCreate,
    ZoningExtractionResult,
    ZoningRegulationCreate,
)
from backend.services.scoring import compute_restrictiveness_score
from backend.services.validator import validate_zoning_regulation

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert in Canadian municipal zoning bylaws and land use regulations.
Your task is to extract structured zoning regulation data from raw text.

Guidelines:
- Extract ALL zones/zone types mentioned in the text
- Convert all measurements to metric (metres, square metres, hectares)
- If a value is not explicitly stated, set it to null rather than guessing
- For permitted_dwelling_types, use standard terms: "single detached", "semi-detached", "duplex", "triplex", "townhouse", "row house", "apartment", "secondary suite", "garden suite"
- For permitted_commercial_uses, use standard terms: "retail store", "restaurant", "office", "medical clinic", "daycare", "personal service shop"
- For prohibited_uses, list uses that are explicitly banned in the zone text
- Set confidence_score based on how clearly the data was stated in the source text (1.0 = directly stated, 0.5 = inferred/ambiguous)
- Zone codes should match the municipality's official designation (e.g. R1, R2, C1, RM1)
- Record the bylaw_number if mentioned anywhere (e.g. "By-law 569-2013")
- Record bylaw_effective_date in YYYY-MM-DD format if a date is mentioned
- For relevant_excerpt, copy the 50-300 character section of source text most relevant to each zone
- For boolean fields (home_occupation_permitted, secondary_suite_permitted, short_term_rental_permitted):
  set to true/false only if the text explicitly addresses them, otherwise null
- For inclusionary_zoning_pct: only set if an affordable housing percentage is explicitly required
- For overlay_district: set if the zone falls within a special overlay (heritage, transit-oriented, waterfront, etc.)
"""


def get_extraction_llm():
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.extraction_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
    )


def _parse_date(date_str: str | None) -> date | None:
    """Parse a YYYY-MM-DD string into a date object, returning None on failure."""
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str.strip())
    except (ValueError, AttributeError):
        return None


async def extract_zoning_from_text(
    raw_text: str,
    municipality: str,
    province: str,
    source_url: str,
    source_document: str,
) -> list[ZoningRegulationCreate]:
    """Use Gemini to extract structured zoning data from raw text.

    For long documents, splits into overlapping chunks and extracts from
    each chunk, then merges/deduplicates by zone_code (keeping the
    higher-confidence version).
    """
    llm = get_extraction_llm()
    structured_llm = llm.with_structured_output(MultiZoneExtractionResult)

    chunk_size = 15000
    overlap = 2000
    all_zones: dict[str, tuple[ZoningExtractionResult, str | None]] = {}
    doc_bylaw_number: str | None = None

    text_chunks = _split_text(raw_text, chunk_size, overlap)

    for i, chunk in enumerate(text_chunks):
        prompt = f"""{EXTRACTION_SYSTEM_PROMPT}

Municipality: {municipality}
Province: {province}
Source: {source_document}
Text chunk {i + 1} of {len(text_chunks)}:

{chunk}
"""
        try:
            result: MultiZoneExtractionResult = await structured_llm.ainvoke(prompt)
        except Exception as e:
            print(f"  [WARN] Extraction failed on chunk {i + 1}: {e}")
            continue

        if result.bylaw_number and not doc_bylaw_number:
            doc_bylaw_number = result.bylaw_number

        for zone in result.zones:
            existing = all_zones.get(zone.zone_code)
            if existing is None or zone.confidence_score > existing[0].confidence_score:
                excerpt = zone.relevant_excerpt or chunk[:300]
                all_zones[zone.zone_code] = (zone, excerpt)

    regulations: list[ZoningRegulationCreate] = []
    for zone_code, (zone, excerpt) in all_zones.items():
        bylaw_num = zone.bylaw_number or doc_bylaw_number
        reg = ZoningRegulationCreate(
            municipality=municipality,
            province=province,
            zone_code=zone.zone_code,
            zone_name=f"{zone.zone_name} ({zone.zone_code})",
            zone_category=zone.zone_category.value,
            min_lot_size_sqm=zone.min_lot_size_sqm,
            max_building_height_m=zone.max_building_height_m,
            max_stories=zone.max_stories,
            max_lot_coverage_pct=zone.max_lot_coverage_pct,
            min_front_setback_m=zone.min_front_setback_m,
            min_rear_setback_m=zone.min_rear_setback_m,
            min_side_setback_m=zone.min_side_setback_m,
            parking_spaces_per_unit=zone.parking_spaces_per_unit,
            permitted_dwelling_types=zone.permitted_dwelling_types,
            max_units_per_lot=zone.max_units_per_lot,
            density_units_per_hectare=zone.density_units_per_hectare,
            min_lot_frontage_m=zone.min_lot_frontage_m,
            max_floor_area_ratio=zone.max_floor_area_ratio,
            min_landscaped_area_pct=zone.min_landscaped_area_pct,
            bylaw_number=bylaw_num,
            bylaw_effective_date=_parse_date(zone.bylaw_effective_date),
            permitted_commercial_uses=zone.permitted_commercial_uses,
            prohibited_uses=zone.prohibited_uses,
            home_occupation_permitted=zone.home_occupation_permitted,
            secondary_suite_permitted=zone.secondary_suite_permitted,
            short_term_rental_permitted=zone.short_term_rental_permitted,
            inclusionary_zoning_pct=zone.inclusionary_zoning_pct,
            overlay_district=zone.overlay_district,
            min_unit_size_sqm=zone.min_unit_size_sqm,
            additional_regulations=zone.additional_regulations,
            source_url=source_url,
            source_document=source_document,
            raw_text_excerpt=excerpt or raw_text[:500],
            confidence_score=zone.confidence_score,
            needs_review=False,
            extraction_date=datetime.utcnow(),
        )
        reg = validate_zoning_regulation(reg)
        reg.restrictiveness_score = compute_restrictiveness_score(reg)
        regulations.append(reg)

    return regulations


async def extract_official_plan_from_text(
    raw_text: str,
    municipality: str,
    province: str,
    source_url: str,
    source_document: str,
) -> list[OfficialPlanPolicyCreate]:
    """Use Gemini to extract official plan policies from raw text.

    Now extracts multiple policies via a list-based extraction schema.
    """
    llm = get_extraction_llm()
    structured_llm = llm.with_structured_output(OfficialPlanExtractionResult)

    prompt = f"""{EXTRACTION_SYSTEM_PROMPT}

Municipality: {municipality}
Province: {province}
Source: {source_document}

Extract ALL official plan policy areas from the following text. Each distinct
policy area, land-use designation, or planning district should be a separate
entry in the policies list.

{raw_text[:40000]}
"""
    try:
        result: OfficialPlanExtractionResult = await structured_llm.ainvoke(prompt)
    except Exception:
        return []

    policies: list[OfficialPlanPolicyCreate] = []
    for pol in result.policies:
        policy = OfficialPlanPolicyCreate(
            municipality=municipality,
            province=province,
            policy_area=pol.policy_area,
            land_use_designation=pol.land_use_designation,
            growth_targets=pol.growth_targets,
            density_targets=pol.density_targets,
            transit_policy=pol.transit_policy,
            affordable_housing_policy=pol.affordable_housing_policy,
            heritage_conservation=pol.heritage_conservation,
            permitted_uses_summary=pol.permitted_uses_summary,
            source_url=source_url,
            source_document=source_document,
            raw_text_excerpt=raw_text[:500],
            confidence_score=pol.confidence_score,
            needs_review=pol.confidence_score < get_settings().confidence_threshold,
            extraction_date=datetime.utcnow(),
        )
        policies.append(policy)

    return policies


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks. Returns at least one chunk."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks
