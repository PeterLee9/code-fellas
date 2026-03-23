"""
Seed script: Download Toronto open data zoning CSV and load into Supabase.
Run with: PYTHONPATH=. python backend/seed_toronto.py
"""
import asyncio
import csv
import io
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

csv.field_size_limit(sys.maxsize)

import httpx
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings
from backend.models.database import (
    MunicipalityDB,
    ScrapeLogDB,
    ZoningRegulationDB,
    get_async_engine,
    get_async_session_factory,
)
from backend.services.scoring import compute_restrictiveness_score
from backend.models.schemas import ZoningRegulationCreate

ZONING_AREA_URL = "https://ckan0.cf.opendata.inter.prod-toronto.ca/nb_NO/datastore/dump/76a2620f-a6b4-495d-8e41-c0ede1f8a928"
HEIGHT_OVERLAY_URL = "https://ckan0.cf.opendata.inter.prod-toronto.ca/nb_NO/datastore/dump/f0a88d06-2430-4025-b15d-362cabd00f31"
LOT_COVERAGE_URL = "https://ckan0.cf.opendata.inter.prod-toronto.ca/nb_NO/datastore/dump/58ad8814-ca4e-43d6-848d-d5fd8d873574"
PARKING_URL = "https://ckan0.cf.opendata.inter.prod-toronto.ca/nb_NO/datastore/dump/8f969df7-9008-49fd-a50b-df53f1f680e6"

SOURCE_URL = "https://open.toronto.ca/dataset/zoning-by-law/"
SOURCE_DOC = "Toronto Zoning By-law 569-2013"

# Toronto zone code prefix -> category mapping
ZONE_PREFIX_MAP = {
    "R": ("residential", "Residential"),
    "RD": ("residential", "Residential Detached"),
    "RS": ("residential", "Residential Semi-Detached"),
    "RT": ("residential", "Residential Townhouse"),
    "RM": ("residential", "Residential Multiple Dwelling"),
    "RA": ("residential", "Residential Apartment"),
    "CR": ("mixed_use", "Commercial Residential Mixed Use"),
    "CL": ("commercial", "Commercial Local"),
    "CG": ("commercial", "Commercial General"),
    "C": ("commercial", "Commercial"),
    "E": ("industrial", "Employment"),
    "EL": ("industrial", "Employment Light Industrial"),
    "EH": ("industrial", "Employment Heavy Industrial"),
    "EO": ("industrial", "Employment Office"),
    "I": ("institutional", "Institutional"),
    "IH": ("institutional", "Institutional Hospital"),
    "IS": ("institutional", "Institutional School"),
    "O": ("open_space", "Open Space"),
    "ON": ("open_space", "Open Space Natural"),
    "OR": ("open_space", "Open Space Recreation"),
    "OG": ("open_space", "Open Space Golf Course"),
    "OC": ("open_space", "Open Space Cemetery"),
    "U": ("other", "Utility"),
    "UT": ("other", "Utility and Transportation"),
    "A": ("agricultural", "Agricultural"),
}

# Toronto zone codes -> typical permitted dwelling types
DWELLING_TYPES_MAP = {
    "residential": ["single detached"],
    "Residential Detached": ["single detached"],
    "Residential Semi-Detached": ["single detached", "semi-detached"],
    "Residential Townhouse": ["townhouse", "row house"],
    "Residential Multiple Dwelling": ["semi-detached", "townhouse", "duplex", "triplex"],
    "Residential Apartment": ["apartment", "townhouse", "semi-detached", "single detached"],
}


def classify_zone(zone_code: str) -> tuple[str, str]:
    """Return (category, full_name) for a Toronto zone code."""
    code = zone_code.strip().upper()
    # Try longest prefix first
    for prefix_len in range(min(3, len(code)), 0, -1):
        prefix = code[:prefix_len]
        if prefix in ZONE_PREFIX_MAP:
            return ZONE_PREFIX_MAP[prefix]
    return ("other", "Other")


def get_dwelling_types(zone_name: str) -> list[str]:
    """Return typical permitted dwelling types for a zone."""
    for key, types in DWELLING_TYPES_MAP.items():
        if key.lower() in zone_name.lower():
            return types
    return []


def safe_float(val: str, default=None) -> float | None:
    try:
        v = float(val)
        return v if v >= 0 else default
    except (ValueError, TypeError):
        return default


def safe_int(val: str, default=None) -> int | None:
    try:
        v = int(float(val))
        return v if v >= 0 else default
    except (ValueError, TypeError):
        return default


async def download_csv(url: str) -> list[dict]:
    """Download CSV data from Toronto open data."""
    print(f"  Downloading {url[:80]}...")
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    return list(reader)


async def seed():
    settings = get_settings()
    engine = get_async_engine(settings.database_url)
    Session = get_async_session_factory(engine)

    print("[1/5] Downloading Toronto zoning data...")
    rows = await download_csv(ZONING_AREA_URL)
    print(f"  Got {len(rows)} zone polygons")

    # Group by zone code to get unique zones with sample geometries
    print("[2/5] Processing zone types...")
    zone_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        zone_code = row.get("ZN_ZONE", "").strip()
        if zone_code:
            zone_groups[zone_code].append(row)

    print(f"  Found {len(zone_groups)} unique zone codes")

    # Build regulation records
    regulations = []
    for zone_code, zone_rows in zone_groups.items():
        category, zone_name = classify_zone(zone_code)
        full_zone_name = f"{zone_name} ({zone_code})"
        dwelling_types = get_dwelling_types(zone_name)

        # Aggregate numeric values across all polygons of this zone type
        coverages = [safe_float(r.get("COVERAGE", "")) for r in zone_rows]
        coverages = [c for c in coverages if c is not None and c > 0]
        densities = [safe_float(r.get("DENSITY", "")) for r in zone_rows]
        densities = [d for d in densities if d is not None and d > 0]
        frontages = [safe_float(r.get("FRONTAGE", "")) for r in zone_rows]
        frontages = [f for f in frontages if f is not None and f > 0]
        fsis = [safe_float(r.get("FSI_TOTAL", "")) for r in zone_rows]
        fsis = [f for f in fsis if f is not None and f > 0]

        # Use median values for representative regulation
        avg_coverage = sum(coverages) / len(coverages) if coverages else None
        avg_density = sum(densities) / len(densities) if densities else None
        avg_frontage = sum(frontages) / len(frontages) if frontages else None
        avg_fsi = sum(fsis) / len(fsis) if fsis else None

        # Estimate max height from FSI (rough heuristic: FSI * 3m per floor)
        max_height = None
        max_stories = None
        if avg_fsi is not None and avg_coverage is not None and avg_coverage > 0:
            estimated_floors = avg_fsi / (avg_coverage / 100.0) if avg_coverage > 0 else avg_fsi
            max_stories = max(1, round(estimated_floors))
            max_height = max_stories * 3.5  # ~3.5m per storey

        raw_excerpt = f"Zone: {zone_code}, Chapter: {zone_rows[0].get('ZBL_CHAPT', '')}, Section: {zone_rows[0].get('ZBL_SECTN', '')}, Coverage: {avg_coverage}, Density: {avg_density}, FSI: {avg_fsi}, Polygons: {len(zone_rows)}"

        reg = ZoningRegulationCreate(
            municipality="Toronto",
            province="Ontario",
            zone_code=zone_code,
            zone_name=full_zone_name,
            zone_category=category,
            min_lot_size_sqm=avg_frontage * 30 if avg_frontage else None,  # rough: frontage * 30m depth
            max_building_height_m=max_height,
            max_stories=max_stories,
            max_lot_coverage_pct=avg_coverage,
            min_front_setback_m=None,
            min_rear_setback_m=None,
            min_side_setback_m=None,
            parking_spaces_per_unit=None,
            permitted_dwelling_types=dwelling_types,
            max_units_per_lot=None,
            density_units_per_hectare=avg_density,
            additional_regulations={
                "fsi_total": avg_fsi,
                "polygon_count": len(zone_rows),
                "bylaw_chapter": zone_rows[0].get("ZBL_CHAPT", ""),
                "bylaw_section": zone_rows[0].get("ZBL_SECTN", ""),
            },
            source_url=SOURCE_URL,
            source_document=SOURCE_DOC,
            raw_text_excerpt=raw_excerpt,
            confidence_score=0.85,
            needs_review=False,
        )
        reg.restrictiveness_score = compute_restrictiveness_score(reg)
        regulations.append(reg)

    print(f"  Built {len(regulations)} zoning regulation records")

    # Pick a sample geometry for each zone (first polygon)
    sample_geometries = {}
    for zone_code, zone_rows in zone_groups.items():
        geom_str = zone_rows[0].get("geometry", "")
        if geom_str:
            try:
                sample_geometries[zone_code] = json.loads(geom_str)
            except json.JSONDecodeError:
                pass

    print("[3/5] Inserting municipality record...")
    async with Session() as session:
        # Insert Toronto municipality
        toronto = MunicipalityDB(
            name="Toronto",
            province="Ontario",
            population=2794356,
            latitude=43.6532,
            longitude=-79.3832,
            geometry=None,
            data_sources=[SOURCE_URL],
            last_updated=datetime.utcnow(),
        )
        session.add(toronto)

        print("[4/5] Inserting zoning regulations...")
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

        # Insert scrape log
        log = ScrapeLogDB(
            municipality="Toronto",
            url=SOURCE_URL,
            status="success",
            documents_found=len(regulations),
            scraped_at=datetime.utcnow(),
        )
        session.add(log)

        print("[5/5] Committing to database...")
        await session.commit()

    await engine.dispose()
    print(f"[DONE] Seeded {len(regulations)} Toronto zoning regulations")

    # Print summary
    categories = defaultdict(int)
    for reg in regulations:
        categories[reg.zone_category] += 1
    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} zones")


if __name__ == "__main__":
    asyncio.run(seed())
