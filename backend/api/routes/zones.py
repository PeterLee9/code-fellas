from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.models.database import ZoningRegulationDB, MunicipalityDB
from backend.models.schemas import ZoningRegulationRead

router = APIRouter(prefix="/zones", tags=["zones"])


@router.get("", response_model=list[ZoningRegulationRead])
async def list_zones(
    municipality: str | None = None,
    category: str | None = None,
    needs_review: bool | None = None,
    min_restrictiveness: float | None = None,
    max_restrictiveness: float | None = None,
    limit: int = Query(100, le=500),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ZoningRegulationDB)

    if municipality:
        stmt = stmt.where(ZoningRegulationDB.municipality.ilike(f"%{municipality}%"))
    if category:
        stmt = stmt.where(ZoningRegulationDB.zone_category == category)
    if needs_review is not None:
        stmt = stmt.where(ZoningRegulationDB.needs_review == needs_review)
    if min_restrictiveness is not None:
        stmt = stmt.where(ZoningRegulationDB.restrictiveness_score >= min_restrictiveness)
    if max_restrictiveness is not None:
        stmt = stmt.where(ZoningRegulationDB.restrictiveness_score <= max_restrictiveness)

    stmt = stmt.order_by(ZoningRegulationDB.municipality, ZoningRegulationDB.zone_code)
    stmt = stmt.offset(offset).limit(limit)

    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [ZoningRegulationRead.model_validate(r) for r in rows]


@router.get("/bbox")
async def zones_bbox(
    sw_lat: float = Query(..., description="Southwest corner latitude"),
    sw_lng: float = Query(..., description="Southwest corner longitude"),
    ne_lat: float = Query(..., description="Northeast corner latitude"),
    ne_lng: float = Query(..., description="Northeast corner longitude"),
    db: AsyncSession = Depends(get_db),
):
    """Find zones in municipalities within a geographic bounding box (for map viewport)."""
    stmt = select(MunicipalityDB).where(
        MunicipalityDB.latitude >= sw_lat,
        MunicipalityDB.latitude <= ne_lat,
        MunicipalityDB.longitude >= sw_lng,
        MunicipalityDB.longitude <= ne_lng,
    )
    result = await db.execute(stmt)
    bbox_munis = [r.name for r in result.scalars().all()]

    if not bbox_munis:
        return []

    stmt = select(ZoningRegulationDB).where(
        ZoningRegulationDB.municipality.in_(bbox_munis)
    ).order_by(ZoningRegulationDB.municipality, ZoningRegulationDB.zone_code)

    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [ZoningRegulationRead.model_validate(r) for r in rows]


@router.get("/nearby")
async def zones_nearby(
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    radius_km: float = Query(10, description="Search radius in km"),
    db: AsyncSession = Depends(get_db),
):
    """Find zones in municipalities near a geographic point."""
    # Find municipalities within radius using Haversine formula
    stmt = select(MunicipalityDB).where(
        func.sqrt(
            func.pow(MunicipalityDB.latitude - lat, 2) +
            func.pow((MunicipalityDB.longitude - lng) * func.cos(func.radians(lat)), 2)
        ) * 111.0 <= radius_km  # 1 degree ≈ 111 km
    )
    result = await db.execute(stmt)
    nearby_munis = [r.name for r in result.scalars().all()]

    if not nearby_munis:
        return []

    stmt = select(ZoningRegulationDB).where(
        ZoningRegulationDB.municipality.in_(nearby_munis)
    ).order_by(ZoningRegulationDB.municipality, ZoningRegulationDB.zone_code)

    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [ZoningRegulationRead.model_validate(r) for r in rows]


@router.get("/stats")
async def zone_stats(db: AsyncSession = Depends(get_db)):
    """Get aggregate statistics across all zones."""
    result = await db.execute(
        select(
            ZoningRegulationDB.municipality,
            func.count(ZoningRegulationDB.id).label("zone_count"),
            func.avg(ZoningRegulationDB.restrictiveness_score).label("avg_restrictiveness"),
            func.count().filter(ZoningRegulationDB.needs_review == True).label("needs_review_count"),
        ).group_by(ZoningRegulationDB.municipality)
    )
    stats = []
    for row in result.all():
        stats.append({
            "municipality": row[0],
            "zone_count": row[1],
            "avg_restrictiveness": round(row[2], 1) if row[2] else None,
            "needs_review_count": row[3],
        })
    return stats
