from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.models.database import ZoningRegulationDB

router = APIRouter(prefix="/compare", tags=["compare"])


@router.get("")
async def compare_municipalities(
    municipalities: str = Query(..., description="Comma-separated municipality names"),
    metric: str = Query("restrictiveness_score", description="Metric to compare"),
    db: AsyncSession = Depends(get_db),
):
    """Compare a specific metric across municipalities."""
    muni_list = [m.strip() for m in municipalities.split(",")]

    valid_metrics = {
        "restrictiveness_score", "min_lot_size_sqm", "max_building_height_m",
        "max_stories", "max_lot_coverage_pct", "parking_spaces_per_unit",
        "density_units_per_hectare",
    }
    if metric not in valid_metrics:
        from fastapi import HTTPException
        raise HTTPException(400, f"Invalid metric. Choose from: {valid_metrics}")

    col = getattr(ZoningRegulationDB, metric)

    result = await db.execute(
        select(
            ZoningRegulationDB.municipality,
            ZoningRegulationDB.zone_category,
            func.avg(col).label("avg_value"),
            func.min(col).label("min_value"),
            func.max(col).label("max_value"),
            func.count(ZoningRegulationDB.id).label("zone_count"),
        )
        .where(ZoningRegulationDB.municipality.in_(muni_list))
        .where(col.isnot(None))
        .group_by(ZoningRegulationDB.municipality, ZoningRegulationDB.zone_category)
        .order_by(ZoningRegulationDB.municipality)
    )

    comparisons = []
    for row in result.all():
        comparisons.append({
            "municipality": row[0],
            "zone_category": row[1],
            "metric": metric,
            "avg_value": round(row[2], 2) if row[2] else None,
            "min_value": round(row[3], 2) if row[3] else None,
            "max_value": round(row[4], 2) if row[4] else None,
            "zone_count": row[5],
        })
    return comparisons


@router.get("/rankings")
async def rankings(
    metric: str = Query("restrictiveness_score", description="Metric to rank by"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    category: str | None = Query(None, description="Filter by zone category"),
    db: AsyncSession = Depends(get_db),
):
    """Rank municipalities by a metric."""
    valid_metrics = {
        "restrictiveness_score", "min_lot_size_sqm", "max_building_height_m",
        "parking_spaces_per_unit", "density_units_per_hectare",
    }
    if metric not in valid_metrics:
        from fastapi import HTTPException
        raise HTTPException(400, f"Invalid metric. Choose from: {valid_metrics}")

    col = getattr(ZoningRegulationDB, metric)

    stmt = select(
        ZoningRegulationDB.municipality,
        func.avg(col).label("avg_value"),
        func.count(ZoningRegulationDB.id).label("zone_count"),
    ).where(col.isnot(None))

    if category:
        stmt = stmt.where(ZoningRegulationDB.zone_category == category)

    stmt = stmt.group_by(ZoningRegulationDB.municipality)

    if order == "desc":
        stmt = stmt.order_by(func.avg(col).desc())
    else:
        stmt = stmt.order_by(func.avg(col).asc())

    result = await db.execute(stmt)
    rankings = []
    for i, row in enumerate(result.all(), 1):
        rankings.append({
            "rank": i,
            "municipality": row[0],
            "avg_value": round(row[1], 2) if row[1] else None,
            "zone_count": row[2],
        })
    return rankings
