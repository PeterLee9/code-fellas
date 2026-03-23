from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.models.database import ZoningRegulationDB
from backend.models.schemas import ZoningRegulationRead

router = APIRouter(prefix="/review", tags=["review"])


@router.get("", response_model=list[ZoningRegulationRead])
async def review_queue(
    municipality: str | None = None,
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Get items flagged for human review (needs_review=True)."""
    stmt = select(ZoningRegulationDB).where(ZoningRegulationDB.needs_review == True)

    if municipality:
        stmt = stmt.where(ZoningRegulationDB.municipality.ilike(f"%{municipality}%"))

    stmt = stmt.order_by(ZoningRegulationDB.confidence_score.asc()).limit(limit)

    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [ZoningRegulationRead.model_validate(r) for r in rows]


@router.get("/summary")
async def review_summary(db: AsyncSession = Depends(get_db)):
    """Summary of items needing review by municipality."""
    result = await db.execute(
        select(
            ZoningRegulationDB.municipality,
            func.count(ZoningRegulationDB.id).label("total"),
            func.count().filter(ZoningRegulationDB.needs_review == True).label("needs_review"),
            func.avg(ZoningRegulationDB.confidence_score).label("avg_confidence"),
        ).group_by(ZoningRegulationDB.municipality)
    )

    summary = []
    for row in result.all():
        summary.append({
            "municipality": row[0],
            "total_zones": row[1],
            "needs_review": row[2],
            "avg_confidence": round(row[3], 2) if row[3] else None,
        })
    return summary
