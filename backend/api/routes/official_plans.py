from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.models.database import OfficialPlanPolicyDB
from backend.models.schemas import OfficialPlanPolicyRead

router = APIRouter(prefix="/official-plans", tags=["official_plans"])


@router.get("", response_model=list[OfficialPlanPolicyRead])
async def list_official_plans(
    municipality: str | None = None,
    needs_review: bool | None = None,
    limit: int = Query(100, le=500),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List official plan policies with optional filters."""
    stmt = select(OfficialPlanPolicyDB)

    if municipality:
        stmt = stmt.where(OfficialPlanPolicyDB.municipality.ilike(f"%{municipality}%"))
    if needs_review is not None:
        stmt = stmt.where(OfficialPlanPolicyDB.needs_review == needs_review)

    stmt = stmt.order_by(OfficialPlanPolicyDB.municipality)
    stmt = stmt.offset(offset).limit(limit)

    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [OfficialPlanPolicyRead.model_validate(r) for r in rows]


@router.get("/{policy_id}", response_model=OfficialPlanPolicyRead)
async def get_official_plan(
    policy_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a single official plan policy by ID."""
    result = await db.execute(
        select(OfficialPlanPolicyDB).where(OfficialPlanPolicyDB.id == policy_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Policy not found")
    return OfficialPlanPolicyRead.model_validate(row)
