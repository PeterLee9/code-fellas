from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.models.database import MunicipalityDB
from backend.models.schemas import MunicipalityRead

router = APIRouter(prefix="/municipalities", tags=["municipalities"])


@router.get("", response_model=list[MunicipalityRead])
async def list_municipalities(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(MunicipalityDB).order_by(MunicipalityDB.name))
    rows = result.scalars().all()
    return [MunicipalityRead.model_validate(r) for r in rows]


@router.get("/{name}", response_model=MunicipalityRead)
async def get_municipality(name: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(MunicipalityDB).where(MunicipalityDB.name.ilike(name))
    )
    row = result.scalar_one_or_none()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(404, f"Municipality '{name}' not found")
    return MunicipalityRead.model_validate(row)
