from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.models.database import ZoningRegulationDB

router = APIRouter(prefix="/export", tags=["export"])

CC_BY_4_HEADER = (
    "# ZoneMap Canada - National Zoning & Land Use Data Platform\n"
    "# License: Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
    "# https://creativecommons.org/licenses/by/4.0/\n"
    "# Attribution: ZoneMap Canada / Code Fellas\n"
)

EXPORT_FIELDS = [
    "id", "municipality", "province", "zone_code", "zone_name", "zone_category",
    "min_lot_size_sqm", "max_building_height_m", "max_stories", "max_lot_coverage_pct",
    "min_front_setback_m", "min_rear_setback_m", "min_side_setback_m",
    "parking_spaces_per_unit", "permitted_dwelling_types", "max_units_per_lot",
    "density_units_per_hectare", "restrictiveness_score", "confidence_score",
    "needs_review", "source_url", "source_document", "extraction_date",
]


@router.get("")
async def export_data(
    format: str = Query("csv", description="Export format: csv or json"),
    municipality: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Export all zoning data as CSV or JSON with CC BY 4.0 license."""
    stmt = select(ZoningRegulationDB)
    if municipality:
        stmt = stmt.where(ZoningRegulationDB.municipality.ilike(f"%{municipality}%"))
    stmt = stmt.order_by(ZoningRegulationDB.municipality, ZoningRegulationDB.zone_code)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    if format == "json":
        data = {
            "license": "CC BY 4.0 - https://creativecommons.org/licenses/by/4.0/",
            "attribution": "ZoneMap Canada / Code Fellas",
            "record_count": len(rows),
            "data": [],
        }
        for r in rows:
            record = {}
            for field in EXPORT_FIELDS:
                val = getattr(r, field, None)
                if hasattr(val, "isoformat"):
                    val = val.isoformat()
                record[field] = val
            data["data"].append(record)

        return StreamingResponse(
            io.BytesIO(json.dumps(data, indent=2).encode()),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=zonemap_canada_data.json"},
        )
    else:
        output = io.StringIO()
        output.write(CC_BY_4_HEADER)
        writer = csv.DictWriter(output, fieldnames=EXPORT_FIELDS)
        writer.writeheader()

        for r in rows:
            record = {}
            for field in EXPORT_FIELDS:
                val = getattr(r, field, None)
                if isinstance(val, list):
                    val = "; ".join(str(v) for v in val)
                elif hasattr(val, "isoformat"):
                    val = val.isoformat()
                record[field] = val
            writer.writerow(record)

        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=zonemap_canada_data.csv"},
        )
