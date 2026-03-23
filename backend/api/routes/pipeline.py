"""Pipeline trigger and status endpoints for in-app scraping."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.models.database import ScrapeLogDB

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# In-memory job tracking (resets on server restart)
_jobs: dict[str, dict[str, Any]] = {}


class PipelineRunRequest(BaseModel):
    municipality: str
    province: str = "Ontario"


class PipelineRunResponse(BaseModel):
    job_id: str
    municipality: str
    status: str


def _progress_callback(job_id: str):
    """Create a progress callback for a specific job."""
    def callback(step: str, data: dict[str, Any]):
        if job_id in _jobs:
            _jobs[job_id]["current_step"] = step
            _jobs[job_id]["step_data"] = data
            if step == "completed":
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                _jobs[job_id]["regulations_found"] = data.get("regulations", 0)
                _jobs[job_id]["policies_found"] = data.get("policies", 0)
            elif step == "failed":
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                _jobs[job_id]["error"] = data.get("error", "unknown")
    return callback


async def _run_pipeline_task(job_id: str, municipality: str, province: str):
    """Background task that runs the pipeline and updates job status."""
    from backend.config import get_settings
    from backend.agents.orchestrator import run_pipeline_for_municipality

    try:
        settings = get_settings()
        result = await run_pipeline_for_municipality(
            municipality=municipality,
            province=province,
            database_url=settings.database_url,
            progress_callback=_progress_callback(job_id),
        )
        if _jobs[job_id]["status"] != "completed":
            _jobs[job_id]["status"] = "completed" if result.get("status") == "success" else "failed"
            _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            _jobs[job_id]["regulations_found"] = result.get("regulations", 0)
            _jobs[job_id]["policies_found"] = result.get("policies", 0)
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        _jobs[job_id]["error"] = str(e)


@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(request: PipelineRunRequest):
    """Trigger the agentic pipeline for a municipality. Runs in the background."""
    # Reject duplicate runs
    for job in _jobs.values():
        if (
            job["municipality"].lower() == request.municipality.lower()
            and job["status"] == "running"
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Pipeline already running for {request.municipality}",
            )

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "job_id": job_id,
        "municipality": request.municipality,
        "province": request.province,
        "status": "running",
        "current_step": "queued",
        "step_data": {},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "regulations_found": 0,
        "policies_found": 0,
        "error": None,
    }

    asyncio.create_task(_run_pipeline_task(job_id, request.municipality, request.province))

    return PipelineRunResponse(
        job_id=job_id,
        municipality=request.municipality,
        status="running",
    )


@router.get("/status")
async def get_pipeline_status():
    """Get status of all active/recent pipeline jobs."""
    return list(_jobs.values())


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific pipeline job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@router.get("/logs")
async def get_pipeline_logs(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Get scrape history from the database."""
    stmt = (
        select(ScrapeLogDB)
        .order_by(ScrapeLogDB.scraped_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "id": r.id,
            "municipality": r.municipality,
            "url": r.url,
            "status": r.status,
            "documents_found": r.documents_found,
            "error_message": r.error_message,
            "scraped_at": r.scraped_at.isoformat() if r.scraped_at else None,
        }
        for r in rows
    ]
