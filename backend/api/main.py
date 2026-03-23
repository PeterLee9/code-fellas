"""FastAPI application for ZoneMap Canada."""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import chat, compare, export, municipalities, official_plans, pipeline, review, zones

app = FastAPI(
    title="ZoneMap Canada",
    description="National Zoning & Land Use Data Platform API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(municipalities.router, prefix="/api")
app.include_router(zones.router, prefix="/api")
app.include_router(official_plans.router, prefix="/api")
app.include_router(compare.router, prefix="/api")
app.include_router(export.router, prefix="/api")
app.include_router(review.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(pipeline.router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "ZoneMap Canada",
        "description": "National Zoning & Land Use Data Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "license": "CC BY 4.0",
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}
