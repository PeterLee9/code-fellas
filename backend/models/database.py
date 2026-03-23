from __future__ import annotations

import json
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class MunicipalityDB(Base):
    __tablename__ = "municipalities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    province = Column(String, nullable=False, index=True)
    population = Column(Integer, nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    geometry = Column(JSONB, nullable=True)
    data_sources = Column(JSONB, nullable=False, default=list)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)


class ZoningRegulationDB(Base):
    __tablename__ = "zoning_regulations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    municipality = Column(String, nullable=False, index=True)
    province = Column(String, nullable=False, index=True)
    zone_code = Column(String, nullable=False, index=True)
    zone_name = Column(String, nullable=False)
    zone_category = Column(String, nullable=False, index=True)
    min_lot_size_sqm = Column(Float, nullable=True)
    max_building_height_m = Column(Float, nullable=True)
    max_stories = Column(Integer, nullable=True)
    max_lot_coverage_pct = Column(Float, nullable=True)
    min_front_setback_m = Column(Float, nullable=True)
    min_rear_setback_m = Column(Float, nullable=True)
    min_side_setback_m = Column(Float, nullable=True)
    parking_spaces_per_unit = Column(Float, nullable=True)
    permitted_dwelling_types = Column(JSONB, nullable=False, default=list)
    max_units_per_lot = Column(Integer, nullable=True)
    density_units_per_hectare = Column(Float, nullable=True)
    min_lot_frontage_m = Column(Float, nullable=True)
    max_floor_area_ratio = Column(Float, nullable=True)
    min_landscaped_area_pct = Column(Float, nullable=True)
    bylaw_number = Column(String, nullable=True)
    bylaw_effective_date = Column(Date, nullable=True)
    permitted_commercial_uses = Column(JSONB, nullable=False, default=list)
    prohibited_uses = Column(JSONB, nullable=False, default=list)
    home_occupation_permitted = Column(Boolean, nullable=True)
    secondary_suite_permitted = Column(Boolean, nullable=True)
    short_term_rental_permitted = Column(Boolean, nullable=True)
    inclusionary_zoning_pct = Column(Float, nullable=True)
    overlay_district = Column(String, nullable=True)
    min_unit_size_sqm = Column(Float, nullable=True)
    additional_regulations = Column(JSONB, nullable=True)
    source_url = Column(String, nullable=False)
    source_document = Column(String, nullable=False)
    extraction_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    raw_text_excerpt = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False, default=1.0)
    needs_review = Column(Boolean, nullable=False, default=False)
    restrictiveness_score = Column(Float, nullable=True)


class OfficialPlanPolicyDB(Base):
    __tablename__ = "official_plan_policies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    municipality = Column(String, nullable=False, index=True)
    province = Column(String, nullable=False, index=True)
    policy_area = Column(String, nullable=False)
    land_use_designation = Column(String, nullable=False)
    growth_targets = Column(Text, nullable=True)
    density_targets = Column(Text, nullable=True)
    transit_policy = Column(Text, nullable=True)
    affordable_housing_policy = Column(Text, nullable=True)
    heritage_conservation = Column(Text, nullable=True)
    permitted_uses_summary = Column(Text, nullable=False)
    source_url = Column(String, nullable=False)
    source_document = Column(String, nullable=False)
    extraction_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    raw_text_excerpt = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False, default=1.0)
    needs_review = Column(Boolean, nullable=False, default=False)


class DocumentChunkDB(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    municipality = Column(String, nullable=False, index=True)
    source_url = Column(String, nullable=False)
    source_document = Column(String, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String, nullable=False, default="text", index=True)
    embedding = Column(Vector(3072), nullable=True)


class ScrapeLogDB(Base):
    __tablename__ = "scrape_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    municipality = Column(String, nullable=False, index=True)
    url = Column(String, nullable=False)
    scraped_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    status = Column(String, nullable=False, default="success")
    documents_found = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)


def get_async_engine(database_url: str):
    async_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
    return create_async_engine(async_url, echo=False)


def get_async_session_factory(engine):
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
