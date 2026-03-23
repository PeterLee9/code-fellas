from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ZoneCategory(str, Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    MIXED_USE = "mixed_use"
    INSTITUTIONAL = "institutional"
    OPEN_SPACE = "open_space"
    AGRICULTURAL = "agricultural"
    OTHER = "other"


class ZoningRegulationBase(BaseModel):
    municipality: str
    province: str
    zone_code: str = Field(description="Zone code, e.g. R1, R2, C1")
    zone_name: str = Field(description="Human-readable zone name, e.g. Residential Single Detached")
    zone_category: ZoneCategory
    min_lot_size_sqm: float | None = Field(None, description="Minimum lot size in square metres")
    max_building_height_m: float | None = Field(None, description="Maximum building height in metres")
    max_stories: int | None = Field(None, description="Maximum number of stories")
    max_lot_coverage_pct: float | None = Field(None, description="Maximum lot coverage percentage (0-100)")
    min_front_setback_m: float | None = Field(None, description="Minimum front yard setback in metres")
    min_rear_setback_m: float | None = Field(None, description="Minimum rear yard setback in metres")
    min_side_setback_m: float | None = Field(None, description="Minimum side yard setback in metres")
    parking_spaces_per_unit: float | None = Field(None, description="Required parking spaces per dwelling unit")
    permitted_dwelling_types: list[str] = Field(default_factory=list, description="List of permitted dwelling types")
    max_units_per_lot: int | None = Field(None, description="Maximum dwelling units per lot")
    density_units_per_hectare: float | None = Field(None, description="Maximum density in units per hectare")
    min_lot_frontage_m: float | None = Field(None, description="Minimum lot frontage (width) in metres")
    max_floor_area_ratio: float | None = Field(None, description="Maximum Floor Area Ratio (FAR)")
    min_landscaped_area_pct: float | None = Field(None, description="Minimum landscaped/green area percentage (0-100)")
    bylaw_number: str | None = Field(None, description="Official bylaw reference, e.g. By-law 569-2013")
    bylaw_effective_date: date | None = Field(None, description="Date bylaw was enacted or last amended")
    permitted_commercial_uses: list[str] = Field(default_factory=list, description="Permitted commercial/retail uses")
    prohibited_uses: list[str] = Field(default_factory=list, description="Explicitly prohibited uses")
    home_occupation_permitted: bool | None = Field(None, description="Whether home-based businesses are allowed")
    secondary_suite_permitted: bool | None = Field(None, description="Whether secondary suites / ADUs are allowed")
    short_term_rental_permitted: bool | None = Field(None, description="Whether short-term rentals (Airbnb) are allowed")
    inclusionary_zoning_pct: float | None = Field(None, description="Required affordable housing percentage (0-100)")
    overlay_district: str | None = Field(None, description="Overlay zone, e.g. heritage, transit-oriented, waterfront")
    min_unit_size_sqm: float | None = Field(None, description="Minimum dwelling unit size in square metres")
    additional_regulations: dict[str, Any] | None = Field(None, description="Jurisdiction-specific rules not captured by standard fields")
    source_url: str = Field(description="URL of the source document")
    source_document: str = Field(description="Name/title of the source document")
    raw_text_excerpt: str = Field(description="Excerpt of raw text used for extraction, for auditability")
    confidence_score: float = Field(ge=0.0, le=1.0, description="LLM self-assessed extraction confidence (0-1)")
    needs_review: bool = Field(False, description="Flagged for human review")


class ZoningRegulationCreate(ZoningRegulationBase):
    extraction_date: datetime = Field(default_factory=datetime.utcnow)
    restrictiveness_score: float | None = None


class ZoningRegulationRead(ZoningRegulationBase):
    id: int
    extraction_date: datetime
    restrictiveness_score: float | None = None

    model_config = {"from_attributes": True}


class OfficialPlanPolicyBase(BaseModel):
    municipality: str
    province: str
    policy_area: str = Field(description="E.g. Urban Growth Centre, Employment Area")
    land_use_designation: str = Field(description="E.g. Mixed Use, Neighbourhood")
    growth_targets: str | None = Field(None, description="E.g. minimum 200 residents and jobs per hectare")
    density_targets: str | None = Field(None, description="Specific density targets, e.g. 150 units/ha")
    transit_policy: str | None = Field(None, description="Transit-oriented development policies")
    affordable_housing_policy: str | None = Field(None, description="Affordable housing policies and requirements")
    heritage_conservation: str | None = Field(None, description="Heritage and conservation area policies")
    permitted_uses_summary: str
    source_url: str
    source_document: str
    raw_text_excerpt: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False


class OfficialPlanPolicyCreate(OfficialPlanPolicyBase):
    extraction_date: datetime = Field(default_factory=datetime.utcnow)


class OfficialPlanPolicyRead(OfficialPlanPolicyBase):
    id: int
    extraction_date: datetime

    model_config = {"from_attributes": True}


class MunicipalityBase(BaseModel):
    name: str
    province: str
    population: int | None = None
    latitude: float
    longitude: float
    geometry: dict[str, Any] | None = Field(None, description="GeoJSON geometry for boundary")
    data_sources: list[str] = Field(default_factory=list)


class MunicipalityCreate(MunicipalityBase):
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class MunicipalityRead(MunicipalityBase):
    id: int
    last_updated: datetime

    model_config = {"from_attributes": True}


class DocumentChunkBase(BaseModel):
    municipality: str
    source_url: str
    source_document: str
    chunk_text: str
    chunk_index: int
    chunk_type: str = "text"


class DocumentChunkCreate(DocumentChunkBase):
    embedding: list[float] | None = None
    pdf_bytes: bytes | None = None


class DocumentChunkRead(DocumentChunkBase):
    id: int

    model_config = {"from_attributes": True}


class ScrapeLogCreate(BaseModel):
    municipality: str
    url: str
    status: str = "success"
    documents_found: int = 0
    error_message: str | None = None


class ScrapeLogRead(ScrapeLogCreate):
    id: int
    scraped_at: datetime

    model_config = {"from_attributes": True}


class ChatHistoryMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    municipality: str | None = None
    history: list[ChatHistoryMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)


# Schema used for LLM structured extraction (simpler, no DB fields)
class ZoningExtractionResult(BaseModel):
    """Schema for LLM to extract zoning data from raw text."""

    zone_code: str = Field(description="The zone code identifier, e.g. R1, R2, C1, M1")
    zone_name: str = Field(description="Full human-readable name of the zone")
    zone_category: ZoneCategory = Field(description="The broad category this zone falls under")
    min_lot_size_sqm: float | None = Field(None, description="Minimum lot/parcel size in square metres. Convert from other units if needed.")
    max_building_height_m: float | None = Field(None, description="Maximum permitted building height in metres. Convert from feet if needed.")
    max_stories: int | None = Field(None, description="Maximum number of stories/floors permitted")
    max_lot_coverage_pct: float | None = Field(None, description="Maximum percentage of lot that can be covered by buildings (0-100)")
    min_front_setback_m: float | None = Field(None, description="Minimum front yard setback distance in metres")
    min_rear_setback_m: float | None = Field(None, description="Minimum rear yard setback distance in metres")
    min_side_setback_m: float | None = Field(None, description="Minimum side yard setback distance in metres")
    parking_spaces_per_unit: float | None = Field(None, description="Number of parking spaces required per dwelling unit")
    permitted_dwelling_types: list[str] = Field(default_factory=list, description="Types of dwellings permitted, e.g. single detached, semi-detached, townhouse, apartment")
    max_units_per_lot: int | None = Field(None, description="Maximum number of dwelling units permitted per lot")
    density_units_per_hectare: float | None = Field(None, description="Maximum density expressed as units per hectare")
    min_lot_frontage_m: float | None = Field(None, description="Minimum lot frontage (width) in metres")
    max_floor_area_ratio: float | None = Field(None, description="Maximum Floor Area Ratio (FAR), e.g. 0.6 means building floor area can be 60% of lot area")
    min_landscaped_area_pct: float | None = Field(None, description="Minimum required landscaped/green area as a percentage of lot (0-100)")
    bylaw_number: str | None = Field(None, description="The official bylaw reference number, e.g. 'By-law 569-2013'")
    bylaw_effective_date: str | None = Field(None, description="Date the bylaw was enacted or last amended, in YYYY-MM-DD format")
    permitted_commercial_uses: list[str] = Field(default_factory=list, description="Permitted commercial/retail uses, e.g. retail store, restaurant, office")
    prohibited_uses: list[str] = Field(default_factory=list, description="Explicitly prohibited uses in this zone")
    home_occupation_permitted: bool | None = Field(None, description="Whether home-based businesses/occupations are permitted")
    secondary_suite_permitted: bool | None = Field(None, description="Whether secondary suites, accessory dwelling units, or basement apartments are permitted")
    short_term_rental_permitted: bool | None = Field(None, description="Whether short-term rentals (e.g. Airbnb) are permitted")
    inclusionary_zoning_pct: float | None = Field(None, description="Required percentage of affordable housing units (0-100), if an inclusionary zoning policy applies")
    overlay_district: str | None = Field(None, description="Name of any overlay district that applies, e.g. heritage conservation, transit-oriented, waterfront")
    min_unit_size_sqm: float | None = Field(None, description="Minimum dwelling unit size in square metres")
    additional_regulations: dict[str, Any] | None = Field(None, description="Any other notable regulations specific to this zone")
    relevant_excerpt: str = Field(default="", description="The specific section of text where this zone's data was found (50-300 chars)")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Your confidence in the accuracy of this extraction (0.0=guessing, 1.0=certain)")


class MultiZoneExtractionResult(BaseModel):
    """Wrapper for extracting multiple zones from a single document section."""

    bylaw_number: str | None = Field(None, description="The overarching bylaw number for this document, e.g. 'By-law 569-2013'")
    zones: list[ZoningExtractionResult] = Field(description="List of all zoning regulations found in the text")


class OfficialPlanPolicyResult(BaseModel):
    """Schema for LLM to extract a single official plan policy from raw text."""

    policy_area: str = Field(description="The policy/planning area name, e.g. Urban Growth Centre, Employment Area")
    land_use_designation: str = Field(description="The land use designation, e.g. Mixed Use, Neighbourhood")
    growth_targets: str | None = Field(None, description="Growth or density targets mentioned")
    density_targets: str | None = Field(None, description="Specific density targets, e.g. 150 units/ha minimum")
    transit_policy: str | None = Field(None, description="Transit-oriented development policies for this area")
    affordable_housing_policy: str | None = Field(None, description="Affordable housing requirements or targets")
    heritage_conservation: str | None = Field(None, description="Heritage or conservation policies for this area")
    permitted_uses_summary: str = Field(description="Summary of permitted uses in this area")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Your confidence in the accuracy of this extraction")


class OfficialPlanExtractionResult(BaseModel):
    """Wrapper for extracting multiple official plan policies from a document."""

    policies: list[OfficialPlanPolicyResult] = Field(description="List of all official plan policies found in the text")
