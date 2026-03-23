from __future__ import annotations

from backend.config import get_settings
from backend.models.schemas import ZoningRegulationCreate


def validate_zoning_regulation(reg: ZoningRegulationCreate, confidence_threshold: float | None = None) -> ZoningRegulationCreate:
    if confidence_threshold is None:
        confidence_threshold = get_settings().confidence_threshold
    """Apply validation rules and set needs_review flag if any check fails."""
    issues: list[str] = []

    if reg.min_lot_size_sqm is not None and reg.min_lot_size_sqm <= 0:
        issues.append("min_lot_size_sqm must be > 0")
        reg.min_lot_size_sqm = None

    if reg.max_building_height_m is not None:
        if reg.max_building_height_m <= 0 or reg.max_building_height_m > 500:
            issues.append("max_building_height_m must be > 0 and < 500")
            reg.max_building_height_m = None

    if reg.max_lot_coverage_pct is not None:
        if reg.max_lot_coverage_pct < 0 or reg.max_lot_coverage_pct > 100:
            issues.append("max_lot_coverage_pct must be between 0 and 100")
            reg.max_lot_coverage_pct = None

    if reg.parking_spaces_per_unit is not None and reg.parking_spaces_per_unit < 0:
        issues.append("parking_spaces_per_unit must be >= 0")
        reg.parking_spaces_per_unit = None

    for field_name in ("min_front_setback_m", "min_rear_setback_m", "min_side_setback_m"):
        val = getattr(reg, field_name)
        if val is not None and val < 0:
            issues.append(f"{field_name} must be >= 0")
            setattr(reg, field_name, None)

    if reg.max_stories is not None and (reg.max_stories <= 0 or reg.max_stories > 200):
        issues.append("max_stories must be > 0 and <= 200")
        reg.max_stories = None

    if reg.density_units_per_hectare is not None and reg.density_units_per_hectare < 0:
        issues.append("density_units_per_hectare must be >= 0")
        reg.density_units_per_hectare = None

    # --- New field validations ---

    if reg.min_lot_frontage_m is not None and reg.min_lot_frontage_m <= 0:
        issues.append("min_lot_frontage_m must be > 0")
        reg.min_lot_frontage_m = None

    if reg.max_floor_area_ratio is not None:
        if reg.max_floor_area_ratio <= 0 or reg.max_floor_area_ratio > 50:
            issues.append("max_floor_area_ratio must be > 0 and <= 50")
            reg.max_floor_area_ratio = None

    if reg.min_landscaped_area_pct is not None:
        if reg.min_landscaped_area_pct < 0 or reg.min_landscaped_area_pct > 100:
            issues.append("min_landscaped_area_pct must be between 0 and 100")
            reg.min_landscaped_area_pct = None

    if reg.inclusionary_zoning_pct is not None:
        if reg.inclusionary_zoning_pct < 0 or reg.inclusionary_zoning_pct > 100:
            issues.append("inclusionary_zoning_pct must be between 0 and 100")
            reg.inclusionary_zoning_pct = None

    if reg.min_unit_size_sqm is not None:
        if reg.min_unit_size_sqm <= 0 or reg.min_unit_size_sqm > 10000:
            issues.append("min_unit_size_sqm must be > 0 and <= 10000")
            reg.min_unit_size_sqm = None

    if reg.confidence_score < confidence_threshold:
        issues.append(f"confidence_score {reg.confidence_score} below threshold {confidence_threshold}")

    if issues:
        reg.needs_review = True
        extras = reg.additional_regulations or {}
        extras["validation_issues"] = issues
        reg.additional_regulations = extras

    return reg
