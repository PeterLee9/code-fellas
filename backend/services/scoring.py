from __future__ import annotations

from backend.models.schemas import ZoningRegulationCreate


# Reference ranges for normalization (based on typical Canadian zoning values)
_LOT_SIZE_MIN = 150.0     # sqm, very small urban lot
_LOT_SIZE_MAX = 5000.0    # sqm, large rural lot
_HEIGHT_MIN = 5.0         # metres
_HEIGHT_MAX = 100.0       # metres
_STORIES_MIN = 1
_STORIES_MAX = 30
_DENSITY_MIN = 10.0       # units/ha
_DENSITY_MAX = 500.0      # units/ha
_PARKING_MIN = 0.0
_PARKING_MAX = 3.0        # spaces per unit
_DWELLING_TYPES_MIN = 1
_DWELLING_TYPES_MAX = 8
_FRONTAGE_MIN = 4.5       # metres, very narrow urban lot
_FRONTAGE_MAX = 30.0      # metres, wide suburban lot
_FAR_MIN = 0.3
_FAR_MAX = 10.0
_LANDSCAPED_MIN = 0.0
_LANDSCAPED_MAX = 60.0    # pct
_UNIT_SIZE_MIN = 20.0     # sqm, micro unit
_UNIT_SIZE_MAX = 150.0    # sqm, large minimum


def _normalize(value: float, low: float, high: float) -> float:
    """Normalize a value to 0-1 range, clamped."""
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


def compute_restrictiveness_score(reg: ZoningRegulationCreate) -> float | None:
    """
    Compute a 0-100 restrictiveness score. Higher = more restrictive.

    Uses a weighted average of normalized metrics:
      - Large minimum lot size -> more restrictive
      - Low maximum height -> more restrictive
      - Few permitted dwelling types -> more restrictive
      - High parking requirements -> more restrictive
      - Low density -> more restrictive
      - Large lot frontage -> more restrictive
      - Low FAR -> more restrictive
      - High landscaped area req -> more restrictive
      - No secondary suites -> more restrictive
      - High minimum unit size -> more restrictive
    """
    components: list[tuple[float, float]] = []  # (score, weight)

    if reg.min_lot_size_sqm is not None:
        score = _normalize(reg.min_lot_size_sqm, _LOT_SIZE_MIN, _LOT_SIZE_MAX)
        components.append((score, 2.0))

    if reg.max_building_height_m is not None:
        score = 1.0 - _normalize(reg.max_building_height_m, _HEIGHT_MIN, _HEIGHT_MAX)
        components.append((score, 1.5))

    if reg.max_stories is not None:
        score = 1.0 - _normalize(float(reg.max_stories), float(_STORIES_MIN), float(_STORIES_MAX))
        components.append((score, 1.0))

    if reg.density_units_per_hectare is not None:
        score = 1.0 - _normalize(reg.density_units_per_hectare, _DENSITY_MIN, _DENSITY_MAX)
        components.append((score, 1.5))

    if reg.parking_spaces_per_unit is not None:
        score = _normalize(reg.parking_spaces_per_unit, _PARKING_MIN, _PARKING_MAX)
        components.append((score, 1.0))

    if reg.permitted_dwelling_types:
        score = 1.0 - _normalize(
            float(len(reg.permitted_dwelling_types)),
            float(_DWELLING_TYPES_MIN),
            float(_DWELLING_TYPES_MAX),
        )
        components.append((score, 1.5))

    # --- New scoring components ---

    if reg.min_lot_frontage_m is not None:
        score = _normalize(reg.min_lot_frontage_m, _FRONTAGE_MIN, _FRONTAGE_MAX)
        components.append((score, 1.0))

    if reg.max_floor_area_ratio is not None:
        score = 1.0 - _normalize(reg.max_floor_area_ratio, _FAR_MIN, _FAR_MAX)
        components.append((score, 1.5))

    if reg.min_landscaped_area_pct is not None:
        score = _normalize(reg.min_landscaped_area_pct, _LANDSCAPED_MIN, _LANDSCAPED_MAX)
        components.append((score, 0.5))

    if reg.secondary_suite_permitted is not None:
        score = 0.0 if reg.secondary_suite_permitted else 1.0
        components.append((score, 1.5))

    if reg.min_unit_size_sqm is not None:
        score = _normalize(reg.min_unit_size_sqm, _UNIT_SIZE_MIN, _UNIT_SIZE_MAX)
        components.append((score, 0.8))

    if not components:
        return None

    total_weight = sum(w for _, w in components)
    weighted_sum = sum(s * w for s, w in components)
    return round((weighted_sum / total_weight) * 100, 1)
