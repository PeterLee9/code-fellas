"""Text chunking and structured-to-natural-language conversion for RAG."""
from __future__ import annotations

from backend.models.schemas import DocumentChunkCreate


def chunk_text(
    text: str,
    municipality: str,
    source_url: str,
    source_document: str,
    chunk_size: int = 800,
    overlap: int = 200,
) -> list[DocumentChunkCreate]:
    """Split raw text into overlapping chunks for embedding."""
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(
                DocumentChunkCreate(
                    municipality=municipality,
                    source_url=source_url,
                    source_document=source_document,
                    chunk_text=chunk.strip(),
                    chunk_index=idx,
                    chunk_type="text",
                )
            )
            idx += 1
        start += chunk_size - overlap

    return chunks


def _format_metric(label: str, value, unit: str = "") -> str:
    if value is None:
        return ""
    return f"{label}: {value}{unit}"


def _format_bool(label: str, value) -> str:
    if value is None:
        return ""
    return f"{label}: {'Yes' if value else 'No'}"


def _format_list(label: str, items) -> str:
    if not items:
        return ""
    return f"{label}: {', '.join(items)}"


def regulation_to_text(reg) -> str:
    """Convert a single ZoningRegulation (DB row or Pydantic model) into a natural-language paragraph."""
    municipality = getattr(reg, "municipality", "Unknown")
    zone_code = getattr(reg, "zone_code", "")
    zone_name = getattr(reg, "zone_name", "")
    zone_category = getattr(reg, "zone_category", "")
    if hasattr(zone_category, "value"):
        zone_category = zone_category.value

    lines = [
        f"In {municipality}, zone {zone_code} ({zone_name}) is classified as {zone_category}."
    ]

    bylaw = getattr(reg, "bylaw_number", None)
    if bylaw:
        effective = getattr(reg, "bylaw_effective_date", None)
        date_part = f" (effective {effective})" if effective else ""
        lines.append(f"Bylaw: {bylaw}{date_part}.")

    metrics = [
        _format_metric("Minimum lot size", getattr(reg, "min_lot_size_sqm", None), " sqm"),
        _format_metric("Minimum lot frontage", getattr(reg, "min_lot_frontage_m", None), " m"),
        _format_metric("Maximum building height", getattr(reg, "max_building_height_m", None), " m"),
        _format_metric("Maximum stories", getattr(reg, "max_stories", None)),
        _format_metric("Maximum lot coverage", getattr(reg, "max_lot_coverage_pct", None), "%"),
        _format_metric("Maximum floor area ratio (FAR)", getattr(reg, "max_floor_area_ratio", None)),
        _format_metric("Minimum front setback", getattr(reg, "min_front_setback_m", None), " m"),
        _format_metric("Minimum rear setback", getattr(reg, "min_rear_setback_m", None), " m"),
        _format_metric("Minimum side setback", getattr(reg, "min_side_setback_m", None), " m"),
        _format_metric("Minimum landscaped area", getattr(reg, "min_landscaped_area_pct", None), "%"),
        _format_metric("Parking spaces per unit", getattr(reg, "parking_spaces_per_unit", None)),
        _format_metric("Maximum units per lot", getattr(reg, "max_units_per_lot", None)),
        _format_metric("Density", getattr(reg, "density_units_per_hectare", None), " units/ha"),
        _format_metric("Minimum unit size", getattr(reg, "min_unit_size_sqm", None), " sqm"),
        _format_metric("Inclusionary zoning", getattr(reg, "inclusionary_zoning_pct", None), "%"),
    ]
    metric_parts = [m for m in metrics if m]
    if metric_parts:
        lines.append("Development standards: " + ". ".join(metric_parts) + ".")

    dwelling_types = getattr(reg, "permitted_dwelling_types", None) or []
    if dwelling_types:
        lines.append("Permitted dwelling types: " + ", ".join(dwelling_types) + ".")

    commercial_uses = getattr(reg, "permitted_commercial_uses", None) or []
    if commercial_uses:
        lines.append("Permitted commercial uses: " + ", ".join(commercial_uses) + ".")

    prohibited = getattr(reg, "prohibited_uses", None) or []
    if prohibited:
        lines.append("Prohibited uses: " + ", ".join(prohibited) + ".")

    bools = [
        _format_bool("Home occupation permitted", getattr(reg, "home_occupation_permitted", None)),
        _format_bool("Secondary suite permitted", getattr(reg, "secondary_suite_permitted", None)),
        _format_bool("Short-term rental permitted", getattr(reg, "short_term_rental_permitted", None)),
    ]
    bool_parts = [b for b in bools if b]
    if bool_parts:
        lines.append(". ".join(bool_parts) + ".")

    overlay = getattr(reg, "overlay_district", None)
    if overlay:
        lines.append(f"Overlay district: {overlay}.")

    score = getattr(reg, "restrictiveness_score", None)
    if score is not None:
        lines.append(f"Restrictiveness score: {score}/100.")

    return " ".join(lines)


def policy_to_text(pol) -> str:
    """Convert a single OfficialPlanPolicy (DB row or Pydantic model) into a natural-language paragraph."""
    municipality = getattr(pol, "municipality", "Unknown")
    policy_area = getattr(pol, "policy_area", "")
    designation = getattr(pol, "land_use_designation", "")
    growth = getattr(pol, "growth_targets", None)
    density = getattr(pol, "density_targets", None)
    transit = getattr(pol, "transit_policy", None)
    affordable = getattr(pol, "affordable_housing_policy", None)
    heritage = getattr(pol, "heritage_conservation", None)
    uses = getattr(pol, "permitted_uses_summary", "")

    lines = [
        f"In {municipality}'s official plan, the {policy_area} area has a land use designation of {designation}."
    ]
    if growth:
        lines.append(f"Growth targets: {growth}.")
    if density:
        lines.append(f"Density targets: {density}.")
    if transit:
        lines.append(f"Transit policy: {transit}.")
    if affordable:
        lines.append(f"Affordable housing policy: {affordable}.")
    if heritage:
        lines.append(f"Heritage conservation: {heritage}.")
    if uses:
        lines.append(f"Permitted uses: {uses}.")
    return " ".join(lines)


def structured_to_chunks(
    regulations: list,
    policies: list | None = None,
    source_url: str = "",
    source_document: str = "",
) -> list[DocumentChunkCreate]:
    """Convert structured zoning regulations and official plan policies into embeddable text chunks."""
    chunks = []

    for i, reg in enumerate(regulations):
        municipality = getattr(reg, "municipality", "Unknown")
        url = getattr(reg, "source_url", source_url) or source_url
        doc = getattr(reg, "source_document", source_document) or source_document
        text = regulation_to_text(reg)
        chunks.append(
            DocumentChunkCreate(
                municipality=municipality,
                source_url=url,
                source_document=doc,
                chunk_text=text,
                chunk_index=i,
                chunk_type="text",
            )
        )

    for j, pol in enumerate(policies or []):
        municipality = getattr(pol, "municipality", "Unknown")
        url = getattr(pol, "source_url", source_url) or source_url
        doc = getattr(pol, "source_document", source_document) or source_document
        text = policy_to_text(pol)
        chunks.append(
            DocumentChunkCreate(
                municipality=municipality,
                source_url=url,
                source_document=doc,
                chunk_text=text,
                chunk_index=len(regulations) + j,
                chunk_type="text",
            )
        )

    return chunks


def pdf_to_page_chunks(
    pdf_bytes: bytes,
    municipality: str,
    source_url: str,
    source_document: str,
) -> tuple[list[DocumentChunkCreate], bytes]:
    """
    Create one chunk per PDF page for multimodal embedding.
    Returns (chunks, pdf_bytes) -- the pdf_bytes are passed through
    so the embedder can embed the PDF pages visually.
    """
    try:
        import pdfplumber
        import io
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
        page_count = len(pdf.pages)
        pdf.close()
    except Exception:
        page_count = 1

    chunks = []
    for i in range(min(page_count, 6)):
        chunks.append(
            DocumentChunkCreate(
                municipality=municipality,
                source_url=source_url,
                source_document=source_document,
                chunk_text=f"PDF page {i + 1} of {source_document} ({municipality})",
                chunk_index=i,
                chunk_type="pdf_page",
            )
        )

    return chunks, pdf_bytes
