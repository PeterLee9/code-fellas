from __future__ import annotations

import httpx

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_HEADERS = {"User-Agent": "ZoneMapCanada/1.0 (hackathon project)"}


async def geocode_municipality(name: str, province: str) -> tuple[float, float] | None:
    """
    Geocode a Canadian municipality name to (latitude, longitude) using Nominatim.
    Returns None if geocoding fails.
    """
    query = f"{name}, {province}, Canada"
    params = {"q": query, "format": "json", "limit": 1, "countrycodes": "ca"}

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(_NOMINATIM_URL, params=params, headers=_HEADERS)
        resp.raise_for_status()
        results = resp.json()

    if not results:
        return None

    return float(results[0]["lat"]), float(results[0]["lon"])
