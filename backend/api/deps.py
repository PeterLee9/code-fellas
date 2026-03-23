"""FastAPI dependency injection."""
from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.models.database import get_async_engine, get_async_session_factory

_engine = None
_session_factory = None


def _init():
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        _engine = get_async_engine(settings.database_url)
        _session_factory = get_async_session_factory(_engine)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    _init()
    async with _session_factory() as session:
        yield session
