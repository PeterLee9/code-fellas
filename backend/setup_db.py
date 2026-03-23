"""
One-time database setup script.
Creates all tables in Supabase and enables required extensions.
Run with: python -m backend.setup_db
"""
import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings
from backend.models.database import Base, get_async_engine


async def setup():
    settings = get_settings()
    engine = get_async_engine(settings.database_url)

    async with engine.begin() as conn:
        # Enable extensions
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        print("[OK] Extensions enabled: vector, postgis")

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("[OK] All tables created")

    await engine.dispose()
    print("[DONE] Database setup complete")


if __name__ == "__main__":
    asyncio.run(setup())
