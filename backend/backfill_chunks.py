"""
Backfill document_chunks for existing data in the database.
Converts structured zoning regulations into natural-language chunks,
embeds them, and stores in the document_chunks table for RAG.

Usage: PYTHONPATH=. python backend/backfill_chunks.py
"""
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from backend.config import get_settings
from backend.models.database import (
    ZoningRegulationDB,
    OfficialPlanPolicyDB,
    DocumentChunkDB,
    get_async_engine,
    get_async_session_factory,
)
from backend.rag.chunker import structured_to_chunks
from backend.rag.embeddings import chunk_embed_and_store

from sqlalchemy import select, func


async def main():
    settings = get_settings()
    engine = get_async_engine(settings.database_url)
    Session = get_async_session_factory(engine)

    async with Session() as session:
        # Check existing chunk count
        count_result = await session.execute(select(func.count(DocumentChunkDB.id)))
        existing = count_result.scalar() or 0
        print(f"Existing document chunks: {existing}")

        # Load all regulations
        result = await session.execute(select(ZoningRegulationDB))
        regulations = result.scalars().all()
        print(f"Found {len(regulations)} zoning regulations to convert")

        # Load all policies
        result = await session.execute(select(OfficialPlanPolicyDB))
        policies = result.scalars().all()
        print(f"Found {len(policies)} official plan policies to convert")

        if not regulations and not policies:
            print("No data to backfill. Run seed_toronto.py or the agent first.")
            return

        # Convert to text chunks
        chunks = structured_to_chunks(
            regulations=regulations,
            policies=policies,
        )
        print(f"Created {len(chunks)} text chunks from structured data")

        # Embed and store in batches
        print("Embedding chunks (this may take a minute)...")
        stored = await chunk_embed_and_store(chunks, session)
        await session.commit()

        print(f"Successfully embedded and stored {stored} chunks")

        # Verify
        count_result = await session.execute(select(func.count(DocumentChunkDB.id)))
        total = count_result.scalar() or 0
        print(f"Total document chunks now: {total}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
