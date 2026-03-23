"""Vector search retriever using pgvector."""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.rag.embeddings import embed_query


async def retrieve_similar_chunks(
    query: str,
    db: AsyncSession,
    municipality: str | None = None,
    top_k: int = 8,
) -> list[dict]:
    """Retrieve the most similar document chunks for a query using vector search."""
    query_embedding = await embed_query(query)

    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    if municipality:
        sql = text(
            "SELECT id, municipality, source_url, source_document, "
            "chunk_text, chunk_index, chunk_type, "
            "embedding <=> cast(:emb as vector) AS distance "
            "FROM document_chunks "
            "WHERE embedding IS NOT NULL "
            "AND municipality ILIKE :municipality "
            "ORDER BY embedding <=> cast(:emb as vector) "
            "LIMIT :top_k"
        )
        params = {"emb": embedding_str, "municipality": f"%{municipality}%", "top_k": top_k}
    else:
        sql = text(
            "SELECT id, municipality, source_url, source_document, "
            "chunk_text, chunk_index, chunk_type, "
            "embedding <=> cast(:emb as vector) AS distance "
            "FROM document_chunks "
            "WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> cast(:emb as vector) "
            "LIMIT :top_k"
        )
        params = {"emb": embedding_str, "top_k": top_k}

    result = await db.execute(sql, params)
    rows = result.fetchall()

    return [
        {
            "id": row[0],
            "municipality": row[1],
            "source_url": row[2],
            "source_document": row[3],
            "chunk_text": row[4],
            "chunk_index": row[5],
            "chunk_type": row[6],
            "distance": float(row[7]),
        }
        for row in rows
    ]
