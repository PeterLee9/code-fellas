"""
Multimodal embedding service using Gemini Embedding 2 Preview.

Custom LangChain Embeddings subclass that uses google.genai SDK directly
to support text, PDF, and image embedding -- all in the same 3072-dim vector space.
"""
from __future__ import annotations

import asyncio
from typing import Any

from google.genai import types as genai_types
from langchain_core.embeddings import Embeddings
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.models.database import DocumentChunkDB
from backend.models.schemas import DocumentChunkCreate

BATCH_SIZE = 50


def _get_client():
    """Get a google.genai Client instance."""
    from google import genai
    return genai.Client(api_key=get_settings().google_api_key)


def _model_name() -> str:
    return f"models/{get_settings().embedding_model}"


class MultimodalGeminiEmbeddings(Embeddings):
    """LangChain-compatible embeddings with multimodal support via google.genai SDK."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text documents. Standard LangChain interface."""
        if not texts:
            return []
        client = _get_client()
        model = _model_name()
        doc_config = genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            result = client.models.embed_content(
                model=model,
                contents=batch,
                config=doc_config,
            )
            for emb in result.embeddings:
                all_embeddings.append(emb.values)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Standard LangChain interface."""
        client = _get_client()
        result = client.models.embed_content(
            model=_model_name(),
            contents=text,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values

    def embed_pdf_pages(self, pdf_bytes: bytes, max_pages: int = 6) -> list[list[float]]:
        """
        Embed PDF pages as visual documents using Gemini's multimodal embedding.
        Each page becomes a separate embedding in the same vector space as text.
        """
        client = _get_client()
        part = genai_types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
        try:
            result = client.models.embed_content(
                model=_model_name(),
                contents=part,
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [emb.values for emb in result.embeddings[:max_pages]]
        except Exception as e:
            print(f"  [WARN] PDF multimodal embedding failed: {e}")
            return []

    def embed_images(self, image_data_list: list[tuple[bytes, str]]) -> list[list[float]]:
        """Embed image files. Each item is (image_bytes, mime_type)."""
        client = _get_client()
        model = _model_name()
        doc_config = genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        embeddings = []
        for img_bytes, mime_type in image_data_list:
            part = genai_types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
            try:
                result = client.models.embed_content(
                    model=model,
                    contents=part,
                    config=doc_config,
                )
                embeddings.append(result.embeddings[0].values)
            except Exception as e:
                print(f"  [WARN] Image embedding failed: {e}")
        return embeddings


_embeddings_instance: MultimodalGeminiEmbeddings | None = None


def get_embeddings() -> MultimodalGeminiEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = MultimodalGeminiEmbeddings()
    return _embeddings_instance


async def embed_query(text: str) -> list[float]:
    """Convenience async wrapper for query embedding."""
    embedder = get_embeddings()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embedder.embed_query, text)


async def chunk_embed_and_store(
    chunks: list[DocumentChunkCreate],
    db: AsyncSession,
    pdf_bytes_list: list[tuple[bytes, str, str]] | None = None,
    image_data_list: list[tuple[bytes, str]] | None = None,
) -> int:
    """
    Embed chunks and store them in the document_chunks table.

    Handles three modalities:
    - Text chunks: embedded via embed_documents()
    - PDF pages: embedded visually via embed_pdf_pages()
    - Images: embedded visually via embed_images()

    Args:
        chunks: List of DocumentChunkCreate objects (text, pdf_page, or image chunks)
        db: Async database session
        pdf_bytes_list: Optional list of (pdf_bytes, source_url, source_document) for multimodal PDF embedding
        image_data_list: Optional list of (image_bytes, mime_type) for multimodal image embedding

    Returns:
        Number of chunks stored.
    """
    embedder = get_embeddings()
    stored = 0
    loop = asyncio.get_event_loop()

    # 1. Embed text chunks in batches
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    if text_chunks:
        texts = [c.chunk_text for c in text_chunks]
        embeddings = await loop.run_in_executor(None, embedder.embed_documents, texts)

        for chunk, embedding in zip(text_chunks, embeddings):
            db_chunk = DocumentChunkDB(
                municipality=chunk.municipality,
                source_url=chunk.source_url,
                source_document=chunk.source_document,
                chunk_text=chunk.chunk_text,
                chunk_index=chunk.chunk_index,
                chunk_type="text",
                embedding=embedding,
            )
            db.add(db_chunk)
            stored += 1

    # 2. Embed PDF pages multimodally
    if pdf_bytes_list:
        for pdf_bytes, source_url, source_document in pdf_bytes_list:
            page_embeddings = await loop.run_in_executor(
                None, embedder.embed_pdf_pages, pdf_bytes
            )
            pdf_page_chunks = [c for c in chunks if c.chunk_type == "pdf_page" and c.source_url == source_url]
            for i, emb in enumerate(page_embeddings):
                chunk_text = (
                    pdf_page_chunks[i].chunk_text
                    if i < len(pdf_page_chunks)
                    else f"PDF page {i + 1} of {source_document}"
                )
                municipality = pdf_page_chunks[i].municipality if i < len(pdf_page_chunks) else ""
                db_chunk = DocumentChunkDB(
                    municipality=municipality,
                    source_url=source_url,
                    source_document=source_document,
                    chunk_text=chunk_text,
                    chunk_index=i,
                    chunk_type="pdf_page",
                    embedding=emb,
                )
                db.add(db_chunk)
                stored += 1

    # 3. Embed images multimodally
    if image_data_list:
        image_chunks = [c for c in chunks if c.chunk_type == "image"]
        image_embeddings = await loop.run_in_executor(
            None, embedder.embed_images, image_data_list
        )
        for i, emb in enumerate(image_embeddings):
            if i < len(image_chunks):
                chunk = image_chunks[i]
                db_chunk = DocumentChunkDB(
                    municipality=chunk.municipality,
                    source_url=chunk.source_url,
                    source_document=chunk.source_document,
                    chunk_text=chunk.chunk_text,
                    chunk_index=chunk.chunk_index,
                    chunk_type="image",
                    embedding=emb,
                )
                db.add(db_chunk)
                stored += 1

    await db.flush()
    return stored
