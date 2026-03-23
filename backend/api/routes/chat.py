from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.models.schemas import ChatRequest, ChatResponse
from backend.rag.chat_agent import agentic_chat_answer, agentic_chat_stream

router = APIRouter(prefix="/chat", tags=["chat"])


def _history_to_dicts(request: ChatRequest) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in request.history]


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Agentic RAG chat: LangGraph agent with SQL, vector search, and web search tools."""
    result = await agentic_chat_answer(
        question=request.question,
        db=db,
        municipality=request.municipality,
        history=_history_to_dicts(request),
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """SSE streaming endpoint for the agentic RAG chat with real-time thought process."""
    return StreamingResponse(
        agentic_chat_stream(
            question=request.question,
            db=db,
            municipality=request.municipality,
            history=_history_to_dicts(request),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
