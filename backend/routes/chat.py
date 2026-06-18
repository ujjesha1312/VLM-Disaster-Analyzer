"""
chat.py — FastAPI router for the disaster intelligence chat endpoint.

POST /chat
    Accepts a user question, the DisasterContext generated client-side from
    all VLM outputs, and recent chat history.
    Returns a context-aware response without re-analyzing the original image.

The endpoint is intentionally stateless — conversation history is maintained
by the client and sent with each request (last N messages).

Hardening changes:
  Fix J — Question length capped at 2 000 characters.  Empty or over-length
           questions return HTTP 400 with a clear error message.
           History items are capped at 500 characters each to prevent
           unbounded context injection.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# Fix J — Input limits
_MAX_QUESTION_LEN: int = 2_000   # characters
_MAX_HISTORY_MSG_LEN: int = 500  # characters per message in history


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role:    str = Field(..., pattern="^(user|assistant)$")
    content: str


class DisasterContext(BaseModel):
    eventType:    str   = "Unknown"
    confidence:   float = 0.0
    caption:      str   = ""
    reasoning:    str   = ""
    sceneAnalysis: str  = ""
    severity:     str   = "Unknown"


class ChatRequest(BaseModel):
    question: str
    context:  DisasterContext
    history:  list[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse, summary="Disaster intelligence chat")
def chat(req: ChatRequest) -> ChatResponse:
    """
    Answer a follow-up question about a previously analysed disaster image.

    The image is NOT re-analysed — only the pre-extracted DisasterContext
    and recent chat history are used to generate the response.
    """
    # Fix J — Validate question
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if len(req.question) > _MAX_QUESTION_LEN:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Question exceeds maximum length of {_MAX_QUESTION_LEN} characters "
                f"({len(req.question)} received). Please shorten your question."
            ),
        )

    # Fix J — Truncate overly-long history messages to limit context injection surface
    sanitized_history = [
        {"role": m.role, "content": m.content[:_MAX_HISTORY_MSG_LEN]}
        for m in req.history
    ]

    try:
        from backend.services.chat_service import generate_response
        answer = generate_response(
            question=req.question,
            ctx=req.context.model_dump(),
            history=sanitized_history,
        )
        return ChatResponse(response=answer)

    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=f"Service not configured: {exc}")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat error: {exc}")
