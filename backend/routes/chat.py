"""
chat.py — FastAPI router for the disaster intelligence chat endpoint.

POST /chat
    Accepts a user question, the DisasterContext generated client-side from
    all VLM outputs, and recent chat history.
    Returns a context-aware response without re-analyzing the original image.

The endpoint is intentionally stateless — conversation history is maintained
by the client and sent with each request (last N messages).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


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
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        from backend.services.chat_service import generate_response
        answer = generate_response(
            question=req.question,
            ctx=req.context.model_dump(),
            history=[m.model_dump() for m in req.history],
        )
        return ChatResponse(response=answer)

    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=f"Service not configured: {exc}")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat error: {exc}")
