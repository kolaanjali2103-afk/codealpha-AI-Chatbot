"""
api.py
------
FastAPI REST API server for the FAQ chatbot.

This exposes the TF-IDF + cosine-similarity matching logic from chatbot.py
as an HTTP API endpoint so that external frontends (Next.js, mobile apps, etc.)
can consume it.

Run with:
    uvicorn src.api:app --reload --port 8000

The endpoint:
    POST /api/chat
    Body: { "question": "How do I reset my password?" }
    Returns: { "answer": "...", "matched_question": "...", "category": "...",
               "similarity": 0.97, "is_fallback": false }
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.chatbot import FAQChatbot

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FAQ Chatbot API",
    description="TF-IDF + cosine similarity FAQ matching engine.",
    version="1.0.0",
)

# Allow any origin so the Next.js frontend (on Vercel or localhost:3000)
# can talk to this API without browser CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load chatbot once at startup – avoids reloading the JSON / re-building
# the TF-IDF matrix on every request.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FAQ_PATH = os.path.join(_HERE, "data", "faqs.json")
_chatbot = FAQChatbot(_FAQ_PATH)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    matched_question: str | None
    category: str | None
    similarity: float
    is_fallback: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    """Simple health-check endpoint."""
    return {"status": "ok", "message": "FAQ Chatbot API is running."}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Accept a user question and return the best matching FAQ answer
    (or a graceful fallback if no FAQ matches with enough confidence).
    """
    result = _chatbot.get_response(request.question)
    return ChatResponse(
        answer=result.answer,
        matched_question=result.matched_question,
        category=result.category,
        similarity=round(result.similarity, 4),
        is_fallback=result.is_fallback,
    )
