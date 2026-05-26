from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.llm_service import LLMService
from services.retrieval_service import RetrievalService
from services.agent_service import AgentService

from api.search import router as search_router
from api.incidents import router as incidents_router
from api.agent import router as agent_router

_llm:       LLMService | None        = None
_retrieval: RetrievalService | None  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm, _retrieval
    _llm       = LLMService()
    _retrieval = _llm._retrieval
    yield
    _llm.close()


app = FastAPI(
    title="Industrial GraphRAG API",
    description="Multimodal GraphRAG for industrial incident analysis",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router,    prefix="/api/v1")
app.include_router(incidents_router, prefix="/api/v1")
app.include_router(agent_router,     prefix="/api/v1")

@app.get("/", tags=["system"])
def root():
    return {
        "message": "Industrial GraphRAG API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}
