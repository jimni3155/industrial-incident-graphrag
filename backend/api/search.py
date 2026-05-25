from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.llm_service import LLMService
from services.retrieval_service import RetrievalService


router = APIRouter(prefix="/search", tags=["search"])


class QueryRequest(BaseModel):
    query:        str
    max_evidence: int = 5


class QueryResponse(BaseModel):
    query:  str
    answer: str
    seeds:  dict
    usage:  dict


def _llm() -> LLMService:
    from main import _llm
    return _llm


def _retrieval() -> RetrievalService:
    from main import _retrieval
    return _retrieval


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty")
    result = _llm().ask(req.query, req.max_evidence)
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        seeds=result["seeds"],
        usage=result["usage"],
    )


@router.get("/context/error-code/{code}")
def context_by_error_code(code: str, max_evidence: int = 5):
    text = _retrieval().retrieve_by_error_code(code.upper(), max_evidence)
    if not text:
        raise HTTPException(404, f"no context found for {code}")
    return {"error_code": code.upper(), "context": text}


@router.get("/context/failure-type/{failure_type}")
def context_by_failure_type(failure_type: str, max_evidence: int = 5):
    text = _retrieval().retrieve_by_failure_type(failure_type.upper(), max_evidence)
    if not text:
        raise HTTPException(404, f"no context found for {failure_type}")
    return {"failure_type": failure_type.upper(), "context": text}
