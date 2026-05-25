from __future__ import annotations
from fastapi import APIRouter, HTTPException
from services.retrieval_service import RetrievalService


router = APIRouter(prefix="/incidents", tags=["incidents"])


def _retrieval() -> RetrievalService:
    from main import _retrieval
    return _retrieval


@router.get("/{incident_id}/similar")
def similar_incidents(incident_id: str):
    result = _retrieval().retrieve_similar(incident_id)
    if not result:
        raise HTTPException(404, f"{incident_id} not found")
    return result


@router.get("/procedures/{error_code}")
def procedures(error_code: str):
    result = _retrieval().retrieve_procedures(error_code.upper())
    if not result:
        raise HTTPException(404, f"no procedures found for {error_code}")
    return result
