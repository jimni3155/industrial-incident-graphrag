from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.agent_service import AgentService


router = APIRouter(prefix="/agent", tags=["agent"])


class InvestigateRequest(BaseModel):
    query:          str
    max_iterations: int = 5


@router.post("/investigate")
def investigate(req: InvestigateRequest):
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty")
    with AgentService() as svc:
        return svc.investigate(req.query, req.max_iterations)