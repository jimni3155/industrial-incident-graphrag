from __future__ import annotations
from typing import Any
from core.config import settings
from services.retrieval_service import RetrievalService
from common.llm_client import get_client

_SYSTEM_PROMPT = """\
You are an industrial maintenance expert AI assistant.
You answer questions about equipment failures, maintenance procedures, and anomaly diagnosis
based strictly on the provided graph context.

Rules:
- Ground every claim in the provided context. Do not hallucinate incident IDs, error codes, or component names.
- If the context does not contain enough information, say so explicitly.
- When referencing evidence (images, sensor graphs, dashboards), mention the ID and key attributes.
- Keep answers concise and actionable.
"""


class LLMService:
    def __init__(self) -> None:
        self._client = get_client()
        self._retrieval = RetrievalService()

    def close(self) -> None:
        self._retrieval.close()

    def __enter__(self) -> "LLMService":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def ask(self, query: str, max_evidence: int = 5) -> dict[str, Any]:
        retrieval_result = self._retrieval.retrieve(query, max_evidence)
        context_text     = retrieval_result["combined"]

        if not context_text:
            context_text = "No relevant graph context found for this query."

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"[Graph Context]\n{context_text}\n\n[Question]\n{query}"},
        ]

        response = self._client.chat.completions.create(
            model=settings.DEFAULT_LLM_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
        )

        return {
            "query":    query,
            "answer":   response.choices[0].message.content,
            "seeds":    retrieval_result["seeds"],
            "contexts": retrieval_result["contexts"],
            "usage":    response.usage.model_dump() if response.usage else {},
        }

    def ask_with_code(self, error_code: str, question: str) -> dict[str, Any]:
        context_text = self._retrieval.retrieve_by_error_code(error_code)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"[Graph Context — {error_code}]\n{context_text}"
                    f"\n\n[Question]\n{question}"
                ),
            },
        ]
        response = self._client.chat.completions.create(
            model=settings.DEFAULT_LLM_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
        )
        return {
            "error_code": error_code,
            "answer":     response.choices[0].message.content,
            "usage":      response.usage.model_dump() if response.usage else {},
        }