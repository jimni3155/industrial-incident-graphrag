from __future__ import annotations

from openai import OpenAI

from core.config import settings


def get_client() -> OpenAI:
    return OpenAI(
        api_key=settings.GATEWAY_API_KEY,
        base_url=settings.GATEWAY_BASE_URL,
    )