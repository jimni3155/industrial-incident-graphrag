from __future__ import annotations

from openai import OpenAI
from core.config import settings


def get_embedding_client() -> OpenAI:
    kwargs: dict = {"api_key": settings.OPENAI_API_KEY}
    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL
    return OpenAI(**kwargs)


def get_embedding_model() -> str:
    return settings.EMBEDDING_MODEL