from openai import AsyncOpenAI
from src.core.settings import settings


def create_client() -> AsyncOpenAI:
    if not settings.groq.api_key:
        print("Error: GROQ_API_KEY not found in .env")
        raise SystemExit(1)

    return AsyncOpenAI(
        base_url=settings.groq.base_url,
        api_key=settings.groq.api_key,
    )