from openai import (
    AsyncOpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError
)
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

from src.core.settings import settings


def create_client() -> AsyncOpenAI:
    if not settings.groq.api_key:
        print("Error: GROQ_API_KEY not found in .env")
        raise SystemExit(1)

    return AsyncOpenAI(
        base_url=settings.groq.base_url,
        api_key=settings.groq.api_key,
    )


class APIClient:
    def __init__(self, client: AsyncOpenAI):
        self._client = client

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)),
        wait=wait_exponential(min=1, max=settings.chat.retry_max_wait),
        stop=stop_after_attempt(settings.chat.max_retries),
        reraise=True,
    )
    async def create_chat_completion(
            self,
            messages: list[dict],
            model: str,
            stream: bool = False,
            tools: list | None = None,
            tool_choice: str | None = None
    ):
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        return await self._client.chat.completions.create(**kwargs)