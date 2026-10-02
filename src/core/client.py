import asyncio
import logging
from pathlib import Path
from typing import Any, cast

from elevenlabs import ElevenLabs
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.enums import WhisperResponseFormat
from src.core.exceptions import MissingAPIKeyError
from src.core.settings import settings
from src.schemas import ChatMessage

logger = logging.getLogger(__name__)


class APIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        elevenlabs_api_key: str | None,
        elevenlabs_voice_id: str,
        elevenlabs_model_id: str,
        elevenlabs_output_format: str,
        timeout: float = settings.groq.timeout,
    ):

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

        self._elevenlabs_voice_id = elevenlabs_voice_id
        self._elevenlabs_model_id = elevenlabs_model_id
        self._elevenlabs_output_format = elevenlabs_output_format

        self._tts_client = None
        if elevenlabs_api_key:
            self._tts_client = ElevenLabs(api_key=elevenlabs_api_key)

    @property
    def tts_available(self) -> bool:
        return self._tts_client is not None

    async def close(self) -> None:
        await self._client.close()

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
        ),
        wait=wait_exponential(min=1, max=settings.chat.retry_max_wait),
        stop=stop_after_attempt(settings.chat.max_retries),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]] | list[ChatMessage],
        model: str,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Any:
        formatted_messages = [
            m.model_dump(exclude_none=True) if isinstance(m, ChatMessage) else m
            for m in messages
        ]

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "stream": stream,
        }

        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        logger.debug(
            "chat.completions.create model=%s stream=%s messages=%d tools=%s",
            model,
            stream,
            len(formatted_messages),
            bool(tools),
        )

        return await self._client.chat.completions.create(**kwargs)

    async def transcribe_audio(
        self,
        file_path: Path,
        model: str,
        response_format: WhisperResponseFormat = WhisperResponseFormat.VERBOSE_JSON,
    ) -> Any:
        file_bytes = await asyncio.to_thread(file_path.read_bytes)
        logger.info("transcribing %s (%d bytes)", file_path.name, len(file_bytes))
        return await self._client.audio.transcriptions.create(
            file=(file_path.name, file_bytes),
            model=model,
            response_format=cast(Any, response_format.value),
        )

    async def generate_speech(self, text: str, output_path: str) -> None:
        if not self._tts_client:
            raise MissingAPIKeyError(
                "Text-to-speech is not configured (no ElevenLabs API key)."
            )

        def _create_audio():
            audio = self._tts_client.text_to_speech.convert(
                text=text,
                voice_id=self._elevenlabs_voice_id,
                model_id=self._elevenlabs_model_id,
                output_format=self._elevenlabs_output_format,
            )

            with open(output_path, "wb") as f:
                for chunk in audio:
                    if chunk:
                        f.write(chunk)

        await asyncio.to_thread(_create_audio)
