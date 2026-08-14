from pathlib import Path

from src.core.client import APIClient
from src.core.settings import settings


async def transcribe(api_client: APIClient, file_path: Path) -> dict:
    response = await api_client.transcribe_audio(
        file_path=file_path,
        model=settings.groq.whisper_model,
        response_format="verbose_json",
    )

    return {
        "text": response.text,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in getattr(response, "segments", [])
        ],
    }
