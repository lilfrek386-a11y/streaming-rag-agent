from pathlib import Path

from rich.console import Console

from src.core.client import APIClient
from src.core.constants import WHISPER_MODEL

console = Console()


async def transcribe(api_client: APIClient, file_path: Path) -> dict | None:
    try:
        response = await api_client.transcribe_audio(
            file_path=file_path,
            model=WHISPER_MODEL,
            response_format="verbose_json",
        )
    except Exception as e:
        console.print(f"[red]Transcription failed for {file_path.name}: {e}[/red]")
        return None

    return {
        "text": response.text,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in getattr(response, "segments", [])
        ],
    }
