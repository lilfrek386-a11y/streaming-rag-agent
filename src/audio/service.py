import asyncio
import logging
import os
import platform
import subprocess
from pathlib import Path

from elevenlabs.core import ApiError
from openai import APIError
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from src.core.client import APIClient
from src.core.constants import AUDIO_EXTENSIONS, WHISPER_MODEL
from src.core.enums import WhisperResponseFormat
from src.core.exceptions import MissingAPIKeyError
from src.schemas import TranscriptionResult, TranscriptSegment

logger = logging.getLogger(__name__)


class AudioService:
    def __init__(self, api_client: APIClient, console: Console) -> None:
        self._api = api_client
        self._console = console

    async def transcribe_file(self, file_path: Path) -> TranscriptionResult | None:
        if not file_path.is_file():
            self._console.print(f"[red]File not found: {escape(str(file_path))}[/red]")
            return None

        if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            self._console.print(
                f"[red]Unsupported audio format: {escape(file_path.name)} "
                f"(expected one of {', '.join(sorted(AUDIO_EXTENSIONS))})[/red]"
            )
            return None

        self._console.print(
            f"[dim]> Transcribing audio: {escape(file_path.name)}[/dim]"
        )

        try:
            response = await self._api.transcribe_audio(
                file_path=file_path,
                model=WHISPER_MODEL,
                response_format=WhisperResponseFormat.VERBOSE_JSON,
            )
        except (APIError, OSError) as e:
            logger.exception("transcription failed for %s", file_path)
            self._console.print(
                f"[red]Transcription failed for {escape(file_path.name)}: "
                f"{escape(str(e))}[/red]"
            )
            return None

        try:
            return TranscriptionResult(
                text=response.text,
                segments=[
                    TranscriptSegment(start=s.start, end=s.end, text=s.text)
                    for s in (getattr(response, "segments", None) or [])
                ],
            )
        except (ValidationError, AttributeError) as e:
            logger.exception("unexpected transcription payload for %s", file_path)
            self._console.print(
                f"[red]Unexpected transcription response: {escape(str(e))}[/red]"
            )
            return None

    async def speak(self, text: str) -> None:
        if not text.strip():
            return

        self._console.print("[dim]Generating audio response...[/dim]")
        audio_path = "assistant_response.mp3"

        try:
            await self._api.generate_speech(text=text, output_path=audio_path)

            system = platform.system()
            if system == "Darwin":
                await asyncio.to_thread(
                    subprocess.run, ["afplay", audio_path], check=False
                )
            elif system == "Windows":
                await asyncio.to_thread(os.startfile, audio_path)  # type: ignore[attr-defined]
            else:
                await asyncio.to_thread(
                    subprocess.run, ["xdg-open", audio_path], check=False
                )
        except MissingAPIKeyError as e:
            self._console.print(f"[yellow]Audio skipped: {escape(str(e))}[/yellow]")
        except (ApiError, OSError) as e:
            logger.exception("TTS failed")
            self._console.print(f"[red]TTS error: {escape(str(e))}[/red]")
