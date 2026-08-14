from pathlib import Path

from rich.console import Console

from src.core.client import APIClient
from src.core.constants import AUDIO_EXTENSIONS, WHISPER_MODEL, WhisperResponseFormat
from src.core.prompts import TRANSCRIPTION_PROMPTS, TranscriptionMode
from src.core.schemas import ChatMessage, TranscriptionResult, TranscriptSegment
from src.core.settings import settings
from src.retrieval.chunker import chunk_text
from src.retrieval.vector_store import VectorStore


class AudioService:
    def __init__(
        self,
        api_client: APIClient,
        vector_store: VectorStore,
        console: Console,
    ) -> None:
        self._api = api_client
        self._vector_store = vector_store
        self._console = console

    async def process_files(self, paths_part: str, mode: TranscriptionMode) -> None:
        paths = [
            p.strip().strip('"').strip("'") for p in paths_part.split(";") if p.strip()
        ]
        for path_str in paths:
            await self._process_single_file(Path(path_str), mode)

    async def _process_single_file(
        self, file_path: Path, mode: TranscriptionMode
    ) -> None:
        if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            self._console.print(
                f"[red]Unsupported audio format: {file_path.name}[/red]"
            )
            return

        if not file_path.exists():
            self._console.print(f"[red]File not found: {file_path}[/red]")
            return

        self._console.print(f"[dim]> File uploaded: {file_path.name}[/dim]")

        transcript = await self._transcribe(file_path)
        if transcript is None:
            return

        self._console.print(
            f"\n[bold]→ Whisper Transcript:[/bold]\n{transcript.text}\n"
        )

        summary = await self._summarize(transcript.text, mode)
        if summary is None:
            return

        self._console.print(f"[bold]→ GPT Summary ({mode.value}):[/bold]\n{summary}\n")

        self._index(transcript.text, file_path.name)

    async def _transcribe(self, file_path: Path) -> TranscriptionResult | None:
        try:
            response = await self._api.transcribe_audio(
                file_path=file_path,
                model=WHISPER_MODEL,
                response_format=WhisperResponseFormat.VERBOSE_JSON,
            )
        except Exception as e:
            self._console.print(
                f"[red]Transcription failed for {file_path.name}: {e}[/red]"
            )
            return None

        return TranscriptionResult(
            text=response.text,
            segments=[
                TranscriptSegment(start=s.start, end=s.end, text=s.text)
                for s in getattr(response, "segments", [])
            ],
        )

    async def _summarize(
        self, transcript_text: str, mode: TranscriptionMode
    ) -> str | None:
        prompt = TRANSCRIPTION_PROMPTS[mode]
        try:
            response = await self._api.create_chat_completion(
                messages=[
                    ChatMessage(
                        role="system",
                        content=prompt,
                    ),
                    ChatMessage(
                        role="user",
                        content=transcript_text,
                    ),
                ],
                model=settings.groq.model,
                stream=False,
            )
        except Exception as e:
            self._console.print(f"[red]Summary failed: {e}[/red]")
            return None

        return response.choices[0].message.content or ""

    def _index(self, text: str, source: str) -> None:
        chunks = chunk_text(text)
        entries = [{"text": c, "source": source} for c in chunks]
        self._vector_store.add_many(entries)
        self._console.print(
            f"[dim]Indexed {len(entries)} chunks from {source} into the knowledge base.[/dim]"
        )
