from pathlib import Path
from rich.console import Console

from src.core.client import APIClient
from src.core.constants import AUDIO_EXTENSIONS
from src.core.prompts import TranscriptionMode
from src.core.settings import settings
from src.retrieval.chunker import chunk_text
from src.retrieval.vector_store import VectorStore
from src.audio.transcriber import transcribe
from src.audio.summarizer import summarize

console = Console()


async def process_audio_files(
    api_client: APIClient,
    vector_store: VectorStore,
    paths_part: str,
    mode: TranscriptionMode,
) -> None:
    paths = [
        p.strip().strip('"').strip("'") for p in paths_part.split(";") if p.strip()
    ]

    for path_str in paths:
        file_path = Path(path_str)

        if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            console.print(f"[red]Unsupported audio format: {file_path.name}[/red]")
            continue

        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            continue

        console.print(f"[dim]> File uploaded: {file_path.name}[/dim]")

        result = await transcribe(api_client, file_path)
        if result is None:
            continue

        console.print(f"\n[bold]→ Whisper Transcript:[/bold]\n{result['text']}\n")

        summary = await summarize(
            api_client, result["text"], model=settings.groq.model, mode=mode
        )
        if summary is None:
            continue

        console.print(f"[bold]→ GPT Summary ({mode.value}):[/bold]\n{summary}\n")

        chunks = chunk_text(result["text"])
        entries = [{"text": c, "source": file_path.name} for c in chunks]
        vector_store.add_many(entries)
        console.print(
            f"[dim]Indexed {len(entries)} chunks from {file_path.name} into the knowledge base.[/dim]"
        )
