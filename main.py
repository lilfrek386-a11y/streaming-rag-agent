import argparse
import asyncio
from pathlib import Path

from openai import AuthenticationError
from rich.console import Console

from src.chat_session import ChatSession
from src.core.client import APIClient
from src.core.constants import QUIT_COMMANDS, AUDIO_EXTENSIONS
from src.core.prompts import SummaryMode
from src.core.settings import settings
from src.corpus.loader import load_corpus
from src.retrieval.chunker import chunk_text
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.tools.functions import TOOL_FUNCTIONS, make_search_knowledge_base
from src.audio.transcriber import transcribe
from src.audio.summarizer import summarize

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-turn CLI chat with streaming, tool calling, and audio transcription."
    )
    parser.add_argument(
        "-prompt",
        type=str,
        default=None,
        help="Custom system prompt (e.g. 'You are a math tutor')",
    )
    parser.add_argument(
        "-mode",
        type=SummaryMode,
        choices=list(SummaryMode),
        default=SummaryMode.SUMMARY,
        help="Summarization mode for 'file:' audio input (default: summary)",
    )
    return parser.parse_args()


def setup_knowledge_base() -> tuple[VectorStore, Embedder]:
    console.print("[dim]Indexing knowledge base...[/dim]")
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    corpus_path = Path("src/corpus")
    entries = load_corpus(corpus_dir=corpus_path)
    vector_store.add_many(entries)
    return vector_store, embedder


async def handle_audio_input(
    api_client: APIClient,
    vector_store: VectorStore,
    raw_input: str,
    mode: SummaryMode,
) -> None:
    paths_part = raw_input.split(":", 1)[1].strip()
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


async def main():
    args = parse_args()

    vector_store, _ = setup_knowledge_base()

    TOOL_FUNCTIONS["search_knowledge_base"] = make_search_knowledge_base(vector_store)

    api_client = APIClient(
        api_key=settings.groq.api_key, base_url=settings.groq.base_url
    )
    session = ChatSession(
        api_client, model_name=settings.groq.model, system_prompt=args.prompt
    )

    if args.prompt:
        console.print(f"> System Prompt: {args.prompt}", style="bold magenta")

    console.print(
        "[dim]Tip: type 'file: path/to/audio.mp3' to transcribe and summarize audio.[/dim]"
    )

    try:
        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ")
            except KeyboardInterrupt:
                console.print("\nInterrupted.", style="dim")
                break

            if not user_input.strip():
                continue

            if user_input.lower() in QUIT_COMMANDS:
                break

            if user_input.lower().startswith("file:"):
                await handle_audio_input(
                    api_client, vector_store, user_input, args.mode
                )
                continue

            try:
                await session.send_message(user_input)
            except AuthenticationError:
                console.print("Invalid API key — exiting.", style="bold red")
                break
            except Exception as e:
                console.print(f"Message failed: {e}", style="red")
                continue
    finally:
        session.print_summary()
        session.save_log()


if __name__ == "__main__":
    asyncio.run(main())
