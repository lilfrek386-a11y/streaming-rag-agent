import argparse
import asyncio
from pathlib import Path

from openai import AuthenticationError
from rich.console import Console

from src.audio.service import AudioService
from src.chat_session import ChatSession
from src.core.client import APIClient
from src.core.constants import QUIT_COMMANDS
from src.core.prompts import TranscriptionMode
from src.core.settings import settings
from src.corpus.loader import load_corpus
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.tools.functions import TOOL_FUNCTIONS, make_search_knowledge_base

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
        type=TranscriptionMode,
        choices=list(TranscriptionMode),
        default=TranscriptionMode.SUMMARY,
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


async def main():
    args = parse_args()

    vector_store, _ = setup_knowledge_base()

    TOOL_FUNCTIONS["search_knowledge_base"] = make_search_knowledge_base(
        vector_store, console
    )

    api_client = APIClient(
        api_key=settings.groq.api_key, base_url=settings.groq.base_url
    )
    audio_service = AudioService(api_client, vector_store, console)

    session = ChatSession(
        api_client,
        model_name=settings.groq.model,
        console=console,
        system_prompt=args.prompt,
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
                paths_string = user_input.split(":", 1)[1].strip()
                await audio_service.process_files(paths_string, args.mode)
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
