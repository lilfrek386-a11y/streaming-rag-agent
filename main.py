import argparse
import asyncio
import logging
from pathlib import Path

from openai import AuthenticationError
from rich.console import Console
from rich.markup import escape

from src.audio.service import AudioService
from src.chat import ChatSession, ToolExecutor
from src.commands.router import CommandRouter
from src.core.client import APIClient
from src.core.constants import CORPUS_DIR, KB_CACHE_DIR
from src.core.enums import PersonaMode
from src.core.logging_config import setup_logging
from src.core.prompts import DEFAULT_SYSTEM_PROMPT, PERSONA_PROMPTS
from src.core.settings import settings
from src.retrieval.embedder import Embedder
from src.retrieval.kb_service import KnowledgeBaseService
from src.retrieval.vector_store import VectorStore
from src.schemas.tools import TOOL_MODELS
from src.tools.functions import create_tool_functions

console = Console()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-turn CLI chat with streaming, tool calling, and audio transcription."
    )
    parser.add_argument(
        "-prompt",
        "--prompt",
        type=str,
        default=None,
        help="Custom system prompt (e.g. 'You are a math tutor')",
    )
    parser.add_argument(
        "-model",
        "--model",
        type=str,
        default=settings.groq.model,
        help="Choose the LLM model (e.g., llama3-8b-8192)",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable Text-to-Speech (TTS) for assistant responses",
    )
    parser.add_argument(
        "-persona",
        "--persona",
        type=PersonaMode,
        choices=list(PersonaMode),
        default=PersonaMode.FORMAL,
        help="Apply a specific persona to the assistant",
    )
    parser.add_argument(
        "-formal",
        "--formal",
        action="store_const",
        dest="persona",
        const=PersonaMode.FORMAL,
        help="Shortcut for -persona formal",
    )
    parser.add_argument(
        "-funny",
        "--funny",
        action="store_const",
        dest="persona",
        const=PersonaMode.FUNNY,
        help="Shortcut for -persona funny",
    )
    parser.add_argument(
        "--rebuild-kb",
        action="store_true",
        help="Ignore the cached FAISS index and re-embed the corpus from scratch",
    )
    return parser.parse_args()


def setup_knowledge_base(
    rebuild: bool = False,
) -> tuple[VectorStore, KnowledgeBaseService, dict]:
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    kb_service = KnowledgeBaseService(vector_store)

    corpus_path = Path(CORPUS_DIR)
    cache_path = Path(KB_CACHE_DIR)
    meta = kb_service.build_cache_meta(corpus_path)

    if not rebuild and vector_store.load(cache_path, expected_meta=meta):
        console.print(
            f"[dim]Loaded {len(vector_store)} chunks from cached index.[/dim]"
        )
        return vector_store, kb_service, meta

    console.print("[dim]Indexing knowledge base...[/dim]")

    if corpus_path.is_dir():
        chunks_count = kb_service.load_corpus_from_dir(corpus_path)
        console.print(f"[dim]Loaded {chunks_count} chunks from corpus.[/dim]")
        vector_store.save(cache_path, meta)
    else:
        console.print("[dim]No corpus directory found. Skipping initial loading.[/dim]")

    return vector_store, kb_service, meta


async def main():
    args = parse_args()

    setup_logging(settings.chat.log_dir, settings.chat.log_level)
    logger.info("starting session model=%s voice=%s", args.model, args.voice)

    vector_store, kb_service, kb_meta = setup_knowledge_base(rebuild=args.rebuild_kb)

    api_client = APIClient(
        api_key=settings.groq.api_key.get_secret_value(),
        base_url=settings.groq.base_url,
        elevenlabs_api_key=settings.elevenlabs.api_key_value,
        elevenlabs_voice_id=settings.elevenlabs.voice_id,
        elevenlabs_model_id=settings.elevenlabs.model_id,
        elevenlabs_output_format=settings.elevenlabs.output_format,
    )

    audio_service = AudioService(api_client, console)

    voice_enabled = args.voice
    if voice_enabled and not api_client.tts_available:
        console.print(
            "[yellow]--voice requested but ELEVENLABS_API_KEY is not set — "
            "TTS disabled for this session.[/yellow]"
        )
        voice_enabled = False

    system_prompt = args.prompt or PERSONA_PROMPTS.get(
        args.persona, DEFAULT_SYSTEM_PROMPT
    )

    tools_map = create_tool_functions(
        vector_store=vector_store,
        get_messages_callback=lambda: session.messages,
    )
    tool_executor = ToolExecutor(tools_map, TOOL_MODELS)

    session = ChatSession(
        api_client=api_client,
        model_name=args.model,
        console=console,
        tool_executor=tool_executor,
        audio_service=audio_service,
        system_prompt=system_prompt,
        voice_enabled=voice_enabled,
    )

    router = CommandRouter(
        session=session,
        kb_service=kb_service,
        audio_service=audio_service,
        console=console,
    )

    if args.prompt:
        console.print(f"> System Prompt: {escape(args.prompt)}", style="bold magenta")
    else:
        console.print(f"> Persona: {args.persona.value}", style="bold magenta")

    if voice_enabled:
        console.print("> Voice Output (TTS): ENABLED", style="bold green")

    try:
        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ")
            except (KeyboardInterrupt, EOFError):
                console.print("\nInterrupted.", style="dim")
                break

            if not user_input.strip():
                continue

            try:
                if user_input.lstrip().startswith("/"):
                    should_continue = await router.dispatch(user_input)
                    if not should_continue:
                        break
                    continue

                await session.send_message(user_input)
            except AuthenticationError:
                console.print("Invalid API key — exiting.", style="bold red")
                break
            except KeyboardInterrupt:
                console.print("\n[dim]Turn cancelled.[/dim]")
                continue
            except Exception:  # noqa: BLE001
                logger.exception("turn failed")
                continue
    finally:
        try:
            vector_store.save(Path(KB_CACHE_DIR), kb_meta)
        except OSError as e:
            logger.exception("could not save index")
            console.print(f"[dim]Could not save index: {escape(str(e))}[/dim]")

        session.print_summary()
        try:
            session.save_log()
        except OSError as e:
            logger.exception("could not save log")
            console.print(f"[dim]Could not save log: {escape(str(e))}[/dim]")
        finally:
            await api_client.close()
        logger.info("session finished, total tokens=%d", session.total_tokens)


if __name__ == "__main__":
    asyncio.run(main())
