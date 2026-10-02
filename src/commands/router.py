import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from src.audio.service import AudioService
from src.chat import ChatSession
from src.retrieval.kb_service import KnowledgeBaseService

logger = logging.getLogger(__name__)


class CommandRouter:
    def __init__(
        self,
        session: ChatSession,
        kb_service: KnowledgeBaseService,
        audio_service: AudioService,
        console: Console,
    ):
        self.session = session
        self.kb_service = kb_service
        self.audio_service = audio_service
        self.console = console

        self._handlers: dict[str, Callable[[str], Awaitable[bool]]] = {
            "/exit": self._handle_exit,
            "/help": self._handle_help,
            "/update_kb_text": self._handle_update_kb_text,
            "/update_kb_voice": self._handle_update_kb_voice,
            "/change_prompt": self._handle_change_prompt,
            "/search": self._handle_search,
            "/summarize_session": self._handle_summarize,
            "/save_session": self._handle_save_session,
            "/load_session": self._handle_load_session,
            "/retry": self._handle_retry,
        }

    async def dispatch(self, user_input: str) -> bool:
        parts = user_input.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handler = self._handlers.get(command)

        if not handler:
            self.console.print(f"[red]Unknown command:[/red] {escape(command)}")
            return True

        logger.info("command %s", command)
        return await handler(arg)

    async def _handle_exit(self, _arg: str) -> bool:
        self.console.print("[bold red]Exiting... Goodbye![/bold red]")
        return False

    async def _handle_help(self, _arg: str) -> bool:
        self.console.print("[bold]Commands:[/bold] " + ", ".join(self._handlers))
        return True

    async def _handle_update_kb_text(self, arg: str) -> bool:
        if not arg:
            arg = self.console.input(
                "[bold yellow]Enter your fact:[/bold yellow] "
            ).strip()

        if not arg:
            self.console.print("[yellow]No text provided. Skipped.[/yellow]")
            return True

        chunks_count = self.kb_service.add_document(text=arg, source="manual_input")

        self.console.print(
            f"[green]Successfully added {chunks_count} chunk(s) to Knowledge Base![/green]"
        )

        return True

    async def _handle_update_kb_voice(self, arg: str) -> bool:
        if not arg:
            arg = self.console.input(
                "[bold yellow]Enter audio file path (or paths separated by semicolon):[/bold yellow] "
            ).strip()

        if not arg:
            self.console.print("[yellow]No path provided. Skipped.[/yellow]")
            return True

        paths = [p.strip().strip('"').strip("'") for p in arg.split(";") if p.strip()]

        for path_str in paths:
            file_path = Path(path_str)

            transcript = await self.audio_service.transcribe_file(file_path)

            if not transcript:
                continue

            self.console.print(
                f"\n[bold]→ Whisper Transcript for {escape(file_path.name)}:[/bold]\n"
                f"{escape(transcript.text)}\n"
            )

            chunks_count = self.kb_service.add_document(
                text=transcript.text, source=f"voice:{file_path.name}"
            )

            self.console.print(
                f"[green]Successfully added {chunks_count} chunk(s) from "
                f"{escape(file_path.name)} to Knowledge Base![/green]"
            )

        return True

    async def _handle_change_prompt(self, arg: str) -> bool:
        if not arg:
            arg = self.console.input(
                "[bold yellow]Enter new system prompt or file path:[/bold yellow] "
            ).strip()

        if not arg:
            self.console.print("[yellow]No prompt provided. Skipped.[/yellow]")
            return True

        new_prompt = arg

        if os.path.isfile(arg):
            try:
                new_prompt = await asyncio.to_thread(
                    Path(arg).read_text, encoding="utf-8"
                )
            except OSError as e:
                logger.exception("could not read prompt file %s", arg)
                self.console.print(f"[red]Could not read file: {escape(str(e))}[/red]")
                return True

        self.session.system_prompt = new_prompt
        self.console.print("[green]System prompt successfully updated![/green]")

        return True

    async def _handle_search(self, arg: str) -> bool:
        if not arg:
            arg = self.console.input(
                "[bold yellow]What do you want to search for?:[/bold yellow] "
            ).strip()
        if not arg:
            self.console.print("[yellow]Search cancelled.[/yellow]")
            return True

        await self.session.send_message(arg, force_tool="semantic_search")
        return True

    async def _handle_summarize(self, _arg: str) -> bool:
        await self.session.send_message(
            "Summarize our conversation so far.", force_tool="summarize_session"
        )
        return True

    async def _handle_save_session(self, arg: str) -> bool:
        if not arg:
            arg = self.console.input(
                "[bold yellow]Enter filename to save session (e.g., chat.json):[/bold yellow] "
            ).strip()

        if not arg:
            self.console.print("[yellow]No filename provided. Skipped.[/yellow]")
            return True

        try:
            if not arg.endswith(".json"):
                arg += ".json"

            self.session.save_session(arg)
            self.console.print(
                f"[green]Session successfully saved to {escape(arg)}[/green]"
            )
        except (OSError, TypeError, ValueError) as e:
            logger.exception("save_session failed")
            self.console.print(f"[red]Failed to save session: {escape(str(e))}[/red]")

        return True

    async def _handle_load_session(self, arg: str) -> bool:
        if not arg:
            arg = self.console.input(
                "[bold yellow]Enter filename to load session from:[/bold yellow] "
            ).strip()

        if not arg:
            self.console.print("[yellow]No filename provided. Skipped.[/yellow]")
            return True

        try:
            if not arg.endswith(".json") and not Path(arg).exists():
                arg += ".json"

            self.session.load_session(arg)
            self.console.print(
                f"[green]Session successfully loaded from {escape(arg)}[/green]"
            )
        except (OSError, ValueError, TypeError) as e:
            logger.exception("load_session failed")
            self.console.print(f"[red]Failed to load session: {escape(str(e))}[/red]")

        return True

    async def _handle_retry(self, _arg: str) -> bool:
        last_user_idx = None
        for i in range(len(self.session.messages) - 1, -1, -1):
            if self.session.messages[i].role == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            self.console.print("[yellow]Could not find your last prompt.[/yellow]")
            return True

        prompt = self.session.messages[last_user_idx].content

        if not prompt:
            self.console.print(
                "[yellow]Last prompt text is empty. Nothing to retry.[/yellow]"
            )
            return True

        self.session.messages = self.session.messages[:last_user_idx]

        self.console.print(f"[dim]Retrying: {escape(prompt)}[/dim]")

        await self.session.send_message(prompt)

        return True
