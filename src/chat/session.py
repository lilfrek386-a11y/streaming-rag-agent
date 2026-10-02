import json
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markup import escape

from src.chat.streaming import StreamProcessor
from src.chat.tool_executor import ToolExecutor
from src.core.client import APIClient
from src.core.constants import MAX_ROUNDS
from src.core.prompts import DEFAULT_SYSTEM_PROMPT
from src.core.settings import settings
from src.schemas.chat import ChatMessage, StreamResult, ToolCall, ToolCallFunction
from src.schemas.tools import TOOL_MODELS, tools
from src.tools.functions import create_tool_functions

if TYPE_CHECKING:
    from src.audio.service import AudioService

logger = logging.getLogger(__name__)


class ChatSession:
    def __init__(
        self,
        api_client: APIClient,
        model_name: str,
        console: Console,
        tool_executor: ToolExecutor | None = None,
        audio_service: "AudioService | None" = None,
        system_prompt: str | None = None,
        voice_enabled: bool = False,
        vector_store: Any | None = None,
    ):
        self.api = api_client
        self.model_name = model_name
        self._console = console
        self.voice_enabled = voice_enabled
        self._audio_service = audio_service

        self.messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt or DEFAULT_SYSTEM_PROMPT)
        ]
        self.total_tokens = 0
        self.started_at = datetime.now(UTC)

        self._stream_processor = StreamProcessor(
            on_content_chunk=self._on_content_chunk,
        )

        if tool_executor is not None:
            self._tool_executor = tool_executor
        elif vector_store is not None:
            tools_map = create_tool_functions(
                vector_store=vector_store,
                get_messages_callback=lambda: self.messages,
            )
            self._tool_executor = ToolExecutor(tools_map, TOOL_MODELS)
        else:
            self._tool_executor = ToolExecutor({}, TOOL_MODELS)

    @property
    def system_prompt(self) -> str:
        if self.messages and self.messages[0].role == "system":
            return self.messages[0].content or DEFAULT_SYSTEM_PROMPT
        return DEFAULT_SYSTEM_PROMPT

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        if self.messages and self.messages[0].role == "system":
            self.messages[0].content = value
        else:
            self.messages.insert(0, ChatMessage(role="system", content=value))
        logger.info("system prompt updated (%d chars)", len(value))

    async def send_message(self, user_input: str, force_tool: str | None = None) -> str:
        initial_msg_count = len(self.messages)
        start_tokens = self.total_tokens

        self.messages.append(ChatMessage(role="user", content=user_input))
        full_text = ""

        try:
            rounds_exhausted = True

            for step in range(MAX_ROUNDS):
                self._console.print(
                    f"[dim][Agent thinking... Round {step + 1}/{MAX_ROUNDS}][/dim]",
                    end="\r",
                )

                tool_choice = (
                    {"type": "function", "function": {"name": force_tool}}
                    if step == 0 and force_tool
                    else "auto"
                )

                response = await self.api.create_chat_completion(
                    messages=self._get_api_messages(),
                    model=self.model_name,
                    stream=True,
                    tools=tools,
                    tool_choice=tool_choice,
                )

                result = await self._stream_processor.process(response)

                if result.content:
                    print()
                self._clear_status_line()

                round_msg = self._build_round_message(result)
                self._update_tokens(result.usage)

                if not result.tool_calls:
                    rounds_exhausted = False
                    full_text = result.content
                    self.messages.append(round_msg)
                    break

                self.messages.append(round_msg)

                for c in result.tool_calls:
                    name = c["name"]
                    arguments = c["arguments"]
                    logger.info("tool call %s args=%s", name, arguments)
                    self._console.print(
                        f"[dim]  Calling tool: {escape(f'{name}({arguments})')}[/dim]"
                    )

                    output = await self._tool_executor.execute(name, arguments)
                    self.messages.append(
                        ChatMessage(role="tool", tool_call_id=c["id"], content=output)
                    )

            if rounds_exhausted:
                logger.warning("hit MAX_ROUNDS=%d with pending tool calls", MAX_ROUNDS)
                warning_text = (
                    f"[Warning: Hit MAX_ROUNDS={MAX_ROUNDS} with pending tool calls.]"
                )
                self._console.print(f"[yellow]{warning_text}[/yellow]")
                full_text = warning_text

        except Exception as e:  # noqa: BLE001
            self.messages = self.messages[:initial_msg_count]
            logger.exception(
                "turn failed, history rolled back to %d", initial_msg_count
            )
            self._clear_status_line()
            self._console.print(
                f"\n[bold red]Turn failed ({type(e).__name__}): "
                f"{escape(str(e))}[/bold red]"
            )
            raise

        if self.voice_enabled and self._audio_service:
            await self._audio_service.speak(full_text)

        tokens_this_turn = self.total_tokens - start_tokens
        self._console.print(
            f"[dim][Tokens used: {tokens_this_turn} | "
            f"Total so far: {self.total_tokens}][/dim]"
        )

        return full_text

    async def execute_tool(self, name: str, arguments: str) -> str:
        return await self._tool_executor.execute(name, arguments)

    def _on_content_chunk(self, text: str, is_first: bool) -> None:
        if is_first:
            self._clear_status_line()
            self._console.print("[bold green]Assistant:[/bold green] ", end="")
        print(text, end="", flush=True)

    def _get_api_messages(self) -> list[dict]:
        return [m.model_dump(exclude_none=True) for m in self.messages]

    def _clear_status_line(self) -> None:
        print("\r" + " " * 40 + "\r", end="")

    @staticmethod
    def _build_round_message(result: StreamResult) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=result.content or None,
            tool_calls=[
                ToolCall(
                    id=c["id"],
                    type="function",
                    function=ToolCallFunction(
                        name=c["name"],
                        arguments=c["arguments"],
                    ),
                )
                for c in result.tool_calls
            ]
            if result.tool_calls
            else None,
        )

    def _update_tokens(self, usage: object | None) -> None:
        if usage and hasattr(usage, "total_tokens"):
            self.total_tokens += usage.total_tokens


    def save_log(self, path: str | None = None):
        if path is None:
            date_str = datetime.now(UTC).strftime("%Y-%m-%d")
            os.makedirs(settings.chat.log_dir, exist_ok=True)
            path = f"{settings.chat.log_dir}/{date_str}.md"

        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## Session {self.started_at.isoformat()}\n")
            f.write(f"Model: {self.model_name} | Total tokens: {self.total_tokens}\n\n")
            for msg in self.messages:
                content_str = msg.content if msg.content is not None else ""
                f.write(f"**{msg.role}**: {content_str}\\n\\n")

        self._console.print(f"[dim][Conversation saved to {escape(path)}][/dim]")

    def print_summary(self):
        duration = datetime.now(UTC) - self.started_at
        user_turns = sum(1 for m in self.messages if m.role == "user")

        self._console.print("\n[bold cyan]── Session Summary ──[/bold cyan]")
        self._console.print(f"Duration: {str(duration).split('.')[0]}")
        self._console.print(f"User turns: {user_turns}")
        self._console.print(f"Total tokens used: [bold]{self.total_tokens}[/bold]")

    def save_session(self, filename: str) -> None:
        data = [msg.model_dump() for msg in self.messages]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_session(self, filename: str) -> None:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")

        with open(filename, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Session file must contain a list of messages.")

        loaded = [ChatMessage(**msg) for msg in data]

        if not loaded or loaded[0].role != "system":
            loaded.insert(0, ChatMessage(role="system", content=self.system_prompt))

        self.messages = loaded
        logger.info("loaded session from %s (%d messages)", filename, len(loaded))
