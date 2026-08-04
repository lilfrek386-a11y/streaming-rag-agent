import os
from datetime import datetime

import tiktoken
from openai import (
    AuthenticationError,
    RateLimitError,
    BadRequestError,
    InternalServerError,
    APITimeoutError,
    APIConnectionError
)
from rich.console import Console

from src.core.constants import FALLBACK_ENCODING
from src.core.settings import settings
from src.core.client import APIClient

console = Console()


class ChatSession:
    def __init__(self, api_client: APIClient, model_name: str, system_prompt: str | None = None):
        self.api = api_client
        self.model_name = model_name
        self.messages = [
            {"role": "system", "content": system_prompt or settings.chat.default_system_prompt}
        ]
        self.total_tokens = 0
        self.started_at = datetime.now()

    async def send_message(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        console.print("[dim][Assistant is typing...][/dim]", end="\r")

        try:
            stream = await self.api.create_chat_completion(
                messages=self.messages,
                model=self.model_name,
                stream=True
            )
        except AuthenticationError:
            console.print("Error: Invalid API key. Check your .env file.", style="bold red")
            self.messages.pop()
            raise
        except BadRequestError as e:
            console.print(f"Error: Bad request — {e}. Check your input.", style="red")
            self.messages.pop()
            raise
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
            console.print(f"Error: {type(e).__name__} after retries — {e}", style="red")
            self.messages.pop()
            raise
        except Exception as e:
            console.print(f"Unexpected error: {e}", style="bold red")
            self.messages.pop()
            raise

        collected = []
        usage = None
        first_chunk = True

        async for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content

                if content:
                    if first_chunk:
                        print("\r" + " " * 40 + "\r", end="")
                        console.print("[bold green]Assistant:[/bold green] ", end="")
                        first_chunk = False

                    print(content, end="", flush=True)
                    collected.append(content)

            if chunk.usage:
                usage = chunk.usage

        print()

        full_text = "".join(collected)
        self.messages.append({"role": "assistant", "content": full_text})

        if usage:
            tokens_this_turn = usage.total_tokens
        else:
            encoding = tiktoken.get_encoding(FALLBACK_ENCODING)
            tokens_this_turn = len(encoding.encode(user_input)) + len(encoding.encode(full_text))

        self.total_tokens += tokens_this_turn
        console.print(f"[dim][Tokens used: {tokens_this_turn} | Total so far: {self.total_tokens}][/dim]")

        return full_text

    def save_log(self, path: str | None = None):
        if path is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            os.makedirs(settings.chat.log_dir, exist_ok=True)
            path = f"{settings.chat.log_dir}/{date_str}.md"

        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## Session {self.started_at.isoformat()}\n")
            f.write(f"Model: {self.model_name} | Total tokens: {self.total_tokens}\n\n")
            for msg in self.messages:
                f.write(f"**{msg['role']}**: {msg['content']}\n\n")

        print(f"[Conversation saved to {path}]")

    def print_summary(self):
        duration = datetime.now() - self.started_at
        user_turns = sum(1 for m in self.messages if m["role"] == "user")

        console.print("\n[bold cyan]── Session Summary ──[/bold cyan]")
        console.print(f"Duration: {str(duration).split('.')[0]}")
        console.print(f"Messages exchanged: {user_turns}")
        console.print(f"Total tokens used: [bold]{self.total_tokens}[/bold]")