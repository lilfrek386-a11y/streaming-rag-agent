import argparse
import asyncio

from openai import AuthenticationError
from rich.console import Console

from src.chat_session import ChatSession
from src.core.client import create_client, APIClient
from src.core.constants import QUIT_COMMANDS
from src.core.settings import settings

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-turn CLI chat with streaming and token tracking.")
    parser.add_argument(
        "-prompt",
        type=str,
        default=None,
        help="Custom system prompt (e.g. 'You are a math tutor')",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    api_client = APIClient(create_client())
    session = ChatSession(api_client, model_name=settings.groq.model, system_prompt=args.prompt)

    if args.prompt:
        console.print(f"> System Prompt: {args.prompt}", style="bold magenta")

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