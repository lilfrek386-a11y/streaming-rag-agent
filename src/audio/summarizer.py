from rich.console import Console

from src.core.client import APIClient
from src.core.prompts import SUMMARY_PROMPTS, TranscriptionMode

console = Console()


async def summarize(
    api_client: APIClient,
    transcript: str,
    model: str,
    mode: TranscriptionMode = TranscriptionMode.SUMMARY,
) -> str | None:

    prompt = SUMMARY_PROMPTS[mode]

    try:
        response = await api_client.create_chat_completion(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcript},
            ],
            model=model,
            stream=False,
        )
    except Exception as e:
        console.print(f"[red]Summary failed: {e}[/red]")
        return None

    return response.choices[0].message.content or ""
