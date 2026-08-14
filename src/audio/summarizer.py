from src.core.client import APIClient
from src.core.constants import SUMMARY_PROMPTS


async def summarize(
    api_client: APIClient, transcript: str, mode: str = "summary"
) -> str:
    prompt = SUMMARY_PROMPTS.get(mode, SUMMARY_PROMPTS["summary"])

    response = await api_client.create_chat_completion(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
        model="llama-3.3-70b-versatile",
        stream=False,
    )

    return response.choices[0].message.content or ""
