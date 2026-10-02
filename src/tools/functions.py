from collections.abc import Callable
from typing import Any


def create_tool_functions(
    vector_store: Any, get_messages_callback: Callable[[], list]
) -> dict[str, Callable]:
    def semantic_search(query: str, top_n: int = 3) -> dict[str, Any]:
        results = vector_store.search(query, top_n=top_n)
        return {"result": results if results else "Nothing."}

    def summarize_session() -> dict[str, Any]:
        messages = get_messages_callback()

        raw_history = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role in ("user", "assistant") and msg.content
        ]
        return {"raw_history": raw_history}

    return {
        "semantic_search": semantic_search,
        "summarize_session": summarize_session,
    }
