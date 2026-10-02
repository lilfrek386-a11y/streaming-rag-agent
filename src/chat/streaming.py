from collections.abc import Callable

from src.schemas.chat import StreamResult


class StreamProcessor:
    def __init__(
        self,
        on_content_chunk: Callable[[str, bool], None] | None = None,
    ):
        self._on_content_chunk = on_content_chunk

    async def process(self, response) -> StreamResult:
        collected: list[str] = []
        tool_calls_acc: dict[int, dict[str, str]] = {}
        is_first = True
        usage = None

        async for chunk in response:
            if getattr(chunk, "usage", None):
                usage = chunk.usage

            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta

            content = getattr(delta, "content", None)
            if content:
                collected.append(content)
                if self._on_content_chunk:
                    self._on_content_chunk(content, is_first)
                is_first = False

            tc_list = getattr(delta, "tool_calls", None)
            if tc_list:
                self._accumulate(tc_list, tool_calls_acc)

        ordered = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
        return StreamResult(
            content="".join(collected),
            tool_calls=ordered,
            usage=usage,
        )

    @staticmethod
    def _accumulate(tc_list, acc: dict[int, dict[str, str]]) -> None:
        for tc in tc_list:
            idx = getattr(tc, "index", 0)
            if idx not in acc:
                acc[idx] = {"id": "", "name": "", "arguments": ""}
            if getattr(tc, "id", None):
                acc[idx]["id"] = tc.id
            func = getattr(tc, "function", None)
            if func:
                if getattr(func, "name", None):
                    acc[idx]["name"] += func.name
                if getattr(func, "arguments", None):
                    acc[idx]["arguments"] += func.arguments
