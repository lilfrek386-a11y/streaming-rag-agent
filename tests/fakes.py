from types import SimpleNamespace


def text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))],
        usage=None,
    )


def tool_chunk(
    call_id: str, name: str, arguments: str, index: int = 0
) -> SimpleNamespace:
    delta_call = SimpleNamespace(
        id=call_id,
        index=index,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[delta_call])
            )
        ],
        usage=None,
    )


class FakeStream:
    def __init__(self, items, total_tokens: int = 5):
        self._items = items if isinstance(items, list) else [items]
        self._total_tokens = total_tokens

    async def __aiter__(self):
        for item in self._items:
            if isinstance(item, str):
                yield text_chunk(item)
            else:
                yield item
        yield SimpleNamespace(
            choices=[], usage=SimpleNamespace(total_tokens=self._total_tokens)
        )


class FakeAPIClient:
    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    async def create_chat_completion(
        self, messages, model, stream=True, tools=None, tool_choice=None
    ):
        self.calls.append(
            {
                "messages": messages,
                "stream": stream,
                "tool_choice": tool_choice,
                "tools": tools,
            }
        )

        if not self.script:
            raise AssertionError("FakeAPIClient ran out of scripted responses")

        item = self.script.pop(0)

        if isinstance(item, Exception):
            raise item
        return FakeStream(item)


class FakeVectorStore:
    def __init__(self, results: list[dict] | None = None):
        self._results = results or []

    def search(self, query: str, top_n: int = 3, **_kwargs) -> list[dict]:
        return self._results[:top_n]
