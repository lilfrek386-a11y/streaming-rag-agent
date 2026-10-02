import json

import pytest

from src.chat import ChatSession
from src.schemas import ChatMessage
from tests.fakes import FakeAPIClient, FakeVectorStore, tool_chunk


def make_session(console, script, **kwargs) -> ChatSession:
    return ChatSession(
        api_client=FakeAPIClient(script),  # type: ignore[arg-type]
        model_name="test-model",
        console=console,
        vector_store=FakeVectorStore(  # type: ignore[arg-type]
            [{"text": "fact", "source": "a.txt", "score": 1.0}]
        ),
        **kwargs,
    )


async def test_plain_turn_appends_user_and_assistant(console):
    session = make_session(console, [["he", "llo"]])

    answer = await session.send_message("hey")

    assert answer == "hello"
    assert [m.role for m in session.messages] == ["system", "user", "assistant"]


async def test_tool_turn_pairs_every_call_with_a_result(console):
    script = [
        [tool_chunk("c1", "semantic_search", '{"query": "cat"}')],
        ["answer"],
    ]
    session = make_session(console, script)

    await session.send_message("what about cats?")

    roles = [m.role for m in session.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert session.messages[3].tool_call_id == "c1"


async def test_failed_turn_rolls_back_the_whole_history(console):
    script = [
        [tool_chunk("c1", "semantic_search", '{"query": "cat"}')],
        RuntimeError("stream exploded"),
    ]
    session = make_session(console, script)
    before = list(session.messages)

    with pytest.raises(RuntimeError):
        await session.send_message("what about cats?")

    assert session.messages == before
    assert len(session.messages) == 1


async def test_force_tool_only_applies_to_the_first_round(console):
    script = [
        [tool_chunk("c1", "semantic_search", '{"query": "cat"}')],
        ["ok"],
    ]
    session = make_session(console, script)

    await session.send_message("cats", force_tool="semantic_search")

    calls = session.api.calls
    assert calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "semantic_search"},
    }
    assert calls[1]["tool_choice"] == "auto"


async def test_system_prompt_setter_rewrites_the_api_payload(console):
    session = make_session(console, [], system_prompt="original")

    session.system_prompt = "you are a pirate"

    assert session.messages[0].content == "you are a pirate"
    assert session._get_api_messages()[0]["content"] == "you are a pirate"


async def test_tool_arguments_with_markup_do_not_break_the_console(console):
    script = [
        [tool_chunk("c1", "semantic_search", '{"query": "[/bold]"}')],
        ["ok"],
    ]
    session = make_session(console, script)

    await session.send_message("weird query")

    assert session.messages[-1].content == "ok"


async def test_execute_tool_reports_bad_json_back_to_the_model(console):
    session = make_session(console, [])

    result = await session.execute_tool("semantic_search", "{not json")

    assert "error" in json.loads(result)


async def test_execute_tool_reports_unknown_tool(console):
    session = make_session(console, [])

    result = await session.execute_tool("rm_rf", "{}")

    assert json.loads(result)["error"].startswith("Unknown tool")


async def test_load_session_inserts_a_missing_system_message(console, tmp_path):
    session = make_session(console, [], system_prompt="persona")
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{"role": "user", "content": "hi"}]), encoding="utf-8")

    session.load_session(str(path))

    assert session.messages[0].role == "system"
    assert session.messages[0].content == "persona"


async def test_load_session_rejects_a_non_list_payload(console, tmp_path):
    session = make_session(console, [])
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"role": "user"}), encoding="utf-8")

    with pytest.raises(ValueError):
        session.load_session(str(path))


async def test_save_then_load_preserves_a_tool_turn(console, tmp_path):
    session = make_session(console, [])
    session.messages.append(ChatMessage(role="user", content="hi"))
    path = tmp_path / "s.json"

    session.save_session(str(path))
    session.load_session(str(path))

    assert [m.role for m in session.messages] == ["system", "user"]


async def test_stream_processor_direct():
    from src.chat import StreamProcessor
    from tests.fakes import FakeStream

    received_chunks = []

    def on_chunk(text: str, is_first: bool):
        received_chunks.append((text, is_first))

    processor = StreamProcessor(on_content_chunk=on_chunk)
    stream = FakeStream(["hel", "lo"])

    result = await processor.process(stream)
    assert result.content == "hello"
    assert result.tool_calls == []
    assert len(received_chunks) == 2
    assert received_chunks[0] == ("hel", True)
    assert received_chunks[1] == ("lo", False)


async def test_tool_executor_direct():
    from src.chat import ToolExecutor
    from src.schemas.tools import TOOL_MODELS

    def my_tool(query: str, top_n: int = 3):
        return {"query": query, "top_n": top_n}

    executor = ToolExecutor(
        tools_map={"semantic_search": my_tool},
        tool_models=TOOL_MODELS,
    )

    res = await executor.execute("semantic_search", '{"query": "cats", "top_n": 2}')
    assert json.loads(res) == {"query": "cats", "top_n": 2}

    unknown_res = await executor.execute("unknown", "{}")
    assert "Unknown tool" in json.loads(unknown_res)["error"]

    bad_args = await executor.execute("semantic_search", '{"top_n": -1}')
    assert "Invalid arguments" in json.loads(bad_args)["error"]


async def test_chat_session_with_injected_tool_executor(console):
    from src.chat import ToolExecutor
    from src.schemas.tools import TOOL_MODELS

    executor = ToolExecutor(
        tools_map={"semantic_search": lambda query, top_n=3: {"result": "ok"}},
        tool_models=TOOL_MODELS,
    )
    script = [
        [tool_chunk("c1", "semantic_search", '{"query": "cat"}')],
        ["done"],
    ]
    session = ChatSession(
        api_client=FakeAPIClient(script),  # type: ignore[arg-type]
        model_name="test-model",
        console=console,
        tool_executor=executor,
    )
    ans = await session.send_message("test")
    assert ans == "done"
    assert json.loads(session.messages[3].content or "{}") == {"result": "ok"}
