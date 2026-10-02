from unittest.mock import AsyncMock, MagicMock

import pytest

from src.chat import ChatSession
from src.commands.router import CommandRouter
from src.schemas import ChatMessage, ToolCall, ToolCallFunction
from tests.fakes import FakeAPIClient, FakeVectorStore, tool_chunk


def make_router(console, script, system_prompt="persona"):
    session = ChatSession(
        api_client=FakeAPIClient(script),
        model_name="test-model",
        console=console,
        vector_store=FakeVectorStore(),
        system_prompt=system_prompt,
    )
    kb_service = MagicMock()
    kb_service.add_document.return_value = 1
    audio_service = AsyncMock()
    audio_service.transcribe_file.return_value = None

    router = CommandRouter(
        session=session,
        kb_service=kb_service,
        audio_service=audio_service,
        console=console,
    )
    return router, session


async def test_unknown_command_keeps_the_loop_alive(console):
    router, _ = make_router(console, [])

    assert await router.dispatch("/nope") is True


async def test_exit_stops_the_loop(console):
    router, _ = make_router(console, [])

    assert await router.dispatch("/exit") is False


async def test_change_prompt_actually_changes_the_payload(console):
    router, session = make_router(console, [])

    await router.dispatch("/change_prompt you are a pirate")

    assert session.messages[0].content == "you are a pirate"


async def test_change_prompt_reads_a_file(console, tmp_path):
    prompt_file = tmp_path / "p.txt"
    prompt_file.write_text("from file", encoding="utf-8")
    router, session = make_router(console, [])

    await router.dispatch(f"/change_prompt {prompt_file}")

    assert session.messages[0].content == "from file"


async def test_retry_after_a_tool_turn(console):
    router, session = make_router(console, [["retried"]])
    session.messages += [
        ChatMessage(role="user", content="cats?"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    type="function",
                    function=ToolCallFunction(name="semantic_search", arguments="{}"),
                )
            ],
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="{}"),
        ChatMessage(role="assistant", content="first answer"),
    ]

    await router.dispatch("/retry")

    assert [m.role for m in session.messages] == ["system", "user", "assistant"]
    assert session.messages[1].content == "cats?"
    assert session.messages[-1].content == "retried"


async def test_retry_after_a_plain_turn(console):
    router, session = make_router(console, [["again"]])
    session.messages += [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi"),
    ]

    await router.dispatch("/retry")

    assert session.messages[-1].content == "again"
    assert sum(1 for m in session.messages if m.role == "user") == 1


async def test_retry_with_no_user_message_is_a_no_op(console):
    router, session = make_router(console, [])

    assert await router.dispatch("/retry") is True
    assert [m.role for m in session.messages] == ["system"]


async def test_search_forces_the_semantic_search_tool(console):
    script = [
        [tool_chunk("c1", "semantic_search", '{"query": "cat"}')],
        ["ok"],
    ]
    router, session = make_router(console, script)

    await router.dispatch("/search cats")

    assert session.api.calls[0]["tool_choice"]["function"]["name"] == "semantic_search"


async def test_search_then_retry_still_works(console):
    script = [
        [tool_chunk("c1", "semantic_search", '{"query": "cat"}')],
        ["first"],
        ["second"],
    ]
    router, session = make_router(console, script)

    await router.dispatch("/search cats")
    await router.dispatch("/retry")

    assert session.messages[-1].content == "second"
    assert sum(1 for m in session.messages if m.role == "user") == 1


async def test_update_kb_text_reaches_the_service(console):
    router, _ = make_router(console, [])

    await router.dispatch("/update_kb_text the cat sat on the mat")

    router.kb_service.add_document.assert_called_once_with(
        text="the cat sat on the mat", source="manual_input"
    )


async def test_failing_turn_propagates_so_main_can_guard_it(console):
    router, _ = make_router(console, [RuntimeError("boom")])

    with pytest.raises(RuntimeError):
        await router.dispatch("/search cats")
