import pytest
from pydantic import ValidationError

from src.schemas import ChatMessage, ToolCall, ToolCallFunction


def _tool_call() -> ToolCall:
    return ToolCall(
        id="call_1",
        type="function",
        function=ToolCallFunction(name="semantic_search", arguments="{}"),
    )


def test_known_roles_are_accepted():
    for role in ("system", "user", "assistant"):
        assert ChatMessage(role=role, content="hi").role == role


def test_unknown_role_is_rejected():
    with pytest.raises(ValidationError):
        ChatMessage(role="asistant", content="hi")


def test_tool_message_requires_tool_call_id():
    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content="{}")


def test_tool_call_id_rejected_on_non_tool_message():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="hi", tool_call_id="call_1")


def test_tool_calls_only_on_assistant():
    assert ChatMessage(role="assistant", content="", tool_calls=[_tool_call()])

    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="hi", tool_calls=[_tool_call()])


def test_exclude_none_keeps_payload_minimal():
    dumped = ChatMessage(role="user", content="hi").model_dump(exclude_none=True)
    assert dumped == {"role": "user", "content": "hi"}


def test_semantic_search_params_validation():
    from src.schemas import SemanticSearchParams

    params = SemanticSearchParams(query="hello", top_n=5)
    assert params.query == "hello"
    assert params.top_n == 5

    with pytest.raises(ValidationError):
        SemanticSearchParams(query="hello", top_n=0)
    with pytest.raises(ValidationError):
        SemanticSearchParams(query="hello", top_n=11)
