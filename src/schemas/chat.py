from typing import Literal, Self

from pydantic import BaseModel, model_validator

Role = Literal["system", "user", "assistant", "tool"]


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def check_role_shape(self) -> Self:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("a 'tool' message requires tool_call_id")
        if self.role != "tool" and self.tool_call_id:
            raise ValueError("tool_call_id is only valid on a 'tool' message")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("tool_calls are only valid on an 'assistant' message")
        return self


class StreamResult(BaseModel):
    content: str = ""
    tool_calls: list[dict[str, str]] = []
    usage: object | None = None

    model_config = {"arbitrary_types_allowed": True}
