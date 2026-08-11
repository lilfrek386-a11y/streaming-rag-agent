from pydantic import BaseModel


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str
    function: ToolCallFunction
