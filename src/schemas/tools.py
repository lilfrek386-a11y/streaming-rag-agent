from pydantic import BaseModel, Field


class SemanticSearchParams(BaseModel):
    query: str = Field(..., description="The search question")
    top_n: int = Field(
        default=3,
        ge=1,
        le=10,
        description="How many snippets to return (default 3)",
    )


class SummarizeSessionParams(BaseModel):
    pass


TOOL_MODELS: dict[str, type[BaseModel]] = {
    "semantic_search": SemanticSearchParams,
    "summarize_session": SummarizeSessionParams,
}

semantic_search_tool = {
    "type": "function",
    "function": {
        "name": "semantic_search",
        "description": (
            "Searches the user's vector knowledge base and returns the most "
            "relevant snippets, each with its source and similarity score."
        ),
        "parameters": SemanticSearchParams.model_json_schema(),
    },
}

summarize_session_tool = {
    "type": "function",
    "function": {
        "name": "summarize_session",
        "description": "Summarizes the current chat session",
        "parameters": SummarizeSessionParams.model_json_schema(),
    },
}

tools = [semantic_search_tool, summarize_session_tool]
