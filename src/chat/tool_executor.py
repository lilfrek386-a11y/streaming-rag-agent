import asyncio
import json
import logging
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(
        self,
        tools_map: dict[str, Callable],
        tool_models: dict[str, type[BaseModel]],
    ):
        self._tools_map = tools_map
        self._tool_models = tool_models

    async def execute(self, name: str, arguments: str) -> str:
        func = self._tools_map.get(name)
        if not func:
            logger.warning("model requested unknown tool %s", name)
            return json.dumps({"error": f"Unknown tool: {name}"})

        model_cls = self._tool_models.get(name)
        try:
            if model_cls:
                parsed = model_cls.model_validate_json(arguments)
                args = parsed.model_dump()
            else:
                args = json.loads(arguments)
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning("tool %s validation failed: %s", name, e)
            return json.dumps({"error": f"Invalid arguments for {name}: {e!s}"})

        try:
            result = await asyncio.to_thread(func, **args)

            if result is None:
                result = json.dumps({"result": None})
            elif not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)

            return result
        except Exception as e:  # noqa: BLE001
            logger.exception("tool %s failed", name)
            return json.dumps({"error": f"Error executing tool {name}: {e!s}"})
