from typing import Any, Dict, List

from agents.tools.base import BaseTool


class ToolRegistry:

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool using its name."""

        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered."
            )

        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> BaseTool:
        """Get a tool by name."""

        tool = self._tools.get(tool_name)

        if tool is None:
            raise ValueError(
                f"Tool '{tool_name}' not found."
            )

        return tool

    def list_tools(self) -> List[str]:
        """Return names of all registered tools."""

        return list(self._tools.keys())

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return tool definitions for the LLM."""

        return [
            tool.get_schema()
            for tool in self._tools.values()
        ]