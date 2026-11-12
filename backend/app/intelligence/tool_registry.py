
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal


ToolHandler = Callable[..., Any | Awaitable[Any]]

PermissionLevel = Literal[
    "read",
    "write",
    "destructive",
    "system",
]


@dataclass
class ToolDefinition:
    """
    Describes a capability available to SILVANA.
    """

    name: str
    description: str
    handler: ToolHandler

    parameters: dict[str, Any] = field(default_factory=dict)

    requires_confirmation: bool = False
    enabled: bool = True
    permission_level: PermissionLevel = "read"

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Tool name cannot be empty.")

        if not self.description.strip():
            raise ValueError("Tool description cannot be empty.")

        if not callable(self.handler):
            raise ValueError("Tool handler must be callable.")

        valid_permissions = {
            "read",
            "write",
            "destructive",
            "system",
        }

        if self.permission_level not in valid_permissions:
            raise ValueError(
                "Invalid tool permission level."
            )


class ToolRegistry:
    """
    Central directory for SILVANA's registered tools.
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool."""

        if not isinstance(tool, ToolDefinition):
            raise TypeError(
                "Only ToolDefinition instances can be registered."
            )

        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered."
            )

        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        """Retrieve an enabled tool by name."""

        tool = self._tools.get(name)

        if tool is None or not tool.enabled:
            raise KeyError(
                f"Tool '{name}' is not available."
            )

        return tool

    def list_tools(self) -> list[ToolDefinition]:
        """Return all enabled tools."""

        return [
            tool
            for tool in self._tools.values()
            if tool.enabled
        ]

    def unregister(self, name: str) -> None:
        """Remove a registered tool."""

        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' is not registered."
            )

        del self._tools[name]
