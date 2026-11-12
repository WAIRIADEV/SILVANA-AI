
from typing import Any

from backend.app.intelligence.tool_registry import ToolDefinition


class ToolArgumentError(ValueError):
    """Raised when tool arguments fail validation."""


class ToolPermissionError(PermissionError):
    """Raised when a tool operation is not authorized."""


class ToolSecurityManager:
    """
    Validates tool arguments and enforces permission policies.

    Supported schema features:
    - object properties
    - required fields
    - primitive types
    - arrays and nested objects
    - enum
    - string and array length limits
    - numeric minimum and maximum
    - additionalProperties
    """

    _VALID_TYPES = {
        "object",
        "string",
        "integer",
        "number",
        "boolean",
        "array",
    }

    _PERMISSION_LEVELS = {
        "read": 0,
        "write": 1,
        "destructive": 2,
        "system": 3,
    }

    def __init__(
        self,
        granted_permissions: set[str] | None = None,
    ):
        self.granted_permissions = (
            granted_permissions or {"read"}
        )

        invalid_permissions = (
            self.granted_permissions
            - self._PERMISSION_LEVELS.keys()
        )

        if invalid_permissions:
            raise ValueError(
                f"Invalid permissions: {invalid_permissions}"
            )

    def validate_arguments(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate arguments against a tool's parameter schema."""

        if not isinstance(arguments, dict):
            raise ToolArgumentError(
                "Tool arguments must be an object."
            )

        schema = tool.parameters or {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

        self._validate_value(
            value=arguments,
            schema=schema,
            path="arguments",
        )

        return arguments

    def _validate_value(
        self,
        value: Any,
        schema: dict[str, Any],
        path: str,
    ) -> None:
        """Recursively validate a value against its schema."""

        if not isinstance(schema, dict):
            raise ToolArgumentError(
                f"Invalid schema at '{path}'."
            )

        expected_type = schema.get("type")

        if expected_type not in self._VALID_TYPES:
            raise ToolArgumentError(
                f"Unsupported or missing schema type at '{path}'."
            )

        if not self._matches_type(value, expected_type):
            raise ToolArgumentError(
                f"Invalid type at '{path}'. "
                f"Expected {expected_type}."
            )

        if "enum" in schema and value not in schema["enum"]:
            raise ToolArgumentError(
                f"Invalid value at '{path}'."
            )

        if expected_type == "object":
            self._validate_object(value, schema, path)

        elif expected_type == "string":
            self._validate_string(value, schema, path)

        elif expected_type in {"integer", "number"}:
            self._validate_number(value, schema, path)

        elif expected_type == "array":
            self._validate_array(value, schema, path)

    def _matches_type(
        self,
        value: Any,
        expected_type: str,
    ) -> bool:
        """Check Python values without accepting bool as a number."""

        if expected_type == "object":
            return isinstance(value, dict)

        if expected_type == "string":
            return isinstance(value, str)

        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)

        if expected_type == "number":
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )

        if expected_type == "boolean":
            return isinstance(value, bool)

        if expected_type == "array":
            return isinstance(value, list)

        return False

    def _validate_object(
        self,
        value: dict[str, Any],
        schema: dict[str, Any],
        path: str,
    ) -> None:
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        if not isinstance(properties, dict):
            raise ToolArgumentError(
                f"Invalid properties schema at '{path}'."
            )

        if not isinstance(required, list):
            raise ToolArgumentError(
                f"Invalid required fields at '{path}'."
            )

        if not all(isinstance(item, str) for item in required):
            raise ToolArgumentError(
                f"Invalid required field names at '{path}'."
            )

        missing = [
            key for key in required
            if key not in value
        ]

        if missing:
            raise ToolArgumentError(
                f"Missing required arguments at '{path}': "
                f"{', '.join(missing)}"
            )

        additional_allowed = schema.get(
            "additionalProperties",
            False,
        )

        if not isinstance(additional_allowed, bool):
            raise ToolArgumentError(
                f"Invalid additionalProperties setting at '{path}'."
            )

        if not additional_allowed:
            unexpected = set(value) - set(properties)

            if unexpected:
                raise ToolArgumentError(
                    f"Unexpected arguments at '{path}': "
                    f"{', '.join(sorted(unexpected))}"
                )

        for key, item in value.items():
            if key in properties:
                self._validate_value(
                    value=item,
                    schema=properties[key],
                    path=f"{path}.{key}",
                )

    def _validate_string(
        self,
        value: str,
        schema: dict[str, Any],
        path: str,
    ) -> None:
        if len(value) < schema.get("minLength", 0):
            raise ToolArgumentError(
                f"Value at '{path}' is too short."
            )

        if len(value) > schema.get("maxLength", float("inf")):
            raise ToolArgumentError(
                f"Value at '{path}' is too long."
            )

    def _validate_number(
        self,
        value: int | float,
        schema: dict[str, Any],
        path: str,
    ) -> None:
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolArgumentError(
                f"Value at '{path}' is below the minimum."
            )

        if "maximum" in schema and value > schema["maximum"]:
            raise ToolArgumentError(
                f"Value at '{path}' exceeds the maximum."
            )

    def _validate_array(
        self,
        value: list[Any],
        schema: dict[str, Any],
        path: str,
    ) -> None:
        if len(value) < schema.get("minItems", 0):
            raise ToolArgumentError(
                f"Array at '{path}' contains too few items."
            )

        if len(value) > schema.get("maxItems", float("inf")):
            raise ToolArgumentError(
                f"Array at '{path}' contains too many items."
            )

        item_schema = schema.get("items")

        if item_schema is not None:
            for index, item in enumerate(value):
                self._validate_value(
                    value=item,
                    schema=item_schema,
                    path=f"{path}[{index}]",
                )

    def authorize(
        self,
        tool: ToolDefinition,
        user_confirmed: bool = False,
    ) -> None:
        """
        Verify permission and confirmation before execution.

        This method does not execute the tool.
        """

        if not tool.enabled:
            raise ToolPermissionError(
                "This tool is disabled."
            )

        required_permission = tool.permission_level

        if required_permission not in self.granted_permissions:
            raise ToolPermissionError(
                f"Permission '{required_permission}' "
                "has not been granted."
            )

        confirmation_required = (
            tool.requires_confirmation
            or required_permission in {"destructive", "system"}
        )

        if confirmation_required and not user_confirmed:
            raise ToolPermissionError(
                "Explicit user confirmation is required."
            )
