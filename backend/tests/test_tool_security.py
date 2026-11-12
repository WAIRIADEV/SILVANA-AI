
import pytest

from backend.app.intelligence.tool_registry import ToolDefinition
from backend.app.intelligence.tool_security import (
    ToolArgumentError,
    ToolPermissionError,
    ToolSecurityManager,
)


@pytest.fixture
def security():
    return ToolSecurityManager()


@pytest.fixture
def sample_tool():
    return ToolDefinition(
        name="calculate",
        description="Performs a calculation.",
        handler=lambda a, b: a + b,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )


# -----------------------------------------
# ARGUMENT VALIDATION
# -----------------------------------------

def test_valid_arguments_are_accepted(security, sample_tool):
    arguments = {"a": 10, "b": 20}

    result = security.validate_arguments(
        sample_tool,
        arguments,
    )

    assert result == arguments


def test_non_dictionary_arguments_are_rejected(
    security,
    sample_tool,
):
    with pytest.raises(ToolArgumentError):
        security.validate_arguments(
            sample_tool,
            ["not", "a", "dictionary"],
        )


def test_missing_required_argument_is_rejected(
    security,
    sample_tool,
):
    with pytest.raises(ToolArgumentError, match="Missing required"):
        security.validate_arguments(
            sample_tool,
            {"a": 10},
        )


def test_incorrect_argument_type_is_rejected(
    security,
    sample_tool,
):
    with pytest.raises(ToolArgumentError, match="Invalid type"):
        security.validate_arguments(
            sample_tool,
            {"a": "ten", "b": 20},
        )


def test_boolean_is_not_accepted_as_number(
    security,
    sample_tool,
):
    with pytest.raises(ToolArgumentError, match="Invalid type"):
        security.validate_arguments(
            sample_tool,
            {"a": True, "b": 20},
        )


def test_unexpected_arguments_are_rejected(
    security,
    sample_tool,
):
    with pytest.raises(ToolArgumentError, match="Unexpected arguments"):
        security.validate_arguments(
            sample_tool,
            {"a": 10, "b": 20, "admin": True},
        )


def test_numeric_minimum_is_enforced(security):
    tool = ToolDefinition(
        name="set_volume",
        description="Sets the volume.",
        handler=lambda volume: volume,
        parameters={
            "type": "object",
            "properties": {
                "volume": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                }
            },
            "required": ["volume"],
            "additionalProperties": False,
        },
    )

    with pytest.raises(ToolArgumentError, match="below the minimum"):
        security.validate_arguments(
            tool,
            {"volume": -1},
        )


def test_numeric_maximum_is_enforced(security):
    tool = ToolDefinition(
        name="set_volume",
        description="Sets the volume.",
        handler=lambda volume: volume,
        parameters={
            "type": "object",
            "properties": {
                "volume": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                }
            },
            "required": ["volume"],
            "additionalProperties": False,
        },
    )

    with pytest.raises(ToolArgumentError, match="exceeds the maximum"):
        security.validate_arguments(
            tool,
            {"volume": 101},
        )


def test_string_length_is_enforced(security):
    tool = ToolDefinition(
        name="create_folder",
        description="Creates a folder.",
        handler=lambda name: name,
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 20,
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    )

    with pytest.raises(ToolArgumentError, match="too short"):
        security.validate_arguments(
            tool,
            {"name": "AB"},
        )


def test_enum_restrictions_are_enforced(security):
    tool = ToolDefinition(
        name="set_mode",
        description="Changes operating mode.",
        handler=lambda mode: mode,
        parameters={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["safe", "normal"],
                }
            },
            "required": ["mode"],
            "additionalProperties": False,
        },
    )

    with pytest.raises(ToolArgumentError, match="Invalid value"):
        security.validate_arguments(
            tool,
            {"mode": "dangerous"},
        )


def test_nested_objects_are_validated(security):
    tool = ToolDefinition(
        name="configure",
        description="Accepts nested configuration.",
        handler=lambda config: config,
        parameters={
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                    },
                    "required": ["enabled"],
                    "additionalProperties": False,
                }
            },
            "required": ["config"],
            "additionalProperties": False,
        },
    )

    with pytest.raises(ToolArgumentError, match="Invalid type"):
        security.validate_arguments(
            tool,
            {"config": {"enabled": "yes"}},
        )


def test_array_items_are_validated(security):
    tool = ToolDefinition(
        name="process_numbers",
        description="Processes a list of numbers.",
        handler=lambda numbers: numbers,
        parameters={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "maxItems": 5,
                }
            },
            "required": ["numbers"],
            "additionalProperties": False,
        },
    )

    with pytest.raises(ToolArgumentError, match="Invalid type"):
        security.validate_arguments(
            tool,
            {"numbers": [1, "two", 3]},
        )


def test_unknown_schema_type_is_rejected(security):
    tool = ToolDefinition(
        name="invalid_schema",
        description="Uses an unsupported schema type.",
        handler=lambda: None,
        parameters={
            "type": "unsupported",
        },
    )

    with pytest.raises(ToolArgumentError, match="Unsupported"):
        security.validate_arguments(
            tool,
            {},
        )


# -----------------------------------------
# PERMISSION CONTROLS
# -----------------------------------------

def test_read_permission_is_granted_by_default(
    security,
):
    tool = ToolDefinition(
        name="read_document",
        description="Reads a document.",
        handler=lambda: None,
        permission_level="read",
    )

    security.authorize(tool)

    assert True


def test_write_permission_is_denied_by_default(
    security,
):
    tool = ToolDefinition(
        name="create_folder",
        description="Creates a folder.",
        handler=lambda: None,
        permission_level="write",
    )

    with pytest.raises(ToolPermissionError, match="has not been granted"):
        security.authorize(tool)


def test_granted_write_permission_is_accepted():
    security = ToolSecurityManager(
        granted_permissions={"read", "write"}
    )

    tool = ToolDefinition(
        name="create_folder",
        description="Creates a folder.",
        handler=lambda: None,
        permission_level="write",
    )

    security.authorize(tool)

    assert True


def test_destructive_operation_requires_confirmation():
    security = ToolSecurityManager(
        granted_permissions={"read", "destructive"}
    )

    tool = ToolDefinition(
        name="delete_file",
        description="Deletes a file.",
        handler=lambda: None,
        permission_level="destructive",
    )

    with pytest.raises(ToolPermissionError, match="confirmation"):
        security.authorize(tool)


def test_confirmed_destructive_operation_is_accepted():
    security = ToolSecurityManager(
        granted_permissions={"read", "destructive"}
    )

    tool = ToolDefinition(
        name="delete_file",
        description="Deletes a file.",
        handler=lambda: None,
        permission_level="destructive",
    )

    security.authorize(
        tool,
        user_confirmed=True,
    )


def test_explicit_confirmation_is_required_when_configured(
    security,
):
    tool = ToolDefinition(
        name="send_email",
        description="Sends an email.",
        handler=lambda: None,
        permission_level="read",
        requires_confirmation=True,
    )

    with pytest.raises(ToolPermissionError, match="confirmation"):
        security.authorize(tool)


def test_disabled_tool_cannot_be_authorized(security):
    tool = ToolDefinition(
        name="disabled_tool",
        description="A disabled capability.",
        handler=lambda: None,
        enabled=False,
    )

    with pytest.raises(ToolPermissionError, match="disabled"):
        security.authorize(tool)


def test_system_permission_requires_confirmation():
    security = ToolSecurityManager(
        granted_permissions={"read", "system"}
    )

    tool = ToolDefinition(
        name="system_configuration",
        description="Changes system configuration.",
        handler=lambda: None,
        permission_level="system",
    )

    with pytest.raises(ToolPermissionError, match="confirmation"):
        security.authorize(tool)


def test_invalid_permission_level_is_rejected():
    with pytest.raises(ValueError, match="Invalid tool permission"):
        ToolDefinition(
            name="invalid_tool",
            description="Invalid permission.",
            handler=lambda: None,
            permission_level="root",
        )


def test_invalid_granted_permission_is_rejected():
    with pytest.raises(ValueError, match="Invalid permissions"):
        ToolSecurityManager(
            granted_permissions={"read", "superuser"}
        )
