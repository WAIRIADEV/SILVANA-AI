
import pytest

from backend.app.intelligence.capability_resolver import CapabilityMatch
from backend.app.intelligence.intent_engine import Intent, IntentResult
from backend.app.intelligence.tool_planner import (
    ToolPlanStatus,
    ToolPlanner,
)


@pytest.fixture
def planner():
    return ToolPlanner()


@pytest.fixture
def available_tools():
    return [
        CapabilityMatch(
            tool_name="list_directory",
            description="List directory contents.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_entries": {"type": "integer"},
                },
                "required": [],
            },
            permission_level="read",
            requires_confirmation=False,
        ),
        CapabilityMatch(
            tool_name="search_files",
            description="Search for files.",
            parameters={
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "pattern": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                    "max_depth": {"type": "integer"},
                },
                "required": [],
            },
            permission_level="read",
            requires_confirmation=False,
        ),
        CapabilityMatch(
            tool_name="get_file_metadata",
            description="Inspect file metadata.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
            permission_level="read",
            requires_confirmation=False,
        ),
        CapabilityMatch(
            tool_name="check_file_exists",
            description="Check whether a file exists.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
            permission_level="read",
            requires_confirmation=False,
        ),
    ]


def make_intent(rule):
    return IntentResult(
        intent=Intent.TOOL_ACTION,
        confidence=0.95,
        matched_rule=rule,
        requires_confirmation=False,
        requires_tool=True,
    )


def test_search_python_files(planner, available_tools):
    result = planner.plan(
        make_intent("filesystem_search"),
        "Find all Python files",
        available_tools,
    )

    assert result.status == ToolPlanStatus.READY
    assert result.tool_name == "search_files"
    assert result.arguments["pattern"] == "*.py"


def test_search_pdf_files(planner, available_tools):
    result = planner.plan(
        make_intent("filesystem_search"),
        "Search for PDF files",
        available_tools,
    )

    assert result.status == ToolPlanStatus.READY
    assert result.tool_name == "search_files"
    assert result.arguments["pattern"] == "*.pdf"


def test_search_files_in_specific_directory(planner, available_tools):
    result = planner.plan(
        make_intent("filesystem_search"),
        "Find all Python files in src",
        available_tools,
    )

    assert result.status == ToolPlanStatus.READY
    assert result.tool_name == "search_files"
    assert result.arguments["pattern"] == "*.py"
    assert result.arguments["directory"] == "src"


def test_search_explicit_filename(planner, available_tools):
    result = planner.plan(
        make_intent("filesystem_search"),
        "Find settings.json",
        available_tools,
    )

    assert result.status == ToolPlanStatus.READY
    assert result.tool_name == "search_files"
    assert result.arguments["pattern"] == "settings.json"


def test_list_workspace_files(planner, available_tools):
    result = planner.plan(
        make_intent("filesystem_search"),
        "List workspace files",
        available_tools,
    )

    assert result.status == ToolPlanStatus.READY
    assert result.tool_name == "list_directory"
    assert result.arguments["path"] == "."


def test_list_specific_directory(planner, available_tools):
    result = planner.plan(
        make_intent("filesystem_search"),
        "Show folder contents in src",
        available_tools,
    )

    assert result.status == ToolPlanStatus.READY
    assert result.tool_name == "list_directory"
    assert result.arguments["path"] == "src"


def test_check_file_existence(planner, available_tools):
    result = planner.plan(
        make_intent("filesystem_existence"),
        "Does settings.json exist?",
        available_tools,
    )

    assert result.status == ToolPlanStatus.READY
    assert result.tool_name == "check_file_exists"
    assert result.arguments["path"] == "settings.json"


def test_missing_existence_target_requires_clarification(
    planner,
    available_tools,
):
    result = planner.plan(
        make_intent("filesystem_existence"),
        "Does the file exist?",
        available_tools,
    )

    assert result.status == ToolPlanStatus.CLARIFICATION_REQUIRED
    assert result.tool_name is None


def test_inspect_file_metadata(planner, available_tools):
    result = planner.plan(
        make_intent("filesystem_metadata"),
        "Inspect metadata for report.pdf",
        available_tools,
    )

    assert result.status == ToolPlanStatus.READY
    assert result.tool_name == "get_file_metadata"
    assert result.arguments["path"] == "report.pdf"


def test_missing_metadata_target_requires_clarification(
    planner,
    available_tools,
):
    result = planner.plan(
        make_intent("filesystem_metadata"),
        "Inspect the file metadata",
        available_tools,
    )

    assert result.status == ToolPlanStatus.CLARIFICATION_REQUIRED
    assert result.tool_name is None


def test_unavailable_capability_is_rejected(planner):
    result = planner.plan(
        make_intent("filesystem_existence"),
        "Does settings.json exist?",
        [],
    )

    assert result.status == ToolPlanStatus.UNSUPPORTED


def test_general_chat_is_not_planned(planner, available_tools):
    intent = IntentResult(
        intent=Intent.GENERAL_CHAT,
        confidence=0.95,
        matched_rule=None,
        requires_confirmation=False,
        requires_tool=False,
    )

    result = planner.plan(
        intent,
        "Hello SILVANA",
        available_tools,
    )

    assert result.status == ToolPlanStatus.UNSUPPORTED


def test_unsupported_explicit_action_is_not_misrouted(
    planner,
    available_tools,
):
    result = planner.plan(
        make_intent("explicit_action"),
        "Delete report.pdf",
        available_tools,
    )

    assert result.status == ToolPlanStatus.UNSUPPORTED


def test_schema_mismatch_is_rejected(planner):
    invalid_capability = CapabilityMatch(
        tool_name="check_file_exists",
        description="Check whether a file exists.",
        parameters={
            "type": "object",
            "properties": {},
            "required": ["path"],
        },
        permission_level="read",
        requires_confirmation=False,
    )

    result = planner.plan(
        make_intent("filesystem_existence"),
        "Does settings.json exist?",
        [invalid_capability],
    )

    assert result.status == ToolPlanStatus.UNSUPPORTED


def test_empty_request_requires_clarification(
    planner,
    available_tools,
):
    result = planner.plan(
        make_intent("filesystem_existence"),
        "   ",
        available_tools,
    )

    assert result.status == ToolPlanStatus.CLARIFICATION_REQUIRED


def test_invalid_intent_type_raises_error(planner, available_tools):
    with pytest.raises(TypeError):
        planner.plan(
            "filesystem_existence",
            "Does settings.json exist?",
            available_tools,
        )

def test_search_with_quoted_content_term_is_not_directory_listing(
    planner,
    available_tools,
):
    result = planner.plan(
        make_intent("filesystem_search"),
        "Search my workspace for Python files containing the word 'ExecutionEngine'.",
        available_tools,
    )

    assert result.status == ToolPlanStatus.UNSUPPORTED
    assert result.tool_name is None
    assert "file contents" in result.message


def test_show_matching_file_paths_remains_a_search(
    planner,
    available_tools,
):
    result = planner.plan(
        make_intent("filesystem_search"),
        "Find all Python files. Show me the matching file paths.",
        available_tools,
    )

    assert result.status == ToolPlanStatus.READY
    assert result.tool_name == "search_files"
    assert result.arguments["pattern"] == "*.py"


def test_workspace_with_trailing_punctuation_uses_root(
    planner,
    available_tools,
):
    result = planner.plan(
        make_intent("filesystem_search"),
        "Find all Python files in my workspace. Return their relative paths.",
        available_tools,
    )

    assert result.status == ToolPlanStatus.READY
    assert result.tool_name == "search_files"
    assert result.arguments["pattern"] == "*.py"
    assert "directory" not in result.arguments


def test_search_in_workspace_does_not_use_quoted_term_as_directory(
    planner,
    available_tools,
):
    result = planner.plan(
        make_intent("filesystem_search"),
        "Find files named 'ExecutionEngine.py' in my workspace.",
        available_tools,
    )

    assert result.status == ToolPlanStatus.READY
    assert result.tool_name == "search_files"
    assert result.arguments["pattern"] == "ExecutionEngine.py"
    assert "directory" not in result.arguments


def test_show_folder_contents_still_lists_directory(
    planner,
    available_tools,
):
    result = planner.plan(
        make_intent("filesystem_search"),
        "Show folder contents in src",
        available_tools,
    )

    assert result.status == ToolPlanStatus.READY
    assert result.tool_name == "list_directory"
    assert result.arguments["path"] == "src"
