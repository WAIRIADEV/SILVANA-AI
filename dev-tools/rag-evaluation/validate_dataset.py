
import json
from pathlib import Path
from typing import Any


DATASET_PATH = (
    Path(__file__).resolve().parent
    / "datasets"
    / "baseline.json"
)

SUPPORTED_SCOPES = {"none", "library"}


def require_non_empty_string(
    value: Any,
    field_name: str,
    case_id: str,
) -> None:
    """Require a non-empty string field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string "
            f"in case {case_id}."
        )


def validate_string_list(
    value: Any,
    field_name: str,
    case_id: str,
) -> None:
    """Validate a list containing only non-empty strings."""

    if not isinstance(value, list):
        raise ValueError(
            f"{field_name} must be a list in case {case_id}."
        )

    for index, item in enumerate(value):
        require_non_empty_string(
            item,
            f"{field_name}[{index}]",
            case_id,
        )


def validate_abstention_evidence(
    value: Any,
    case_id: str,
) -> None:
    """Validate explicit negative-evidence annotations."""

    if not isinstance(value, list):
        raise ValueError(
            f"abstention_evidence must be a list "
            f"in case {case_id}."
        )

    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"abstention_evidence[{index}] must be an object "
                f"in case {case_id}."
            )

        required_fields = {"document", "evidence"}
        missing = required_fields.difference(item)

        if missing:
            raise ValueError(
                f"abstention_evidence[{index}] in {case_id} "
                f"is missing fields: {sorted(missing)}"
            )

        require_non_empty_string(
            item["document"],
            f"abstention_evidence[{index}].document",
            case_id,
        )

        require_non_empty_string(
            item["evidence"],
            f"abstention_evidence[{index}].evidence",
            case_id,
        )


def validate_case(
    case: Any,
    seen_ids: set[str],
) -> None:
    """Validate one evaluation test case."""

    if not isinstance(case, dict):
        raise ValueError("Every test case must be a JSON object.")

    required_fields = {
        "id",
        "question",
        "relevant_documents",
        "relevant_evidence",
        "abstention_evidence",
        "knowledge_scope",
        "expected_abstention",
    }

    missing = required_fields.difference(case)

    if missing:
        raise ValueError(
            f"Test case is missing fields: {sorted(missing)}"
        )

    case_id = case["id"]

    require_non_empty_string(case_id, "id", case_id)

    if case_id in seen_ids:
        raise ValueError(f"Duplicate test case ID: {case_id}")

    seen_ids.add(case_id)

    require_non_empty_string(
        case["question"],
        "question",
        case_id,
    )

    if case["knowledge_scope"] not in SUPPORTED_SCOPES:
        raise ValueError(
            f"Unsupported knowledge_scope in case {case_id}: "
            f"{case['knowledge_scope']}"
        )

    if not isinstance(case["expected_abstention"], bool):
        raise ValueError(
            f"expected_abstention must be a boolean "
            f"in case {case_id}."
        )

    validate_string_list(
        case["relevant_documents"],
        "relevant_documents",
        case_id,
    )

    validate_string_list(
        case["relevant_evidence"],
        "relevant_evidence",
        case_id,
    )

    validate_abstention_evidence(
        case["abstention_evidence"],
        case_id,
    )

    if case["expected_abstention"]:
        if case["relevant_documents"] or case["relevant_evidence"]:
            raise ValueError(
                f"Abstention case {case_id} must not define "
                "positive answer evidence."
            )

        if case.get("expected_answer") is not None:
            raise ValueError(
                f"Abstention case {case_id} must have "
                "expected_answer set to null."
            )

        if case.get("expected_facts", []):
            raise ValueError(
                f"Abstention case {case_id} must not define "
                "expected_facts."
            )

    else:
        if (
            not case["relevant_documents"]
            or not case["relevant_evidence"]
        ):
            raise ValueError(
                f"Answerable case {case_id} must define "
                "positive relevant documents and evidence."
            )

        if case["abstention_evidence"]:
            raise ValueError(
                f"Answerable case {case_id} must not define "
                "abstention_evidence."
            )


def main() -> None:
    """Load and validate the complete evaluation dataset."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise ValueError("Dataset root must be a JSON object.")

    for field in ("dataset_name", "dataset_version"):
        require_non_empty_string(
            dataset.get(field),
            field,
            "dataset",
        )

    cases = dataset.get("test_cases")

    if not isinstance(cases, list):
        raise ValueError("Dataset must contain a test_cases list.")

    if not cases:
        raise ValueError("Dataset contains no test cases.")

    seen_ids: set[str] = set()

    for case in cases:
        validate_case(case, seen_ids)

    answerable_count = sum(
        not case["expected_abstention"]
        for case in cases
    )

    abstention_count = sum(
        case["expected_abstention"]
        for case in cases
    )

    explicit_negative_evidence_count = sum(
        bool(case["abstention_evidence"])
        for case in cases
    )

    print("DATASET VALID")
    print("Name:", dataset["dataset_name"])
    print("Version:", dataset["dataset_version"])
    print("Test cases:", len(cases))
    print("Answerable cases:", answerable_count)
    print("Abstention cases:", abstention_count)
    print(
        "Abstention cases with explicit negative evidence:",
        explicit_negative_evidence_count,
    )

    for case in cases:
        print(
            f"{case['id']}: "
            f"abstention={case['expected_abstention']}, "
            f"scope={case['knowledge_scope']}, "
            f"negative_evidence={len(case['abstention_evidence'])}"
        )


if __name__ == "__main__":
    main()
