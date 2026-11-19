
"""Combine individual answer evaluations and retrieval results.

Supports legacy citation_completeness and current citation_presence
fields. Makes no provider API calls.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"

DATASET_PATH = ROOT / "datasets" / "baseline.json"
RETRIEVAL_PATH = RESULTS_ROOT / "retrieval_latest.json"
OUTPUT_PATH = RESULTS_ROOT / "combined_baseline_latest.json"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")

    return data


def calculate_rate(
    cases: list[dict[str, Any]],
    field: str,
) -> float | None:
    """Average numeric or boolean values, excluding missing values."""

    values = [
        float(case[field])
        for case in cases
        if case.get(field) is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


def normalize_answer_case(
    case: dict[str, Any],
    report: dict[str, Any],
    filename: str,
) -> dict[str, Any]:
    """Normalize old and new answer evaluation case formats."""

    normalized = dict(case)

    # Support both the old and new citation-presence fields.
    if "citation_presence" in case:
        citation_presence = case["citation_presence"]
    else:
        citation_presence = case.get("citation_completeness")

    normalized["citation_presence"] = citation_presence

    normalized["provider"] = case.get(
        "provider",
        report.get("provider_requested"),
    )

    normalized["model"] = case.get("model")
    normalized["report_file"] = filename
    normalized["dataset_version"] = report.get("dataset_version")
    normalized["dataset_name"] = report.get("dataset_name")

    return normalized


def load_answer_cases(
    expected_case_ids: list[str],
    dataset_name: str,
    dataset_version: str,
) -> list[dict[str, Any]]:
    """Load one saved answer report for every expected dataset case."""

    answer_files = sorted(
        RESULTS_ROOT.glob("answers_water-*.json")
    )

    if not answer_files:
        raise FileNotFoundError(
            "No individual answer evaluation reports were found."
        )

    cases_by_id: dict[str, dict[str, Any]] = {}

    for path in answer_files:
        report = load_json(path)

        if report.get("evaluation_type") != "answer_generation":
            raise ValueError(
                f"{path.name} is not an answer-generation report."
            )

        if report.get("dataset_name") != dataset_name:
            raise ValueError(
                f"Dataset name mismatch in {path.name}."
            )

        if str(report.get("dataset_version")) != dataset_version:
            raise ValueError(
                f"Dataset version mismatch in {path.name}."
            )

        raw_cases = report.get("test_cases")

        if not isinstance(raw_cases, list):
            raise ValueError(
                f"{path.name} has no valid test_cases list."
            )

        if len(raw_cases) != 1:
            raise ValueError(
                f"{path.name} must contain exactly one evaluated case."
            )

        case = raw_cases[0]
        case_id = case.get("id")

        if not case_id:
            raise ValueError(
                f"{path.name} contains a case without an ID."
            )

        if case_id not in expected_case_ids:
            raise ValueError(
                f"Unexpected answer case ID '{case_id}' "
                f"in {path.name}."
            )

        if case_id in cases_by_id:
            raise ValueError(
                f"Duplicate answer report for case '{case_id}'."
            )

        cases_by_id[case_id] = normalize_answer_case(
            case=case,
            report=report,
            filename=path.name,
        )

    missing = sorted(
        set(expected_case_ids) - set(cases_by_id)
    )

    if missing:
        raise ValueError(
            "Missing answer evaluation reports for: "
            + ", ".join(missing)
        )

    return [
        cases_by_id[case_id]
        for case_id in expected_case_ids
    ]


def build_answer_metrics(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate normalized answer-generation metrics."""

    answerable = [
        case for case in cases
        if not case["expected_abstention"]
    ]

    abstention_cases = [
        case for case in cases
        if case["expected_abstention"]
    ]

    providers = Counter(
        case["provider"]
        for case in cases
        if case.get("provider")
    )

    models = Counter(
        case["model"]
        for case in cases
        if case.get("model")
    )

    return {
        "total_cases": len(cases),
        "answerable_cases": len(answerable),
        "abstention_cases": len(abstention_cases),
        "answer_correctness_rate": calculate_rate(
            answerable,
            "answer_correct",
        ),
        "abstention_accuracy": calculate_rate(
            cases,
            "abstention_correct",
        ),
        "citation_validity_rate": calculate_rate(
            cases,
            "citation_validity",
        ),
        "citation_relevance_rate": calculate_rate(
            cases,
            "citation_relevance",
        ),
        "citation_presence_rate_answerable": calculate_rate(
            answerable,
            "citation_presence",
        ),
        "citation_presence_rate_abstention": calculate_rate(
            abstention_cases,
            "citation_presence",
        ),
        "provider_distribution": dict(providers),
        "model_distribution": dict(models),
    }


def load_retrieval_cases(
    retrieval_report: dict[str, Any],
    expected_case_ids: list[str],
    dataset_name: str,
    dataset_version: str,
) -> dict[str, dict[str, Any]]:
    """Validate and index retrieval results by case ID."""

    if retrieval_report.get("dataset_name") != dataset_name:
        raise ValueError(
            "Retrieval report dataset name does not match baseline."
        )

    if str(retrieval_report.get("dataset_version")) != dataset_version:
        raise ValueError(
            "Retrieval report dataset version does not match baseline."
        )

    raw_cases = retrieval_report.get("test_cases")

    if not isinstance(raw_cases, list):
        raise ValueError(
            "Retrieval report has no valid test_cases list."
        )

    cases_by_id: dict[str, dict[str, Any]] = {}

    for case in raw_cases:
        case_id = case.get("id")

        if not case_id:
            raise ValueError(
                "Retrieval report contains a case without an ID."
            )

        if case_id in cases_by_id:
            raise ValueError(
                f"Duplicate retrieval case ID: {case_id}."
            )

        if case_id not in expected_case_ids:
            raise ValueError(
                f"Unexpected retrieval case ID: {case_id}."
            )

        cases_by_id[case_id] = case

    missing = sorted(
        set(expected_case_ids) - set(cases_by_id)
    )

    if missing:
        raise ValueError(
            "Missing retrieval results for: "
            + ", ".join(missing)
        )

    return cases_by_id


def build_case_results(
    answer_cases: list[dict[str, Any]],
    retrieval_cases: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine answer and retrieval information per question."""

    combined = []

    for answer_case in answer_cases:
        case_id = answer_case["id"]
        retrieval_case = retrieval_cases[case_id]

        combined.append(
            {
                "id": case_id,
                "question": answer_case.get("question"),
                "expected_abstention": answer_case.get(
                    "expected_abstention"
                ),
                "provider": answer_case.get("provider"),
                "model": answer_case.get("model"),
                "answer": answer_case.get("answer"),
                "answer_correct": answer_case.get(
                    "answer_correct"
                ),
                "expected_answer_found": answer_case.get(
                    "expected_answer_found"
                ),
                "matched_expected_facts": answer_case.get(
                    "matched_expected_facts",
                    [],
                ),
                "all_expected_facts_found": answer_case.get(
                    "all_expected_facts_found"
                ),
                "abstention_detected_heuristically": (
                    answer_case.get(
                        "abstention_detected_heuristically"
                    )
                ),
                "abstention_correct": answer_case.get(
                    "abstention_correct"
                ),
                "citations": answer_case.get("citations", []),
                "valid_citations": answer_case.get(
                    "valid_citations",
                    [],
                ),
                "relevant_citations": answer_case.get(
                    "relevant_citations",
                    [],
                ),
                "citation_presence": answer_case.get(
                    "citation_presence"
                ),
                "citation_validity": answer_case.get(
                    "citation_validity"
                ),
                "citation_relevance": answer_case.get(
                    "citation_relevance"
                ),
                "unsupported_claims": answer_case.get(
                    "unsupported_claims"
                ),
                "sources": answer_case.get("sources", []),
                "answer_report_file": answer_case.get(
                    "report_file"
                ),
                "retrieval": {
                    "retrieved_count": retrieval_case.get(
                        "retrieved_count"
                    ),
                    "evidence_type_evaluated": (
                        retrieval_case.get(
                            "evidence_type_evaluated"
                        )
                    ),
                    "metrics": retrieval_case.get(
                        "metrics",
                        {},
                    ),
                    "negative_evidence_found": (
                        retrieval_case.get(
                            "negative_evidence_found"
                        )
                    ),
                    "negative_evidence_coverage_at_5": (
                        retrieval_case.get(
                            "negative_evidence_coverage_at_5"
                        )
                    ),
                    "abstention_retrieval_passed": (
                        retrieval_case.get(
                            "abstention_retrieval_passed"
                        )
                    ),
                    "retrieved_chunks": retrieval_case.get(
                        "retrieved_chunks",
                        [],
                    ),
                },
            }
        )

    return combined


def main() -> None:
    """Build the consolidated evaluation report."""

    print("SILVANA CONSOLIDATED RAG EVALUATION")
    print()

    dataset = load_json(DATASET_PATH)
    retrieval_report = load_json(RETRIEVAL_PATH)

    # Match the actual baseline.json schema.
    dataset_name = dataset.get("dataset_name")
    dataset_version = str(dataset.get("dataset_version"))

    raw_dataset_cases = dataset.get("test_cases")

    if not dataset_name or not dataset_version:
        raise ValueError(
            "Dataset must define dataset_name and dataset_version."
        )

    if not isinstance(raw_dataset_cases, list) or not raw_dataset_cases:
        raise ValueError(
            "Dataset must contain a non-empty test_cases list."
        )

    expected_case_ids = [
        case["id"]
        for case in raw_dataset_cases
    ]

    if any(not case_id for case_id in expected_case_ids):
        raise ValueError(
            "Dataset contains a test case without an ID."
        )

    if len(expected_case_ids) != len(set(expected_case_ids)):
        raise ValueError(
            "Dataset contains duplicate case IDs."
        )

    answer_cases = load_answer_cases(
        expected_case_ids=expected_case_ids,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )

    retrieval_cases = load_retrieval_cases(
        retrieval_report=retrieval_report,
        expected_case_ids=expected_case_ids,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )

    combined_cases = build_case_results(
        answer_cases=answer_cases,
        retrieval_cases=retrieval_cases,
    )

    answer_metrics = build_answer_metrics(answer_cases)
    retrieval_metrics = retrieval_report.get("metrics", {})

    report = {
        "evaluation_type": "consolidated_rag_baseline",
        "report_version": "1.0",
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluation_environment": {
            "database": "isolated evaluation database",
            "embedding_model": retrieval_report.get(
                "embedding_model"
            ),
            "retrieval_top_k": retrieval_report.get(
                "retrieval_top_k"
            ),
            "providers": answer_metrics[
                "provider_distribution"
            ],
            "models": answer_metrics[
                "model_distribution"
            ],
        },
        "metrics": {
            "answer_generation": answer_metrics,
            "retrieval": retrieval_metrics,
        },
        "case_count": len(combined_cases),
        "case_ids": expected_case_ids,
        "test_cases": combined_cases,
        "evaluation_notes": [
            (
                "Legacy citation_completeness values were normalized "
                "to citation_presence."
            ),
            (
                "Citation validity measures references against "
                "the sources returned by the production pipeline."
            ),
            (
                "Citation relevance measures references against "
                "the annotated expected evidence documents."
            ),
            (
                "Correct abstentions do not require citations."
            ),
            (
                "Answer correctness uses deterministic expected-answer "
                "and expected-fact matching, not semantic entailment."
            ),
            (
                "Retrieval metrics use annotated evidence matching "
                "and should not be interpreted as general semantic "
                "relevance measurements."
            ),
            (
                "Provider and model attribution is retained for "
                "every answer-generation case."
            ),
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    print(f"Dataset: {dataset_name}")
    print(f"Version: {dataset_version}")
    print(f"Cases combined: {len(combined_cases)}")
    print()

    print("ANSWER GENERATION")
    for key, value in answer_metrics.items():
        print(f"{key}: {value}")

    print()
    print("RETRIEVAL")
    for key, value in retrieval_metrics.items():
        print(f"{key}: {value}")

    print()
    print(f"Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
