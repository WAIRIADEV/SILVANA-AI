import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.intelligence.documents.repository import DocumentRepository
from backend.app.intelligence.documents.retrieval import (
    DocumentRetrievalService,
)
from backend.app.intelligence.embeddings.fastembed import FastEmbedProvider

from config import evaluation_config


EVALUATION_K_VALUES = (1, 3, 5)


def load_dataset() -> dict[str, Any]:
    """Load and validate the baseline evaluation dataset."""

    path = evaluation_config.dataset_path

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset.get("test_cases"), list):
        raise ValueError("Dataset must contain a test_cases list.")

    if not dataset["test_cases"]:
        raise ValueError("Dataset contains no test cases.")

    seen_ids = set()

    for case in dataset["test_cases"]:
        required = {
            "id",
            "question",
            "relevant_documents",
            "relevant_evidence",
            "abstention_evidence",
            "knowledge_scope",
            "expected_abstention",
        }

        missing = required.difference(case)

        if missing:
            raise ValueError(
                f"Test case is missing fields: {sorted(missing)}"
            )

        if case["id"] in seen_ids:
            raise ValueError(f"Duplicate test case ID: {case['id']}")

        seen_ids.add(case["id"])

        if (
            not isinstance(case["question"], str)
            or not case["question"].strip()
        ):
            raise ValueError(f"Invalid question in case {case['id']}.")

        if case["knowledge_scope"] not in {"none", "library"}:
            raise ValueError(
                f"Unsupported evaluation scope in case {case['id']}."
            )

        if not isinstance(case["expected_abstention"], bool):
            raise ValueError(
                f"Invalid expected_abstention in case {case['id']}."
            )

        for field in (
            "relevant_documents",
            "relevant_evidence",
            "abstention_evidence",
        ):
            if not isinstance(case[field], list):
                raise ValueError(
                    f"{field} must be a list in case {case['id']}."
                )

        if case["expected_abstention"]:
            if case["relevant_documents"] or case["relevant_evidence"]:
                raise ValueError(
                    f"Abstention case {case['id']} must not define "
                    "positive evidence."
                )

            if case.get("expected_answer") is not None:
                raise ValueError(
                    f"Abstention case {case['id']} must have "
                    "expected_answer set to null."
                )

            if case.get("expected_facts", []):
                raise ValueError(
                    f"Abstention case {case['id']} must not define "
                    "expected_facts."
                )

        else:
            if (
                not case["relevant_documents"]
                or not case["relevant_evidence"]
            ):
                raise ValueError(
                    f"Answerable case {case['id']} must define "
                    "positive evidence."
                )

            if case["abstention_evidence"]:
                raise ValueError(
                    f"Answerable case {case['id']} must not define "
                    "abstention_evidence."
                )

        for document in case["relevant_documents"]:
            if not isinstance(document, str) or not document.strip():
                raise ValueError(
                    f"Invalid relevant document in case {case['id']}."
                )

        for evidence in case["relevant_evidence"]:
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError(
                    f"Invalid positive evidence in case {case['id']}."
                )

        for item in case["abstention_evidence"]:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid abstention evidence in case {case['id']}."
                )

            if (
                not isinstance(item.get("document"), str)
                or not item["document"].strip()
                or not isinstance(item.get("evidence"), str)
                or not item["evidence"].strip()
            ):
                raise ValueError(
                    f"Invalid abstention evidence fields "
                    f"in case {case['id']}."
                )

    return dataset


def normalize_text(value: str) -> str:
    """Normalize text for case-insensitive evidence matching."""

    return " ".join(value.casefold().split())


def build_retrieval_service():
    """Construct retrieval using isolated evaluation resources."""

    evaluation_config.prepare_directories()

    repository = DocumentRepository(
        database_path=evaluation_config.database_path
    )
    repository.initialize()

    embedding_provider = FastEmbedProvider(
        model_name=evaluation_config.embedding_model,
        batch_size=evaluation_config.embedding_batch_size,
        threads=evaluation_config.embedding_threads,
    )

    retrieval_service = DocumentRetrievalService(
        repository=repository,
        embedding_provider=embedding_provider,
    )

    return repository, retrieval_service


def calculate_metrics(
    retrieved: list[dict[str, Any]],
    expected_evidence: list[str],
    k_values: tuple[int, ...],
    relevance_field: str,
    matched_field: str,
) -> dict[str, Any]:
    """Calculate evidence retrieval metrics at each K."""

    metrics = {}

    for k in k_values:
        top_results = retrieved[:k]

        relevant_chunks = sum(
            1
            for item in top_results
            if item[relevance_field]
        )

        matched_evidence = {
            evidence
            for item in top_results
            for evidence in item[matched_field]
        }

        if expected_evidence:
            recall = (
                len(matched_evidence) / len(expected_evidence)
            )

            precision = (
                relevant_chunks / len(top_results)
                if top_results
                else 0.0
            )

            first_relevant_rank = next(
                (
                    item["rank"]
                    for item in top_results
                    if item[relevance_field]
                ),
                None,
            )

            reciprocal_rank = (
                1 / first_relevant_rank
                if first_relevant_rank is not None
                else 0.0
            )
        else:
            recall = None
            precision = None
            reciprocal_rank = None

        metrics[f"recall_at_{k}"] = recall
        metrics[f"precision_at_{k}"] = precision
        metrics[f"mrr_at_{k}"] = reciprocal_rank

    return metrics


def evaluate_case(
    case: dict[str, Any],
    retrieval_service: DocumentRetrievalService,
) -> dict[str, Any]:
    """Evaluate positive or negative evidence retrieval."""

    expected_documents = set(case["relevant_documents"])

    positive_evidence = [
        normalize_text(item)
        for item in case["relevant_evidence"]
    ]

    negative_evidence = [
        {
            "document": item["document"],
            "evidence": normalize_text(item["evidence"]),
        }
        for item in case["abstention_evidence"]
    ]

    results = retrieval_service.retrieve(
        case["question"],
        knowledge_scope=case["knowledge_scope"],
        limit=max(EVALUATION_K_VALUES),
    )

    retrieved = []

    for rank, result in enumerate(results, start=1):
        normalized_content = normalize_text(result.content)

        matched_positive = [
            evidence
            for evidence in positive_evidence
            if evidence in normalized_content
            and result.filename in expected_documents
        ]

        matched_negative = [
            item["evidence"]
            for item in negative_evidence
            if (
                item["document"] == result.filename
                and item["evidence"] in normalized_content
            )
        ]

        positive_relevant = bool(matched_positive)
        negative_relevant = bool(matched_negative)

        retrieved.append({
            "rank": rank,
            "filename": result.filename,
            "chunk_index": result.chunk_index,
            "page_number": result.page_number,
            "similarity": result.similarity,
            "positive_relevant": positive_relevant,
            "negative_relevant": negative_relevant,
            "relevant": positive_relevant or negative_relevant,
            "matched_positive_evidence": matched_positive,
            "matched_negative_evidence": matched_negative,
            "content": result.content,
        })

    is_abstention = case["expected_abstention"]

    if is_abstention:
        active_evidence = negative_evidence
        active_matched_field = "matched_negative_evidence"
        active_relevance_field = "negative_relevant"
    else:
        active_evidence = positive_evidence
        active_matched_field = "matched_positive_evidence"
        active_relevance_field = "positive_relevant"

    expected_evidence_values = [
        item["evidence"] if isinstance(item, dict) else item
        for item in active_evidence
    ]

    metrics = calculate_metrics(
        retrieved=retrieved,
        expected_evidence=expected_evidence_values,
        k_values=EVALUATION_K_VALUES,
        relevance_field=active_relevance_field,
        matched_field=active_matched_field,
    )

    negative_evidence_found = any(
        item["negative_relevant"] for item in retrieved
    )

    negative_evidence_coverage = (
        metrics["recall_at_5"]
        if is_abstention
        else None
    )

    abstention_retrieval_passed = (
        negative_evidence_coverage == 1.0
        if is_abstention and expected_evidence_values
        else None
    )

    return {
        "id": case["id"],
        "question": case["question"],
        "knowledge_scope": case["knowledge_scope"],
        "expected_abstention": is_abstention,
        "retrieved_count": len(retrieved),
        "evidence_type_evaluated": (
            "negative" if is_abstention else "positive"
        ),
        "metrics": metrics,
        "negative_evidence_found": (
            negative_evidence_found if is_abstention else None
        ),
        "negative_evidence_coverage_at_5": (
            negative_evidence_coverage
        ),
        "abstention_retrieval_passed": (
            abstention_retrieval_passed
        ),
        "retrieved_chunks": retrieved,
    }


def aggregate_metrics(
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate separate averages for positive and negative evidence."""

    answerable_cases = [
        result
        for result in case_results
        if not result["expected_abstention"]
    ]

    abstention_cases = [
        result
        for result in case_results
        if result["expected_abstention"]
    ]

    aggregate = {}

    for k in EVALUATION_K_VALUES:
        for metric_name in (
            f"recall_at_{k}",
            f"precision_at_{k}",
            f"mrr_at_{k}",
        ):
            positive_values = [
                result["metrics"][metric_name]
                for result in answerable_cases
                if result["metrics"][metric_name] is not None
            ]

            negative_values = [
                result["metrics"][metric_name]
                for result in abstention_cases
                if result["metrics"][metric_name] is not None
            ]

            aggregate[f"positive_{metric_name}"] = (
                mean(positive_values) if positive_values else None
            )

            aggregate[f"negative_{metric_name}"] = (
                mean(negative_values) if negative_values else None
            )

    abstention_passes = sum(
        1
        for result in abstention_cases
        if result["abstention_retrieval_passed"] is True
    )

    evaluated_abstention_cases = sum(
        1
        for result in abstention_cases
        if result["abstention_retrieval_passed"] is not None
    )

    aggregate["abstention_evidence_retrieval_pass_rate"] = (
        abstention_passes / evaluated_abstention_cases
        if evaluated_abstention_cases
        else None
    )

    aggregate["answerable_case_count"] = len(answerable_cases)
    aggregate["abstention_case_count"] = len(abstention_cases)
    aggregate["abstention_cases_with_evidence"] = (
        evaluated_abstention_cases
    )

    return aggregate


def save_report(report: dict[str, Any]) -> Path:
    """Save the evaluation report as JSON."""

    evaluation_config.results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        evaluation_config.results_root / "retrieval_latest.json"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    return report_path


def main() -> int:
    print("SILVANA RAG EVALUATION — RETRIEVAL BENCHMARK")
    print()

    evaluation_config.validate_isolation()

    dataset = load_dataset()
    repository, retrieval_service = build_retrieval_service()

    indexed_documents = repository.list_documents()

    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Dataset version: {dataset['dataset_version']}")
    print(f"Embedding model: {evaluation_config.embedding_model}")
    print(f"Evaluation documents: {len(indexed_documents)}")
    print(f"Test cases: {len(dataset['test_cases'])}")
    print()

    case_results = []

    for case in dataset["test_cases"]:
        result = evaluate_case(case, retrieval_service)
        case_results.append(result)

        print(f"CASE {result['id']}: {result['evidence_type_evaluated']} evidence")

        for k in EVALUATION_K_VALUES:
            metrics = result["metrics"]

            recall = metrics[f"recall_at_{k}"]
            precision = metrics[f"precision_at_{k}"]
            mrr = metrics[f"mrr_at_{k}"]

            if recall is None:
                print(f"  @K={k}: N/A (no annotated evidence)")
            else:
                print(
                    f"  @K={k}: "
                    f"Recall={recall:.3f}, "
                    f"Precision={precision:.3f}, "
                    f"MRR={mrr:.3f}"
                )

        if result["expected_abstention"]:
            print(
                "  Negative evidence found:",
                "YES" if result["negative_evidence_found"] else "NO",
            )

            print(
                "  Abstention evidence retrieval:",
                (
                    "PASS"
                    if result["abstention_retrieval_passed"] is True
                    else "FAIL"
                    if result["abstention_retrieval_passed"] is False
                    else "NOT EVALUATED"
                ),
            )

        print(f"  Retrieved chunks: {result['retrieved_count']}")
        print()

    aggregate = aggregate_metrics(case_results)

    report = {
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["dataset_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": evaluation_config.embedding_model,
        "retrieval_top_k": evaluation_config.retrieval_top_k,
        "evaluation_k_values": list(EVALUATION_K_VALUES),
        "metrics": aggregate,
        "test_cases": case_results,
    }

    report_path = save_report(report)

    print("AGGREGATE RETRIEVAL RESULTS")

    for name, value in aggregate.items():
        if isinstance(value, float):
            print(f"{name}: {value:.3f}")
        else:
            print(f"{name}: {value}")

    print()
    print(f"Detailed report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
