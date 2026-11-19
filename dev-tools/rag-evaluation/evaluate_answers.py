
import argparse
import asyncio
import json
import re
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import HTTPException

from backend.app.api.routes.chat import process_chat_request
from backend.app.api.schemas.chat import (
    ChatMessageSchema,
    ChatRequest,
)
from backend.app.intelligence.base import ChatMessage
from backend.app.intelligence.conversation_context import (
    ConversationContextAnalyzer,
)
from backend.app.intelligence.intent_engine import IntentEngine, Intent
from backend.app.intelligence.provider_manager import ProviderManager
from backend.app.intelligence.tool_planner import ToolPlanner

from config import evaluation_config
from evaluate_retrieval import (
    build_retrieval_service,
    load_dataset,
)


ABSTENTION_PATTERNS = (
    "not specified",
    "do not specify",
    "does not specify",
    "doesn't specify",
    "not provided",
    "not available",
    "no information",
    "no relevant information",
    "no evidence",
    "cannot determine",
    "can't determine",
    "unable to determine",
    "i don't know",
    "i do not know",
    "insufficient information",
    "not stated",
    "does not mention",
    "doesn't mention",
    "do not mention",
    "not mentioned",
    "no mention of",
    "none of the retrieved documents",
    "none of the documents contain",
    "no documents contain",
    "no document contains",
    "none of the available documents",
    "retrieved documents contain no",
    "documents do not contain",
    "documents don't contain",
)


def normalize_text(value: str) -> str:
    """Normalize text for punctuation-insensitive comparisons."""

    normalized = re.sub(
        r"[^\w\s]",
        " ",
        value.casefold(),
    )

    return " ".join(normalized.split())


def detect_abstention(answer: str) -> bool:
    """Detect common explicit abstention phrases."""

    normalized = normalize_text(answer)

    return any(
        normalize_text(phrase) in normalized
        for phrase in ABSTENTION_PATTERNS
    )


def extract_citations(answer: str) -> list[str]:
    """Extract unique source references such as [S1] and [S2]."""

    return list(dict.fromkeys(
        re.findall(r"\[(S\d+)\]", answer, flags=re.IGNORECASE)
    ))


def evaluate_answer(
    case: dict[str, Any],
    response: Any,
) -> dict[str, Any]:
    """Score factual matches, abstention, and citation behavior."""

    answer = response.content or ""
    normalized_answer = normalize_text(answer)

    expected_answer = case.get("expected_answer")

    expected_answer_found = (
        normalize_text(expected_answer) in normalized_answer
        if expected_answer
        else None
    )

    expected_facts = case.get("expected_facts", [])

    matched_facts = [
        fact
        for fact in expected_facts
        if normalize_text(fact) in normalized_answer
    ]

    all_expected_facts_found = (
        len(matched_facts) == len(expected_facts)
        if expected_facts
        else None
    )

    answer_correct = (
        (
            expected_answer_found is True
            or all_expected_facts_found is True
        )
        if not case["expected_abstention"]
        else None
    )

    abstention_detected = detect_abstention(answer)

    abstention_correct = (
        abstention_detected == case["expected_abstention"]
    )

    sources = [
        {
            "ref": source.ref,
            "filename": source.filename,
            "document_id": str(source.document_id),
            "scope": source.scope,
            "page_number": source.page_number,
            "chunk_index": source.chunk_index,
        }
        for source in response.sources
    ]

    valid_source_refs = {
        source["ref"].upper()
        for source in sources
    }

    citations = extract_citations(answer)

    valid_citations = [
        citation
        for citation in citations
        if citation.upper() in valid_source_refs
    ]

    if case["expected_abstention"]:
        expected_documents = {
            item["document"]
            for item in case["abstention_evidence"]
        }
    else:
        expected_documents = set(case["relevant_documents"])

    source_by_ref = {
        source["ref"].upper(): source
        for source in sources
    }

    relevant_citations = [
        citation
        for citation in valid_citations
        if source_by_ref[citation.upper()]["filename"]
        in expected_documents
    ]

    citation_validity = (
        len(valid_citations) / len(citations)
        if citations
        else None
    )

    citation_relevance = (
        len(relevant_citations) / len(citations)
        if citations
        else None
    )

    citation_presence = bool(citations)

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_abstention": case["expected_abstention"],
        "answer": answer,
        "answer_correct": answer_correct,
        "expected_answer_found": expected_answer_found,
        "expected_facts": expected_facts,
        "matched_expected_facts": matched_facts,
        "all_expected_facts_found": all_expected_facts_found,
        "abstention_detected_heuristically": abstention_detected,
        "abstention_correct": abstention_correct,
        "citations": citations,
        "valid_citations": valid_citations,
        "relevant_citations": relevant_citations,
        "citation_validity": citation_validity,
        "citation_relevance": citation_relevance,
        "citation_presence": citation_presence,
        "sources": sources,
        "provider": response.provider,
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "finish_reason": response.finish_reason,
        "unsupported_claims": {
            "status": "not_evaluated",
            "reason": (
                "Requires semantic claim extraction and "
                "evidence entailment evaluation."
            ),
        },
    }


def select_cases(
    dataset: dict[str, Any],
    case_id: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Select cases without changing the source dataset."""

    cases = dataset["test_cases"]

    if case_id:
        cases = [
            case
            for case in cases
            if case["id"] == case_id
        ]

        if not cases:
            raise ValueError(
                f"Test case not found: {case_id}"
            )

    if limit is not None:
        cases = cases[:limit]

    if not cases:
        raise ValueError("No evaluation cases selected.")

    return cases


def validate_intents(cases: list[dict[str, Any]]) -> None:
    """Reject cases that would enter the production tool path."""

    analyzer = ConversationContextAnalyzer()
    intent_engine = IntentEngine()

    for case in cases:
        message = ChatMessage(
            role="user",
            content=case["question"],
        )

        context = analyzer.analyze([message])

        intent_result = intent_engine.classify(
            case["question"],
            context=context,
        )

        if intent_result.intent == Intent.TOOL_ACTION:
            raise ValueError(
                f"Case {case['id']} was classified as TOOL_ACTION. "
                "It cannot safely run in this RAG benchmark."
            )

        print(
            f"Preflight {case['id']}: "
            f"{intent_result.intent.value}"
        )


def save_report(report: dict[str, Any]) -> Path:
    """Save the answer evaluation report."""

    evaluation_config.results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        evaluation_config.results_root
        / "answers_latest.json"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return report_path


def aggregate_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate aggregate answer and citation scores."""

    answerable = [
        item for item in results
        if not item["expected_abstention"]
    ]

    abstention_cases = [
        item for item in results
        if item["expected_abstention"]
    ]

    def rate(items, field):
        evaluated = [
            float(item[field])
            for item in items
            if item[field] is not None
        ]

        if not evaluated:
            return None

        return sum(evaluated) / len(evaluated)

    return {
        "answerable_cases": len(answerable),
        "abstention_cases": len(abstention_cases),

        "answer_correctness_rate": rate(
            answerable,
            "answer_correct",
        ),

        "abstention_accuracy": rate(
            results,
            "abstention_correct",
        ),

        "citation_validity_rate": rate(
            results,
            "citation_validity",
        ),

        "citation_relevance_rate": rate(
            results,
            "citation_relevance",
        ),

        "citation_presence_rate_answerable": rate(
            answerable,
            "citation_presence",
        ),

        "citation_presence_rate_abstention": rate(
            abstention_cases,
            "citation_presence",
        ),
    }


async def run_evaluation(
    cases: list[dict[str, Any]],
    provider_name: str,
) -> list[dict[str, Any]]:
    """Run selected cases through the real production chat pipeline."""

    evaluation_config.validate_isolation()

    _, retrieval_service = build_retrieval_service()

    manager = ProviderManager()
    planner = ToolPlanner()

    results = []

    try:
        for index, case in enumerate(cases, start=1):
            print()
            print(
                f"Running case {index}/{len(cases)}: "
                f"{case['id']}"
            )

            request = ChatRequest(
                messages=[
                    ChatMessageSchema(
                        role="user",
                        content=case["question"],
                    )
                ],
                provider=provider_name,
                temperature=0.0,
                max_output_tokens=512,
                conversation_id=None,
                knowledge_scope=case["knowledge_scope"],
            )

            try:
                response = await process_chat_request(
                    request=request,
                    manager=manager,
                    orchestrator=None,
                    planner=planner,
                    retrieval_service=retrieval_service,
                )

            except HTTPException as error:
                raise RuntimeError(
                    f"Production chat pipeline failed for "
                    f"{case['id']}: HTTP {error.status_code} - "
                    f"{error.detail}"
                ) from error

            result = evaluate_answer(case, response)
            results.append(result)

            print(f"Provider: {result['provider']}")
            print(f"Model: {result['model']}")
            print(f"Answer: {result['answer']}")
            print(
                "Answer correctness:",
                result["answer_correct"],
            )
            print(
                "Abstention detected:",
                result["abstention_detected_heuristically"],
            )
            print(
                "Citations:",
                result["citations"],
            )

    finally:
        await manager.aclose()

    return results


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate SILVANA's production RAG answer pipeline."
    )

    parser.add_argument(
        "--provider",
        choices=("gemini", "groq"),
        required=True,
        help="LLM provider to evaluate.",
    )

    parser.add_argument(
        "--case-id",
        help="Run one specific dataset case.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of selected cases to run.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    evaluation_config.validate_isolation()

    dataset = load_dataset()

    cases = select_cases(
        dataset,
        case_id=args.case_id,
        limit=args.limit,
    )

    print("SILVANA RAG ANSWER EVALUATION")
    print()
    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Dataset version: {dataset['dataset_version']}")
    print(f"Provider: {args.provider}")
    print(f"Selected cases: {len(cases)}")
    print()

    validate_intents(cases)

    results = asyncio.run(
        run_evaluation(
            cases=cases,
            provider_name=args.provider,
        )
    )

    aggregate = aggregate_results(results)

    report = {
        "evaluation_type": "answer_generation",
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["dataset_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider_requested": args.provider,
        "evaluation_database": str(
            evaluation_config.database_path
        ),
        "embedding_model": evaluation_config.embedding_model,
        "selected_case_count": len(cases),
        "metrics": aggregate,
        "test_cases": results,
    }

    report_path = save_report(report)

    print()
    print("AGGREGATE ANSWER RESULTS")

    for name, value in aggregate.items():
        if isinstance(value, float):
            print(f"{name}: {value:.3f}")
        else:
            print(f"{name}: {value}")

    print()
    print(f"Report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
