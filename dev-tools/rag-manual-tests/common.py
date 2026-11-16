
import os
from pathlib import Path
from uuid import UUID

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.getenv(
    "SILVANA_API_BASE_URL",
    "http://127.0.0.1:8000/api/v1",
).rstrip("/")

PROVIDER = os.getenv("SILVANA_TEST_PROVIDER", "groq")
TIMEOUT = 60

RUN_ID = os.getenv("SILVANA_TEST_RUN_ID")
CONVERSATION_ID = os.getenv("SILVANA_TEST_CONVERSATION_ID")

FIXTURE_PATH = Path(__file__).resolve().parent / "silvana_rag_test.txt"

if not RUN_ID:
    raise RuntimeError(
        "SILVANA_TEST_RUN_ID is missing. "
        "Initialize the test environment in PowerShell first."
    )

if not CONVERSATION_ID:
    raise RuntimeError(
        "SILVANA_TEST_CONVERSATION_ID is missing. "
        "Initialize the test environment in PowerShell first."
    )

try:
    UUID(RUN_ID)
    UUID(CONVERSATION_ID)
except ValueError as exc:
    raise RuntimeError("The test run ID and conversation ID must be valid UUIDs.") from exc

if not FIXTURE_PATH.is_file():
    raise FileNotFoundError(f"Test fixture not found: {FIXTURE_PATH}")

TEST_FILENAME = f"silvana_rag_test_{RUN_ID}.txt"

SESSION = requests.Session()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def request_json(method: str, endpoint: str, **kwargs):
    """Make an API request and return its JSON response."""

    url = f"{BASE_URL}{endpoint}"

    try:
        response = SESSION.request(
            method=method,
            url=url,
            timeout=TIMEOUT,
            **kwargs,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach SILVANA API at {url}: {exc}"
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"{method} {url} failed with HTTP {response.status_code}.\n"
            f"Response: {response.text}"
        )

    if not response.content:
        return None

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError(
            f"{method} {url} returned a non-JSON response:\n{response.text}"
        ) from exc


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def list_documents(scope: str) -> list[dict]:
    """List documents in a scope, restricted to this run's conversation."""

    if scope not in {"temporary", "persistent"}:
        raise ValueError(f"Unsupported document scope: {scope}")

    params = {"scope": scope}

    if scope == "temporary":
        params["conversation_id"] = CONVERSATION_ID

    result = request_json("GET", "/documents", params=params)

    # Support either a direct list or a conventional wrapped response.
    if isinstance(result, list):
        return result

    if isinstance(result, dict):
        for key in ("documents", "items", "results"):
            if isinstance(result.get(key), list):
                return result[key]

    raise RuntimeError(
        "Unexpected document-list response. "
        f"Expected a list of documents, received: {result!r}"
    )


def find_test_documents(scope: str) -> list[dict]:
    """Find only documents carrying this run's unique filename."""

    documents = list_documents(scope)

    matches = [
        document
        for document in documents
        if document.get("filename") == TEST_FILENAME
    ]

    if scope == "temporary":
        matches = [
            document
            for document in matches
            if str(document.get("conversation_id")) == CONVERSATION_ID
        ]

    return matches


def find_one_test_document(scope: str) -> dict:
    """Return this run's unique test document, requiring exactly one match."""

    matches = find_test_documents(scope)

    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one {scope} test document named "
            f"{TEST_FILENAME!r}; found {len(matches)}. "
            "Check whether the upload step has run or whether this run "
            "has duplicate uploads."
        )

    document = matches[0]

    if document.get("status") != "indexed":
        raise AssertionError(
            f"{scope} test document is not indexed. "
            f"Current status: {document.get('status')!r}"
        )

    return document


def upload_test_document(scope: str) -> dict:
    """Upload the fixture under this run's unique filename."""

    if scope not in {"temporary", "persistent"}:
        raise ValueError(f"Unsupported document scope: {scope}")

    existing = find_test_documents(scope)

    if existing:
        raise RuntimeError(
            f"A {scope} test document for this run already exists. "
            "Refusing to create a duplicate. Use a fresh run ID or clean "
            "up this run's test document first."
        )

    data = {"scope": scope}

    if scope == "temporary":
        data["conversation_id"] = CONVERSATION_ID

    with FIXTURE_PATH.open("rb") as fixture:
        result = request_json(
            "POST",
            "/documents/upload",
            data=data,
            files={
                "file": (
                    TEST_FILENAME,
                    fixture,
                    "text/plain",
                )
            },
        )

    if not isinstance(result, dict) or not result.get("id"):
        raise RuntimeError(
            f"Upload did not return a document ID: {result!r}"
        )

    if result.get("filename") != TEST_FILENAME:
        raise RuntimeError(
            "The API returned an unexpected filename. "
            f"Expected {TEST_FILENAME!r}, got {result.get('filename')!r}."
        )

    if result.get("scope") != scope:
        raise RuntimeError(
            f"Upload scope mismatch: expected {scope!r}, "
            f"got {result.get('scope')!r}."
        )

    if scope == "temporary" and str(result.get("conversation_id")) != CONVERSATION_ID:
        raise RuntimeError(
            "The uploaded temporary document belongs to an unexpected conversation."
        )

    if result.get("status") != "indexed":
        raise RuntimeError(
            f"Document upload completed but indexing did not succeed. "
            f"Status: {result.get('status')!r}. "
            f"Document ID: {result.get('id')}"
        )

    return result


def delete_test_document(scope: str) -> int:
    """
    Delete only documents with this run's unique filename.

    Existing user documents are never selected by this function.
    """

    matches = find_test_documents(scope)
    deleted = 0

    for document in matches:
        document_id = document.get("id")

        if not document_id:
            raise RuntimeError(
                f"Cannot safely delete a document without an ID: {document!r}"
            )

        params = {"scope": scope}

        if scope == "temporary":
            params["conversation_id"] = CONVERSATION_ID

        request_json(
            "DELETE",
            f"/documents/{document_id}",
            params=params,
        )

        deleted += 1

    return deleted


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------

def ask_silvana(
    question: str,
    knowledge_scope: str,
    conversation_id: str = CONVERSATION_ID,
) -> dict:
    """Send a chat request using a selected knowledge scope and conversation."""

    if knowledge_scope not in {"none", "conversation", "library", "both"}:
        raise ValueError(
            f"Unsupported knowledge scope: {knowledge_scope}"
        )

    payload = {
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "provider": PROVIDER,
        "temperature": 0,
        "max_output_tokens": 2048,
        "conversation_id": conversation_id,
        "knowledge_scope": knowledge_scope,
    }

    result = request_json("POST", "/chat", json=payload)

    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected chat response: {result!r}")

    if "sources" not in result or not isinstance(result["sources"], list):
        raise RuntimeError(
            "Chat response did not contain the expected sources list."
        )

    return result


# ---------------------------------------------------------------------------
# Assertions and reporting
# ---------------------------------------------------------------------------

def assert_source_present(
    response: dict,
    document: dict,
    expected_scope: str,
) -> None:
    """Assert that a response cites the expected document."""

    document_id = str(document["id"])

    matching_sources = [
        source
        for source in response["sources"]
        if str(source.get("document_id")) == document_id
        and source.get("scope") == expected_scope
    ]

    if not matching_sources:
        raise AssertionError(
            f"Expected a citation for document {document_id} "
            f"in scope {expected_scope!r}, but none was returned.\n"
            f"Actual sources: {response['sources']!r}"
        )


def assert_source_absent(response: dict, document: dict) -> None:
    """Assert that a response does not cite the specified document."""

    document_id = str(document["id"])

    matching_sources = [
        source
        for source in response["sources"]
        if str(source.get("document_id")) == document_id
    ]

    if matching_sources:
        raise AssertionError(
            f"Document {document_id} unexpectedly appeared in the response "
            f"sources: {matching_sources!r}"
        )


def print_chat_result(label: str, response: dict) -> None:
    """Print the answer and its citations."""

    print(f"\n{'=' * 65}")
    print(label)
    print("=" * 65)
    print("\nANSWER:")
    print(response.get("content", "(No content returned)"))

    print(f"\nSOURCES ({len(response['sources'])}):")

    for source in response["sources"]:
        print(
            f"- {source.get('filename')} | "
            f"scope={source.get('scope')} | "
            f"document_id={source.get('document_id')} | "
            f"page={source.get('page_number')} | "
            f"chunk={source.get('chunk_index')}"
        )

    print()
