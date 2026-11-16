
from uuid import uuid4

from common import (
    CONVERSATION_ID,
    assert_source_absent,
    assert_source_present,
    ask_silvana,
    find_one_test_document,
    print_chat_result,
)


QUESTION = (
    "What is the unique verification phrase in the document "
    "available to this request? Give the exact phrase and cite it."
)


def main():
    print("\nSILVANA — DOCUMENT ISOLATION TESTS")
    print("=" * 55)

    temporary = find_one_test_document("temporary")
    persistent = find_one_test_document("persistent")

    # Test 1: Library scope must not expose conversation documents.
    print("\nOriginal conversation — library isolation")

    library_result = ask_silvana(
        QUESTION,
        knowledge_scope="library",
        conversation_id=CONVERSATION_ID,
    )

    print_chat_result(
        "LIBRARY SCOPE / ORIGINAL CONVERSATION",
        library_result,
    )

    assert_source_present(
        library_result, persistent, "persistent"
    )
    assert_source_absent(
        library_result, temporary
    )

    print("PASS: Library retrieval excludes temporary documents.")

    # Test 2: A different conversation must not access this run's
    # temporary document.
    other_conversation_id = str(uuid4())

    print("\nIndependent conversation isolation")
    print(f"Original conversation: {CONVERSATION_ID}")
    print(f"Other conversation:    {other_conversation_id}")

    isolated_result = ask_silvana(
        QUESTION,
        knowledge_scope="conversation",
        conversation_id=other_conversation_id,
    )

    print_chat_result(
        "CONVERSATION SCOPE / DIFFERENT CONVERSATION",
        isolated_result,
    )

    assert_source_absent(
        isolated_result, temporary
    )

    assert not any(
        source.get("scope") == "temporary"
        for source in isolated_result["sources"]
    ), (
        "A temporary document was returned to an unrelated conversation."
    )

    print("PASS: Different conversation cannot retrieve this run's temporary document.")

    print("\nALL DOCUMENT ISOLATION TESTS PASSED.")


if __name__ == "__main__":
    main()
