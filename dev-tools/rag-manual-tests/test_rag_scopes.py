
from common import (
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
    print("\nSILVANA — KNOWLEDGE SCOPE TESTS")
    print("=" * 55)

    temporary = find_one_test_document("temporary")
    persistent = find_one_test_document("persistent")

    # 1. Conversation scope
    conversation_result = ask_silvana(
        QUESTION,
        "conversation",
    )
    print_chat_result("1. CONVERSATION SCOPE", conversation_result)

    assert_source_present(
        conversation_result, temporary, "temporary"
    )
    assert_source_absent(
        conversation_result, persistent
    )
    print("PASS: Conversation scope includes temporary document only.")

    # 2. Library scope
    library_result = ask_silvana(
        QUESTION,
        "library",
    )
    print_chat_result("2. LIBRARY SCOPE", library_result)

    assert_source_present(
        library_result, persistent, "persistent"
    )
    assert_source_absent(
        library_result, temporary
    )
    print("PASS: Library scope includes persistent document, not temporary.")

    # 3. Both scopes
    both_result = ask_silvana(
        QUESTION,
        "both",
    )
    print_chat_result("3. BOTH SCOPES", both_result)

    assert_source_present(
        both_result, temporary, "temporary"
    )
    assert_source_present(
        both_result, persistent, "persistent"
    )
    print("PASS: Both scope includes both test documents.")

    # 4. No knowledge
    none_result = ask_silvana(
        QUESTION,
        "none",
    )
    print_chat_result("4. NO KNOWLEDGE SCOPE", none_result)

    assert_source_absent(
        none_result, temporary
    )
    assert_source_absent(
        none_result, persistent
    )
    print("PASS: None scope excludes both test documents.")

    print("\nALL FOUR KNOWLEDGE SCOPE TESTS PASSED.")


if __name__ == "__main__":
    main()
