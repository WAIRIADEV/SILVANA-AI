
from common import (
    assert_source_present,
    find_one_test_document,
    ask_silvana,
    print_chat_result,
)


QUESTION = (
    "According to the persistent knowledge library, "
    "what is the unique verification phrase? "
    "Give the exact phrase and cite your source."
)

EXPECTED_PHRASE = "SILVER COMET 9182"


def main():
    print("\nSILVANA — PERSISTENT RAG RETRIEVAL TEST")
    print("=" * 55)

    persistent_document = find_one_test_document("persistent")

    response = ask_silvana(
        QUESTION,
        knowledge_scope="library",
    )

    print_chat_result("PERSISTENT LIBRARY RETRIEVAL", response)

    assert EXPECTED_PHRASE.lower() in response["content"].lower(), (
        "The expected verification phrase was not found in the answer."
    )

    assert_source_present(
        response,
        persistent_document,
        "persistent",
    )

    print("PASS: Persistent document content and citation verified.")


if __name__ == "__main__":
    main()
