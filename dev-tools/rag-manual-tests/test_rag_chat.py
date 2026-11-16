
from common import (
    assert_source_present,
    find_one_test_document,
    ask_silvana,
    print_chat_result,
)


QUESTION = (
    "According to the document uploaded to this conversation, "
    "what is the unique verification phrase? "
    "Give the exact phrase and cite your source."
)

EXPECTED_PHRASE = "SILVER COMET 9182"


def main():
    print("\nSILVANA — TEMPORARY RAG RETRIEVAL TEST")
    print("=" * 55)

    temporary_document = find_one_test_document("temporary")

    response = ask_silvana(
        QUESTION,
        knowledge_scope="conversation",
    )

    print_chat_result("TEMPORARY CONVERSATION RETRIEVAL", response)

    assert EXPECTED_PHRASE.lower() in response["content"].lower(), (
        "The expected verification phrase was not found in the answer."
    )

    assert_source_present(
        response,
        temporary_document,
        "temporary",
    )

    print("PASS: Temporary document content and citation verified.")


if __name__ == "__main__":
    main()
