
from common import (
    CONVERSATION_ID,
    find_one_test_document,
    upload_test_document,
)


def main():
    print("\nSILVANA — TEMPORARY DOCUMENT UPLOAD TEST")
    print("=" * 55)

    print(f"Conversation ID: {CONVERSATION_ID}")

    uploaded = upload_test_document("temporary")

    print("\nUpload successful.")
    print(f"Document ID: {uploaded['id']}")
    print(f"Filename:    {uploaded['filename']}")
    print(f"Scope:       {uploaded['scope']}")
    print(f"Status:      {uploaded['status']}")
    print(f"Conversation: {uploaded['conversation_id']}")

    document = find_one_test_document("temporary")

    assert str(document["id"]) == str(uploaded["id"]), (
        "The discovered temporary document ID does not match "
        "the uploaded document."
    )

    assert str(document["conversation_id"]) == CONVERSATION_ID, (
        "The temporary document belongs to the wrong conversation."
    )

    print("\nPASS: Temporary document uploaded and discovered successfully.")


if __name__ == "__main__":
    main()
