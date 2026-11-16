
from common import (
    find_one_test_document,
    print_chat_result,
    upload_test_document,
)


def main():
    print("\nSILVANA — PERSISTENT DOCUMENT UPLOAD TEST")
    print("=" * 55)

    uploaded = upload_test_document("persistent")

    print("\nUpload successful.")
    print(f"Document ID: {uploaded['id']}")
    print(f"Filename:    {uploaded['filename']}")
    print(f"Scope:       {uploaded['scope']}")
    print(f"Status:      {uploaded['status']}")

    document = find_one_test_document("persistent")

    assert str(document["id"]) == str(uploaded["id"]), (
        "The discovered persistent document ID does not match "
        "the uploaded document."
    )

    print("\nPASS: Persistent document uploaded and discovered successfully.")


if __name__ == "__main__":
    main()
