from common import delete_test_document, find_test_documents


def main():
    print("\nSILVANA - RAG TEST CLEANUP")
    print("=" * 55)

    print("Cleanup is restricted to the current test run.")
    print("Unrelated knowledge-library documents will not be targeted.\n")

    temporary_deleted = delete_test_document("temporary")
    persistent_deleted = delete_test_document("persistent")

    print(f"Temporary documents deleted:  {temporary_deleted}")
    print(f"Persistent documents deleted: {persistent_deleted}")

    remaining_temporary = find_test_documents("temporary")
    remaining_persistent = find_test_documents("persistent")

    if remaining_temporary or remaining_persistent:
        raise AssertionError(
            "Cleanup incomplete. Documents belonging to this run remain.\n"
            f"Temporary: {remaining_temporary}\n"
            f"Persistent: {remaining_persistent}"
        )

    print("\nPASS: All documents belonging to this test run were deleted.")
    print("PASS: Documents outside this run were not targeted.")


if __name__ == "__main__":
    main()
