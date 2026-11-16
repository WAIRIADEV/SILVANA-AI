
# SILVANA RAG Manual Integration Tests

A collection of manual integration tests for verifying SILVANA's document
ingestion, retrieval, citation, knowledge-scope, and conversation-isolation
behavior.

These tests communicate with a running SILVANA backend through its REST API.

## Prerequisites

- Python virtual environment activated.
- SILVANA backend running locally.
- Backend dependencies installed.
- `requests` installed in the active Python environment.
- A working configured LLM provider and API credentials.

Default API base URL:

`http://127.0.0.1:8000/api/v1`

## Files

| File | Purpose |
|---|---|
| `common.py` | Shared API configuration, requests, document helpers and assertions |
| `silvana_rag_test.txt` | Synthetic document used for integration tests |
| `test_persistent_upload.py` | Persistent document upload |
| `upload_rag_test.py` | Temporary conversation document upload |
| `test_rag_chat.py` | Temporary document retrieval and citation |
| `test_persistent_retrieval.py` | Persistent document retrieval and citation |
| `test_rag_scopes.py` | Tests conversation, library, both and none scopes |
| `test_rag_isolation.py` | Tests library and cross-conversation isolation |
| `cleanup_rag_test.py` | Safely removes documents belonging to the current run |

## 1. Configure a test run

Open PowerShell in the SILVANA project root.

Generate unique IDs:

```powershell
$env:SILVANA_TEST_RUN_ID = [guid]::NewGuid().ToString()
$env:SILVANA_TEST_CONVERSATION_ID = [guid]::NewGuid().ToString()
```

Optional configuration:

```powershell
$env:SILVANA_API_BASE_URL = "http://127.0.0.1:8000/api/v1"
$env:SILVANA_TEST_PROVIDER = "groq"
```

Keep the same PowerShell session throughout the test run.

The run ID is incorporated into the uploaded test filename. This allows
the toolkit to identify its own documents without targeting unrelated
knowledge-library documents.

## 2. Upload test documents

Run these scripts from the project root:

```powershell
python .\dev-tools\rag-manual-tests\test_persistent_upload.py

python .\dev-tools\rag-manual-tests\upload_rag_test.py
```

Both uploads must complete successfully and return indexed documents.

Do not rerun an upload with the same run ID after it has succeeded.
The toolkit intentionally prevents duplicate uploads.

## 3. Test retrieval

```powershell
python .\dev-tools\rag-manual-tests\test_rag_chat.py

python .\dev-tools\rag-manual-tests\test_persistent_retrieval.py
```

These tests verify answer content and source citations.

The shared chat helper uses a 2048-token output budget to give reasoning
models sufficient room to produce a final answer.

## 4. Test knowledge scopes

```powershell
python .\dev-tools\rag-manual-tests\test_rag_scopes.py
```

Expected behavior:

| Scope | Temporary documents | Persistent documents |
|---|---|---|
| `conversation` | Included | Excluded |
| `library` | Excluded | Included |
| `both` | Included | Included |
| `none` | Excluded | Excluded |

The tests check actual document IDs and source scopes, not just answer text.

## 5. Test conversation isolation

```powershell
python .\dev-tools\rag-manual-tests\test_rag_isolation.py
```

Verifies that:
- Library retrieval does not include temporary documents.
- A different conversation cannot retrieve the original conversation's
  temporary documents.

## 6. Clean up test documents

Only after all tests have completed successfully:

```powershell
python .\dev-tools\rag-manual-tests\cleanup_rag_test.py
```

Cleanup targets documents with the exact filename generated from the
current run ID.

It removes the current run's temporary and persistent test documents
and verifies that they are no longer present.

It does not target pre-existing persistent knowledge-library documents.

## 7. Run a fresh test cycle

For another complete test cycle, generate new IDs:

```powershell
$env:SILVANA_TEST_RUN_ID = [guid]::NewGuid().ToString()
$env:SILVANA_TEST_CONVERSATION_ID = [guid]::NewGuid().ToString()
```

Then repeat the workflow from the upload stage.

## Important notes

- These are manual API integration tests, not pytest unit tests.
- Run scripts from the SILVANA project root.
- The backend must remain running while tests execute.
- Keep the same environment variables throughout one test cycle.
- Never manually delete unrelated persistent documents as part of test cleanup.
- Do not commit API credentials, tokens, or `.env` files.
