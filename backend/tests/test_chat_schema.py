
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.api.schemas.chat import ChatRequest


def make_request(**overrides):
    data = {
        "messages": [
            {
                "role": "user",
                "content": "Hello SILVANA",
            }
        ]
    }
    data.update(overrides)
    return ChatRequest.model_validate(data)


def test_chat_request_defaults_to_no_document_retrieval():
    request = make_request()

    assert request.knowledge_scope == "none"
    assert request.conversation_id is None


@pytest.mark.parametrize(
    "scope",
    ["none", "library"],
)
def test_chat_request_allows_scopes_without_conversation_id(scope):
    request = make_request(knowledge_scope=scope)

    assert request.knowledge_scope == scope
    assert request.conversation_id is None


@pytest.mark.parametrize(
    "scope",
    ["conversation", "both"],
)
def test_chat_request_requires_conversation_id_for_temporary_scope(
    scope,
):
    with pytest.raises(ValidationError):
        make_request(knowledge_scope=scope)


@pytest.mark.parametrize(
    "scope",
    ["conversation", "both"],
)
def test_chat_request_accepts_valid_conversation_id(scope):
    conversation_id = uuid4()

    request = make_request(
        knowledge_scope=scope,
        conversation_id=str(conversation_id),
    )

    assert request.knowledge_scope == scope
    assert request.conversation_id == conversation_id


def test_chat_request_rejects_invalid_conversation_id():
    with pytest.raises(ValidationError):
        make_request(
            knowledge_scope="conversation",
            conversation_id="not-a-uuid",
        )


def test_chat_request_rejects_unsupported_knowledge_scope():
    with pytest.raises(ValidationError):
        make_request(knowledge_scope="everything")
