
from unittest.mock import Mock

import pytest

from backend.app.intelligence.embeddings.fastembed import FastEmbedProvider


DIMENSIONS = 384


def make_model(
    *,
    document_vectors=None,
    query_vectors=None,
):
    model = Mock()

    model.embed.return_value = (
        document_vectors
        if document_vectors is not None
        else [[0.1] * DIMENSIONS]
    )

    model.query_embed.return_value = (
        query_vectors
        if query_vectors is not None
        else [[0.2] * DIMENSIONS]
    )

    return model


def make_provider(model=None, **kwargs):
    return FastEmbedProvider(
        model=model,
        **kwargs,
    )


def test_default_configuration():
    provider = make_provider()

    assert provider.provider_name == "fastembed"
    assert provider.model_name == "BAAI/bge-small-en-v1.5"
    assert provider.dimensions == DIMENSIONS
    assert provider.batch_size == 32
    assert provider.threads == 2


@pytest.mark.parametrize(
    "configuration",
    [
        {"model_name": ""},
        {"model_name": "   "},
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": True},
        {"batch_size": 1.5},
        {"threads": 0},
        {"threads": -1},
        {"threads": True},
        {"threads": 1.5},
    ],
)
def test_invalid_configuration(configuration):
    with pytest.raises(ValueError):
        make_provider(**configuration)


def test_unsupported_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported FastEmbed model"):
        make_provider(model_name="unknown/model")


def test_model_is_loaded_lazily():
    model = make_model()
    provider = make_provider(model=model)

    assert provider._model is model


def test_empty_document_batch_returns_empty_list():
    model = make_model()
    provider = make_provider(model=model)

    assert provider.embed_documents([]) == []

    model.embed.assert_not_called()


def test_document_embeddings_are_generated():
    model = make_model(
        document_vectors=[
            [0.1] * DIMENSIONS,
            [0.2] * DIMENSIONS,
        ]
    )

    provider = make_provider(
        model=model,
        batch_size=16,
    )

    vectors = provider.embed_documents(
        [
            "SILVANA is an AI assistant.",
            "Document retrieval uses embeddings.",
        ]
    )

    assert len(vectors) == 2
    assert all(len(vector) == DIMENSIONS for vector in vectors)
    assert all(type(value) is float for value in vectors[0])

    model.embed.assert_called_once_with(
        [
            "SILVANA is an AI assistant.",
            "Document retrieval uses embeddings.",
        ],
        batch_size=16,
    )


def test_query_embedding_is_generated():
    model = make_model(
        query_vectors=[[0.3] * DIMENSIONS]
    )

    provider = make_provider(model=model)

    vector = provider.embed_query("What is SILVANA?")

    assert len(vector) == DIMENSIONS
    assert all(type(value) is float for value in vector)

    model.query_embed.assert_called_once_with("What is SILVANA?")


@pytest.mark.parametrize(
    "texts",
    [
        "not a list",
        None,
        123,
    ],
)
def test_document_input_must_be_a_list(texts):
    provider = make_provider(model=make_model())

    with pytest.raises(TypeError):
        provider.embed_documents(texts)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\n",
        None,
        123,
    ],
)
def test_invalid_document_text_is_rejected(text):
    provider = make_provider(model=make_model())

    with pytest.raises((TypeError, ValueError)):
        provider.embed_documents([text])


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\n",
    ],
)
def test_empty_query_is_rejected(text):
    provider = make_provider(model=make_model())

    with pytest.raises(ValueError):
        provider.embed_query(text)


@pytest.mark.parametrize(
    "text",
    [
        None,
        123,
    ],
)
def test_non_string_query_is_rejected(text):
    provider = make_provider(model=make_model())

    with pytest.raises(TypeError):
        provider.embed_query(text)


def test_wrong_document_vector_count_is_rejected():
    model = make_model(
        document_vectors=[[0.1] * DIMENSIONS]
    )

    provider = make_provider(model=model)

    with pytest.raises(
        RuntimeError,
        match="unexpected number of vectors",
    ):
        provider.embed_documents(["First", "Second"])


def test_wrong_query_vector_count_is_rejected():
    model = make_model(
        query_vectors=[
            [0.1] * DIMENSIONS,
            [0.2] * DIMENSIONS,
        ]
    )

    provider = make_provider(model=model)

    with pytest.raises(
        RuntimeError,
        match="unexpected number of query vectors",
    ):
        provider.embed_query("Test query")


def test_wrong_vector_dimensions_are_rejected():
    model = make_model(
        document_vectors=[[0.1] * 100]
    )

    provider = make_provider(model=model)

    with pytest.raises(ValueError, match="Expected 384 dimensions"):
        provider.embed_documents(["Test document"])


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_vector_values_are_rejected(invalid_value):
    vector = [0.1] * DIMENSIONS
    vector[0] = invalid_value

    model = make_model(
        document_vectors=[vector]
    )

    provider = make_provider(model=model)

    with pytest.raises(ValueError, match="must be finite"):
        provider.embed_documents(["Test document"])


def test_non_numeric_vector_values_are_rejected():
    vector = [0.1] * DIMENSIONS
    vector[0] = "invalid"

    model = make_model(
        document_vectors=[vector]
    )

    provider = make_provider(model=model)

    with pytest.raises(TypeError, match="must be numeric"):
        provider.embed_documents(["Test document"])


def test_boolean_vector_values_are_rejected():
    vector = [0.1] * DIMENSIONS
    vector[0] = True

    model = make_model(
        document_vectors=[vector]
    )

    provider = make_provider(model=model)

    with pytest.raises(TypeError, match="must be numeric"):
        provider.embed_documents(["Test document"])


def test_document_embedding_failure_is_wrapped():
    model = make_model()
    model.embed.side_effect = OSError("Model failure")

    provider = make_provider(model=model)

    with pytest.raises(
        RuntimeError,
        match="document embedding failed",
    ) as error:
        provider.embed_documents(["Test document"])

    assert isinstance(error.value.__cause__, OSError)


def test_query_embedding_failure_is_wrapped():
    model = make_model()
    model.query_embed.side_effect = OSError("Model failure")

    provider = make_provider(model=model)

    with pytest.raises(
        RuntimeError,
        match="query embedding failed",
    ) as error:
        provider.embed_query("Test query")

    assert isinstance(error.value.__cause__, OSError)


def test_model_initialization_failure_is_wrapped(monkeypatch):
    def fail_initialization(**kwargs):
        raise OSError("Model initialization failed")

    monkeypatch.setattr(
        "backend.app.intelligence.embeddings.fastembed.TextEmbedding",
        type(
            "FailingTextEmbedding",
            (),
            {
                "list_supported_models": staticmethod(
                    lambda: [
                        {
                            "model": "BAAI/bge-small-en-v1.5",
                            "dim": DIMENSIONS,
                        }
                    ]
                ),
                "__new__": staticmethod(
                    lambda cls, **kwargs: fail_initialization(**kwargs)
                ),
            },
        ),
    )

    provider = make_provider()

    with pytest.raises(
        RuntimeError,
        match="model initialization failed",
    ) as error:
        provider.embed_query("Test query")

    assert isinstance(error.value.__cause__, OSError)
