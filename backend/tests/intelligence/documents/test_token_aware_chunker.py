
from uuid import uuid4

import pytest

from backend.app.intelligence.documents.chunker import DocumentChunker
from backend.app.intelligence.documents.extractor import (
    ExtractedSection,
    ExtractionResult,
)


def make_extraction(content: str, file_format: str = "txt"):
    return ExtractionResult(
        filename=f"sample.{file_format}",
        format=file_format,
        sections=(
            ExtractedSection(
                content=content,
                page_number=1 if file_format == "pdf" else None,
            ),
        ),
        page_count=1 if file_format == "pdf" else None,
        word_count=len(content.split()),
    )


def character_counter(text: str) -> int:
    return len(text)


def word_counter(text: str) -> int:
    return len(text.split()) + 2


def test_token_aware_configuration():
    chunker = DocumentChunker(
        token_counter=word_counter,
        max_tokens=450,
        token_overlap=60,
    )

    assert chunker.max_tokens == 450
    assert chunker.token_overlap == 60
    assert chunker.token_counter is word_counter


def test_token_aware_chunking_respects_budget():
    chunker = DocumentChunker(
        token_counter=word_counter,
        max_tokens=100,
        token_overlap=15,
    )

    content = " ".join(["SILVANA"] * 1000)
    extraction = make_extraction(content)

    chunks = chunker.chunk(extraction, uuid4())

    assert len(chunks) > 1
    assert all(
        word_counter(chunk.content) <= 100
        for chunk in chunks
    )


def test_token_overlap_is_preserved():
    chunker = DocumentChunker(
        token_counter=character_counter,
        max_tokens=100,
        token_overlap=20,
    )

    extraction = make_extraction("x" * 250)
    chunks = chunker.chunk(extraction, uuid4())

    assert len(chunks) == 3

    assert len(chunks[0].content) == 100
    assert len(chunks[1].content) == 100
    assert len(chunks[2].content) == 90

    assert chunks[0].content[-20:] == chunks[1].content[:20]
    assert chunks[1].content[-20:] == chunks[2].content[:20]


def test_token_aware_chunks_preserve_pdf_page_numbers():
    chunker = DocumentChunker(
        token_counter=word_counter,
        max_tokens=50,
        token_overlap=5,
    )

    extraction = make_extraction(
        " ".join(["document"] * 300),
        file_format="pdf",
    )

    chunks = chunker.chunk(extraction, uuid4())

    assert len(chunks) > 1
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_token_aware_chunks_have_sequential_indexes():
    chunker = DocumentChunker(
        token_counter=word_counter,
        max_tokens=50,
        token_overlap=5,
    )

    extraction = make_extraction(
        " ".join(["document"] * 300)
    )

    chunks = chunker.chunk(extraction, uuid4())

    assert [
        chunk.chunk_index for chunk in chunks
    ] == list(range(len(chunks)))


def test_token_counter_must_return_integer():
    chunker = DocumentChunker(
        token_counter=lambda text: 10.5,
        max_tokens=20,
        token_overlap=2,
    )

    extraction = make_extraction("word " * 100)

    with pytest.raises(TypeError, match="integer"):
        chunker.chunk(extraction, uuid4())


def test_token_counter_cannot_return_negative_count():
    chunker = DocumentChunker(
        token_counter=lambda text: -1,
        max_tokens=20,
        token_overlap=2,
    )

    extraction = make_extraction("word " * 100)

    with pytest.raises(ValueError, match="negative"):
        chunker.chunk(extraction, uuid4())


@pytest.mark.parametrize(
    "configuration",
    [
        {"max_tokens": 2},
        {"max_tokens": 100, "token_overlap": -1},
        {"max_tokens": 100, "token_overlap": 100},
    ],
)
def test_invalid_token_configuration(configuration):
    with pytest.raises(ValueError):
        DocumentChunker(
            token_counter=word_counter,
            **configuration,
        )
