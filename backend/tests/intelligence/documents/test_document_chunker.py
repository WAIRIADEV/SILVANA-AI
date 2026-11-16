from uuid import uuid4

import pytest

from backend.app.intelligence.documents.chunker import DocumentChunker
from backend.app.intelligence.documents.extractor import (
    ExtractedSection,
    ExtractionResult,
)


def make_extraction(
    sections: tuple[ExtractedSection, ...],
    *,
    file_format: str = "txt",
) -> ExtractionResult:
    return ExtractionResult(
        filename=f"sample.{file_format}",
        format=file_format,
        sections=sections,
        page_count=None,
        word_count=sum(
            len(section.content.split())
            for section in sections
        ),
    )


def test_default_configuration():
    chunker = DocumentChunker()

    assert chunker.chunk_size == 800
    assert chunker.overlap == 120
    assert chunker.min_chunk_size == 100


@pytest.mark.parametrize(
    "configuration",
    [
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"chunk_size": 100, "overlap": 100},
        {"chunk_size": 100, "overlap": -1},
        {"chunk_size": 100, "min_chunk_size": 0},
        {"chunk_size": 100, "min_chunk_size": 101},
    ],
)
def test_invalid_configuration(configuration):
    with pytest.raises(ValueError):
        DocumentChunker(**configuration)


def test_short_document_produces_one_chunk():
    chunker = DocumentChunker()
    document_id = uuid4()
    content = "This is a short document."

    extraction = make_extraction(
        (ExtractedSection(content=content),)
    )

    chunks = chunker.chunk(extraction, document_id)

    assert len(chunks) == 1
    assert chunks[0].content == content
    assert chunks[0].document_id == document_id
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_number is None


def test_empty_extraction_produces_no_chunks():
    chunker = DocumentChunker()
    extraction = make_extraction(())

    assert chunker.chunk(extraction, uuid4()) == []


def test_whitespace_only_sections_produce_no_chunks():
    chunker = DocumentChunker()

    extraction = make_extraction(
        (
            ExtractedSection(content="   "),
            ExtractedSection(content="\n\n"),
        )
    )

    assert chunker.chunk(extraction, uuid4()) == []


def test_long_text_respects_chunk_size():
    chunker = DocumentChunker()
    extraction = make_extraction(
        (ExtractedSection(content="x" * 2000),)
    )

    chunks = chunker.chunk(extraction, uuid4())

    assert len(chunks) > 1
    assert all(
        len(chunk.content) <= chunker.chunk_size
        for chunk in chunks
    )


def test_character_overlap_is_preserved():
    chunker = DocumentChunker(
        chunk_size=800,
        overlap=120,
    )

    extraction = make_extraction(
        (ExtractedSection(content="x" * 2000),)
    )

    chunks = chunker.chunk(extraction, uuid4())

    assert len(chunks) == 3
    assert len(chunks[0].content) == 800
    assert len(chunks[1].content) == 800
    assert len(chunks[2].content) == 640

    assert chunks[0].content[-120:] == chunks[1].content[:120]
    assert chunks[1].content[-120:] == chunks[2].content[:120]


def test_chunk_indexes_are_sequential():
    chunker = DocumentChunker()

    extraction = make_extraction(
        (ExtractedSection(content="x" * 2000),)
    )

    chunks = chunker.chunk(extraction, uuid4())

    assert [
        chunk.chunk_index for chunk in chunks
    ] == list(range(len(chunks)))


def test_chunk_ids_are_unique():
    chunker = DocumentChunker()

    extraction = make_extraction(
        (ExtractedSection(content="x" * 2000),)
    )

    chunks = chunker.chunk(extraction, uuid4())
    identifiers = [chunk.id for chunk in chunks]

    assert len(identifiers) == len(set(identifiers))


def test_all_chunks_reference_the_same_document():
    chunker = DocumentChunker()
    document_id = uuid4()

    extraction = make_extraction(
        (ExtractedSection(content="x" * 2000),)
    )

    chunks = chunker.chunk(extraction, document_id)

    assert all(
        chunk.document_id == document_id
        for chunk in chunks
    )


def test_pdf_page_numbers_are_preserved():
    chunker = DocumentChunker()

    extraction = make_extraction(
        (
            ExtractedSection(
                content="Page one content " * 80,
                page_number=1,
            ),
            ExtractedSection(
                content="Page two content " * 80,
                page_number=2,
            ),
        ),
        file_format="pdf",
    )

    chunks = chunker.chunk(extraction, uuid4())

    assert chunks
    assert all(
        chunk.page_number in {1, 2}
        for chunk in chunks
    )

    assert any(chunk.page_number == 1 for chunk in chunks)
    assert any(chunk.page_number == 2 for chunk in chunks)


def test_pdf_sections_on_same_page_are_combined():
    chunker = DocumentChunker()

    extraction = make_extraction(
        (
            ExtractedSection(
                content="First paragraph.",
                page_number=1,
            ),
            ExtractedSection(
                content="Second paragraph.",
                page_number=1,
            ),
        ),
        file_format="pdf",
    )

    chunks = chunker.chunk(extraction, uuid4())

    assert len(chunks) == 1
    assert "First paragraph." in chunks[0].content
    assert "Second paragraph." in chunks[0].content
    assert chunks[0].page_number == 1


def test_pdf_pages_are_not_combined():
    chunker = DocumentChunker()

    extraction = make_extraction(
        (
            ExtractedSection(
                content="First page content.",
                page_number=1,
            ),
            ExtractedSection(
                content="Second page content.",
                page_number=2,
            ),
        ),
        file_format="pdf",
    )

    chunks = chunker.chunk(extraction, uuid4())

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2


def test_non_pdf_sections_are_combined():
    chunker = DocumentChunker()

    extraction = make_extraction(
        (
            ExtractedSection(content="First paragraph."),
            ExtractedSection(content="Second paragraph."),
            ExtractedSection(content="Third paragraph."),
        ),
        file_format="docx",
    )

    chunks = chunker.chunk(extraction, uuid4())

    assert len(chunks) == 1

    content = chunks[0].content

    assert "First paragraph." in content
    assert "Second paragraph." in content
    assert "Third paragraph." in content


def test_natural_boundaries_are_preferred():
    chunker = DocumentChunker(
        chunk_size=120,
        overlap=20,
        min_chunk_size=40,
    )

    first_sentence = "First sentence. " * 6
    second_sentence = "Second sentence. " * 6

    extraction = make_extraction(
        (
            ExtractedSection(
                content=first_sentence + second_sentence
            ),
        )
    )

    chunks = chunker.chunk(extraction, uuid4())

    assert len(chunks) > 1
    assert all(
        len(chunk.content) <= 120
        for chunk in chunks
    )


def test_text_without_natural_boundaries_is_split():
    chunker = DocumentChunker(
        chunk_size=100,
        overlap=20,
        min_chunk_size=30,
    )

    extraction = make_extraction(
        (ExtractedSection(content="a" * 250),)
    )

    chunks = chunker.chunk(extraction, uuid4())

    assert len(chunks) > 1
    assert all(
        len(chunk.content) <= 100
        for chunk in chunks
    )
