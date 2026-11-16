
from io import BytesIO

import pymupdf
import pytest
from docx import Document

from backend.app.intelligence.documents.extractor import (
    DocumentExtractionError,
    DocumentExtractor,
    UnsupportedDocumentTypeError,
)


@pytest.fixture
def extractor():
    return DocumentExtractor()


def create_pdf(*pages: str) -> bytes:
    """Generate an in-memory PDF with the supplied page contents."""

    document = pymupdf.open()

    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)

    content = document.tobytes()
    document.close()

    return content


def create_docx() -> bytes:
    """Generate an in-memory DOCX with paragraphs and a table."""

    document = Document()

    document.add_paragraph("SILVANA document extraction test.")

    document.add_paragraph(
        "This paragraph verifies Word text extraction."
    )

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Role"
    table.cell(1, 0).text = "SILVANA"
    table.cell(1, 1).text = "AI Assistant"

    output = BytesIO()
    document.save(output)

    return output.getvalue()


def create_encrypted_pdf() -> bytes:
    """Generate a password-protected PDF."""

    document = pymupdf.open()

    page = document.new_page()
    page.insert_text((72, 72), "Protected document.")

    content = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )

    document.close()

    return content


def test_extract_plain_text(extractor):
    result = extractor.extract(
        "notes.txt",
        b"SILVANA extracts plain text.",
    )

    assert result.filename == "notes.txt"
    assert result.format == "txt"
    assert result.page_count is None
    assert result.word_count == 4
    assert len(result.sections) == 1
    assert result.sections[0].content == (
        "SILVANA extracts plain text."
    )
    assert result.sections[0].page_number is None


def test_extract_text_with_utf8_bom(extractor):
    content = "\ufeffSILVANA UTF-8 document.".encode("utf-8")

    result = extractor.extract("notes.txt", content)

    assert result.sections[0].content == (
        "SILVANA UTF-8 document."
    )


def test_extract_pdf_text(extractor):
    content = create_pdf("SILVANA PDF extraction test.")

    result = extractor.extract("report.pdf", content)

    assert result.format == "pdf"
    assert result.page_count == 1
    assert result.word_count == 4
    assert result.sections[0].page_number == 1
    assert "SILVANA PDF extraction test." in (
        result.sections[0].content
    )


def test_extract_multipage_pdf_with_page_numbers(extractor):
    content = create_pdf(
        "First page content.",
        "Second page content.",
        "Third page content.",
    )

    result = extractor.extract("report.pdf", content)

    assert result.page_count == 3
    assert len(result.sections) == 3

    assert [
        section.page_number for section in result.sections
    ] == [1, 2, 3]


def test_pdf_with_blank_pages_preserves_page_numbers(extractor):
    content = create_pdf(
        "First page.",
        "",
        "Third page.",
    )

    result = extractor.extract("report.pdf", content)

    assert result.page_count == 3
    assert [
        section.page_number for section in result.sections
    ] == [1, 3]


def test_extract_docx_paragraphs_and_tables(extractor):
    result = extractor.extract(
        "report.docx",
        create_docx(),
    )

    extracted_text = "\n".join(
        section.content for section in result.sections
    )

    assert result.format == "docx"
    assert result.page_count is None
    assert "SILVANA document extraction test." in extracted_text
    assert "This paragraph verifies Word text extraction." in (
        extracted_text
    )
    assert "SILVANA | AI Assistant" in extracted_text
    assert result.word_count > 0


def test_extract_docx_preserves_document_order(extractor):
    result = extractor.extract(
        "report.docx",
        create_docx(),
    )

    contents = [
        section.content for section in result.sections
    ]

    assert contents[0] == "SILVANA document extraction test."
    assert contents[1] == (
        "This paragraph verifies Word text extraction."
    )
    assert "Name | Role" in contents[2]
    assert "SILVANA | AI Assistant" in contents[2]


def test_unsupported_file_extension(extractor):
    with pytest.raises(UnsupportedDocumentTypeError):
        extractor.extract("image.png", b"not an image")


def test_missing_file_extension(extractor):
    with pytest.raises(UnsupportedDocumentTypeError):
        extractor.extract("document", b"some content")


def test_empty_content_is_rejected(extractor):
    with pytest.raises(DocumentExtractionError):
        extractor.extract("notes.txt", b"")


def test_non_bytes_content_is_rejected(extractor):
    with pytest.raises(TypeError):
        extractor.extract(
            "notes.txt",
            "not bytes",  # type: ignore[arg-type]
        )


def test_invalid_utf8_text_is_rejected(extractor):
    with pytest.raises(
        DocumentExtractionError,
        match="valid UTF-8",
    ):
        extractor.extract("notes.txt", b"\xff\xfe\xfa")


def test_corrupted_pdf_is_rejected(extractor):
    with pytest.raises(DocumentExtractionError):
        extractor.extract(
            "broken.pdf",
            b"This is not a valid PDF.",
        )


def test_corrupted_docx_is_rejected(extractor):
    with pytest.raises(DocumentExtractionError):
        extractor.extract(
            "broken.docx",
            b"This is not a valid DOCX.",
        )


def test_password_protected_pdf_is_rejected(extractor):
    content = create_encrypted_pdf()

    with pytest.raises(
        DocumentExtractionError,
        match="Password-protected",
    ):
        extractor.extract("protected.pdf", content)


def test_pdf_without_extractable_text_is_rejected(extractor):
    content = create_pdf("")

    with pytest.raises(
        DocumentExtractionError,
        match="No readable text",
    ):
        extractor.extract("blank.pdf", content)


def test_text_without_readable_words_is_rejected(extractor):
    with pytest.raises(
        DocumentExtractionError,
        match="No readable text",
    ):
        extractor.extract("blank.txt", b"   \n\t  ")


def test_filename_is_preserved(extractor):
    result = extractor.extract(
        "annual_report_2026.txt",
        b"Annual report content.",
    )

    assert result.filename == "annual_report_2026.txt"


def test_file_extension_is_case_insensitive(extractor):
    result = extractor.extract(
        "REPORT.TXT",
        b"Uppercase extension test.",
    )

    assert result.format == "txt"


def test_multiple_docx_sections_are_returned(extractor):
    result = extractor.extract(
        "report.docx",
        create_docx(),
    )

    assert len(result.sections) == 3
    assert all(section.content.strip() for section in result.sections)
