
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pymupdf
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


class DocumentExtractionError(Exception):
    """Raised when document contents cannot be extracted."""


class UnsupportedDocumentTypeError(DocumentExtractionError):
    """Raised when a document format is not supported."""


@dataclass(frozen=True, slots=True)
class ExtractedSection:
    """A section of extracted document text."""

    content: str
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Structured output from document extraction."""

    filename: str
    format: str
    sections: tuple[ExtractedSection, ...]
    page_count: int | None
    word_count: int


class DocumentExtractor:
    """Extract text from supported document formats."""

    SUPPORTED_FORMATS = {".pdf", ".docx", ".txt"}

    def extract(
        self,
        filename: str,
        content: bytes,
    ) -> ExtractionResult:
        """Extract document contents based on the filename extension."""

        if not isinstance(content, bytes):
            raise TypeError("Document content must be bytes.")

        if not content:
            raise DocumentExtractionError(
                "Document content cannot be empty."
            )

        extension = Path(filename).suffix.lower()

        if extension not in self.SUPPORTED_FORMATS:
            raise UnsupportedDocumentTypeError(
                f"Unsupported document format: {extension or 'unknown'}."
            )

        if extension == ".pdf":
            return self._extract_pdf(filename, content)

        if extension == ".docx":
            return self._extract_docx(filename, content)

        return self._extract_txt(filename, content)

    @staticmethod
    def _build_result(
        filename: str,
        file_format: str,
        sections: list[ExtractedSection],
        page_count: int | None,
    ) -> ExtractionResult:
        """Build a consistent extraction result."""

        if not sections:
            raise DocumentExtractionError(
                "No readable text was found in the document."
            )

        word_count = sum(
            len(section.content.split())
            for section in sections
        )

        if word_count == 0:
            raise DocumentExtractionError(
                "No readable text was found in the document."
            )

        return ExtractionResult(
            filename=filename,
            format=file_format,
            sections=tuple(sections),
            page_count=page_count,
            word_count=word_count,
        )

    def _extract_pdf(
        self,
        filename: str,
        content: bytes,
    ) -> ExtractionResult:
        """Extract readable text while preserving PDF page numbers."""

        try:
            with pymupdf.open(
                stream=content,
                filetype="pdf",
            ) as document:

                if document.needs_pass:
                    raise DocumentExtractionError(
                        "Password-protected PDFs are not supported."
                    )

                sections = []

                for page_index, page in enumerate(document):
                    text = page.get_text("text").strip()

                    if text:
                        sections.append(
                            ExtractedSection(
                                content=text,
                                page_number=page_index + 1,
                            )
                        )

                return self._build_result(
                    filename=filename,
                    file_format="pdf",
                    sections=sections,
                    page_count=document.page_count,
                )

        except DocumentExtractionError:
            raise

        except Exception as exc:
            raise DocumentExtractionError(
                "The PDF could not be read."
            ) from exc

    def _extract_docx(
        self,
        filename: str,
        content: bytes,
    ) -> ExtractionResult:
        """Extract Word paragraphs and tables in document order."""

        try:
            document = Document(BytesIO(content))
            sections = []

            for element in document.element.body.iterchildren():

                if element.tag == qn("w:p"):
                    paragraph = Paragraph(
                        element,
                        document,
                    )

                    text = paragraph.text.strip()

                    if text:
                        sections.append(
                            ExtractedSection(content=text)
                        )

                elif element.tag == qn("w:tbl"):
                    table = Table(
                        element,
                        document,
                    )

                    rows = []

                    for row in table.rows:
                        cells = [
                            cell.text.strip().replace("\n", " ")
                            for cell in row.cells
                        ]

                        rows.append(" | ".join(cells))

                    table_text = "\n".join(
                        row for row in rows if row.strip()
                    )

                    if table_text:
                        sections.append(
                            ExtractedSection(
                                content=table_text
                            )
                        )

            return self._build_result(
                filename=filename,
                file_format="docx",
                sections=sections,
                page_count=None,
            )

        except DocumentExtractionError:
            raise

        except Exception as exc:
            raise DocumentExtractionError(
                "The Word document could not be read."
            ) from exc

    def _extract_txt(
        self,
        filename: str,
        content: bytes,
    ) -> ExtractionResult:
        """Decode UTF-8 plain-text documents."""

        try:
            text = content.decode("utf-8-sig").strip()

        except UnicodeDecodeError as exc:
            raise DocumentExtractionError(
                "The text document is not valid UTF-8."
            ) from exc

        return self._build_result(
            filename=filename,
            file_format="txt",
            sections=[
                ExtractedSection(content=text)
            ] if text else [],
            page_count=None,
        )
