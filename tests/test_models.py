"""Tests for core data models."""

from datetime import datetime
from pathlib import Path

from pdf2markdown.core import (
    Document,
    Page,
    PageMetadata,
    ProcessingStatus,
)


class TestPageMetadata:
    """Test PageMetadata model."""

    def test_creation(self):
        """Test creating PageMetadata."""
        metadata = PageMetadata(page_number=1, total_pages=10, width=800, height=600, dpi=300)

        assert metadata.page_number == 1
        assert metadata.total_pages == 10
        assert metadata.width == 800
        assert metadata.height == 600
        assert metadata.dpi == 300
        assert metadata.rotation == 0
        assert metadata.extraction_timestamp is None


class TestPage:
    """Test Page model."""

    def test_creation(self):
        """Test creating a Page."""
        page = Page(id="page-1", document_id="doc-1", page_number=1)

        assert page.id == "page-1"
        assert page.document_id == "doc-1"
        assert page.page_number == 1
        assert page.status == ProcessingStatus.PENDING
        assert page.image_path is None
        assert page.markdown_content is None

    def test_auto_id(self):
        """Test automatic ID generation."""
        page = Page(id="", document_id="doc-1", page_number=1)

        assert page.id != ""
        assert len(page.id) == 36  # UUID4 format

    def test_is_processed(self):
        """Test is_processed method."""
        page = Page(id="page-1", document_id="doc-1", page_number=1)

        assert not page.is_processed()

        page.status = ProcessingStatus.COMPLETED
        assert page.is_processed()

        page.status = ProcessingStatus.FAILED
        assert not page.is_processed()


class TestDocument:
    """Test Document model."""

    def test_creation(self):
        """Test creating a Document."""
        doc = Document(id="doc-1", source_path=Path("/path/to/document.pdf"))

        assert doc.id == "doc-1"
        assert doc.source_path == Path("/path/to/document.pdf")
        assert doc.status == ProcessingStatus.PENDING
        assert len(doc.pages) == 0
        assert doc.completed_at is None

    def test_auto_id(self):
        """Test automatic ID generation."""
        doc = Document(id="", source_path="/path/to/document.pdf")

        assert doc.id != ""
        assert len(doc.id) == 36  # UUID4 format

    def test_add_page(self):
        """Test adding pages to document."""
        doc = Document(id="doc-1", source_path=Path("/path/to/document.pdf"))

        page1 = Page(id="page-1", document_id="doc-1", page_number=1)
        page2 = Page(id="page-2", document_id="doc-1", page_number=2)

        doc.add_page(page1)
        doc.add_page(page2)

        assert len(doc.pages) == 2
        assert doc.pages[0] == page1
        assert doc.pages[1] == page2

    def test_get_page(self):
        """Test getting page by number."""
        doc = Document(id="doc-1", source_path=Path("/path/to/document.pdf"))

        page1 = Page(id="page-1", document_id="doc-1", page_number=1)
        page2 = Page(id="page-2", document_id="doc-1", page_number=2)

        doc.add_page(page1)
        doc.add_page(page2)

        assert doc.get_page(1) == page1
        assert doc.get_page(2) == page2
        assert doc.get_page(3) is None

    def test_mark_complete(self):
        """Test marking document as complete."""
        doc = Document(id="doc-1", source_path=Path("/path/to/document.pdf"))

        assert doc.status == ProcessingStatus.PENDING
        assert doc.completed_at is None

        doc.mark_complete()

        assert doc.status == ProcessingStatus.COMPLETED
        assert doc.completed_at is not None
        assert isinstance(doc.completed_at, datetime)

    def test_mark_failed(self):
        """Test marking document as failed."""
        doc = Document(id="doc-1", source_path=Path("/path/to/document.pdf"))

        doc.mark_failed("Test error")

        assert doc.status == ProcessingStatus.FAILED
        assert doc.metadata["error"] == "Test error"
