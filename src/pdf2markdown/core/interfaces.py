"""Abstract base classes and interfaces for PDF to Markdown conversion."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import Document, Page


class DocumentParser(ABC):
    """Abstract base class for document parsers."""

    def __init__(self, config: dict[str, Any]):
        """Initialize parser with configuration."""
        self.config = config

    @abstractmethod
    async def parse(self, document_path: Path) -> Document:
        """Parse a PDF document into a Document object with Pages."""
        pass

    @abstractmethod
    async def parse_page(self, document_path: Path, page_number: int) -> Page:
        """Parse a single page from a document."""
        pass

    @abstractmethod
    def validate_document(self, document_path: Path) -> bool:
        """Validate if the document can be parsed."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class PageParser(ABC):
    """Abstract base class for page parsers."""

    def __init__(self, config: dict[str, Any]):
        """Initialize parser with configuration."""
        self.config = config

    @abstractmethod
    async def parse(self, page: Page) -> Page:
        """Convert a page image to markdown."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class Pipeline(ABC):
    """Abstract base class for processing pipelines."""

    @abstractmethod
    async def process(self, document_path: Path) -> Document:
        """Process a complete document through the pipeline."""
        pass

    @abstractmethod
    async def get_status(self, document_id: str) -> dict[str, Any]:
        """Get processing status for a document."""
        pass

    @abstractmethod
    async def cancel(self, document_id: str) -> bool:
        """Cancel processing for a document."""
        pass
