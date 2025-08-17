"""Core data models for PDF to Markdown conversion."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ProcessingStatus(Enum):
    """Processing status for documents and pages."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PageMetadata:
    """Metadata for a single page."""

    page_number: int
    total_pages: int
    width: int
    height: int
    dpi: int
    rotation: int = 0
    extraction_timestamp: datetime | None = None
    additional_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    """Represents a single page from a document."""

    id: str
    document_id: str
    page_number: int
    image_path: Path | None = None
    markdown_content: str | None = None
    metadata: PageMetadata | None = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: str | None = None

    def __post_init__(self):
        """Ensure id is always a string."""
        if not self.id:
            self.id = str(uuid.uuid4())

    def is_processed(self) -> bool:
        """Check if the page has been successfully processed."""
        return self.status == ProcessingStatus.COMPLETED


@dataclass
class Document:
    """Represents a complete PDF document."""

    id: str
    source_path: Path
    output_path: Path | None = None
    pages: list[Page] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def __post_init__(self):
        """Ensure id is always a string and path is Path object."""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not isinstance(self.source_path, Path):
            self.source_path = Path(self.source_path)
        if self.output_path and not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)

    def add_page(self, page: Page) -> None:
        """Add a page to the document."""
        self.pages.append(page)

    def get_page(self, page_number: int) -> Page | None:
        """Get a page by its page number."""
        return next((p for p in self.pages if p.page_number == page_number), None)

    def mark_complete(self) -> None:
        """Mark the document as complete."""
        self.status = ProcessingStatus.COMPLETED
        self.completed_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        """Mark the document as failed."""
        self.status = ProcessingStatus.FAILED
        self.metadata["error"] = error
