"""Type definitions for the pdf2markdown library API."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ConversionStatus(Enum):
    """Status of document conversion."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PageResult:
    """Result of processing a single page."""

    page_number: int
    content: str
    status: ConversionStatus
    processing_time: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentResult:
    """Result of processing an entire document."""

    source_path: Path
    pages: list[PageResult]
    total_pages: int
    status: ConversionStatus
    markdown_content: str | None = None
    processing_time: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_markdown(self, page_separator: str = "\n\n--[PAGE: {page_number}]--\n\n") -> str:
        """Combine all pages into a single markdown document."""
        if self.markdown_content:
            return self.markdown_content

        parts = []
        for page in self.pages:
            if page.content:
                if parts:  # Add separator between pages
                    parts.append(page_separator.format(page_number=page.page_number))
                parts.append(page.content)

        self.markdown_content = "".join(parts)
        return self.markdown_content

    def save(self, output_path: str | Path) -> None:
        """Save the markdown content to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")


# Type aliases for callbacks
ProgressCallback = Callable[[int, int, str], None]
AsyncProgressCallback = Callable[[int, int, str], Any]

# Configuration dictionary type
ConfigDict = dict[str, Any]
