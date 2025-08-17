"""Type definitions for the pdf-to-markdown library API."""

from typing import Dict, Any, Optional, List, Callable, Union, AsyncIterator
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


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
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentResult:
    """Result of processing an entire document."""
    source_path: Path
    pages: List[PageResult]
    total_pages: int
    status: ConversionStatus
    markdown_content: Optional[str] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
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
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save the markdown content to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding='utf-8')


# Type aliases for callbacks
ProgressCallback = Callable[[int, int, str], None]
AsyncProgressCallback = Callable[[int, int, str], Any]

# Configuration dictionary type
ConfigDict = Dict[str, Any]