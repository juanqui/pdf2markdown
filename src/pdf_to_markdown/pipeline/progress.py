"""Progress tracking for pipeline processing."""

import logging
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks progress of pipeline processing with tqdm."""

    def __init__(self, enable: bool = True):
        """Initialize the progress tracker.

        Args:
            enable: Whether to enable progress tracking
        """
        self.enable = enable
        self.document_progress: tqdm | None = None
        self.page_progress: tqdm | None = None
        self.current_document: str | None = None

        # Statistics
        self.stats = {
            "total_documents": 0,
            "completed_documents": 0,
            "total_pages": 0,
            "completed_pages": 0,
            "failed_pages": 0,
        }

    def start_document_processing(self, total_documents: int) -> None:
        """Start tracking document processing.

        Args:
            total_documents: Total number of documents to process
        """
        if self.enable and total_documents > 0:
            self.document_progress = tqdm(
                total=total_documents, desc="Documents", unit="doc", position=0, leave=True
            )
            self.stats["total_documents"] = total_documents

    def start_page_processing(self, total_pages: int, document_name: str = "") -> None:
        """Start tracking page processing.

        Args:
            total_pages: Total number of pages to process
            document_name: Name of the current document
        """
        if self.enable and total_pages > 0:
            desc = f"Pages ({document_name})" if document_name else "Pages"
            self.page_progress = tqdm(
                total=total_pages, desc=desc, unit="page", position=1, leave=False
            )
            self.stats["total_pages"] += total_pages
            self.current_document = document_name

    def update_document_progress(self, count: int = 1) -> None:
        """Update document processing progress.

        Args:
            count: Number of documents processed
        """
        if self.document_progress:
            self.document_progress.update(count)
            self.stats["completed_documents"] += count

    def update_page_progress(self, count: int = 1, failed: bool = False) -> None:
        """Update page processing progress.

        Args:
            count: Number of pages processed
            failed: Whether the page(s) failed processing
        """
        if self.page_progress:
            self.page_progress.update(count)

        if failed:
            self.stats["failed_pages"] += count
        else:
            self.stats["completed_pages"] += count

    def set_document_description(self, description: str) -> None:
        """Update the document progress bar description.

        Args:
            description: New description
        """
        if self.document_progress:
            self.document_progress.set_description(description)

    def set_page_description(self, description: str) -> None:
        """Update the page progress bar description.

        Args:
            description: New description
        """
        if self.page_progress:
            self.page_progress.set_description(description)

    def close_page_progress(self) -> None:
        """Close the page progress bar."""
        if self.page_progress:
            self.page_progress.close()
            self.page_progress = None

    def close(self) -> None:
        """Close all progress bars."""
        if self.page_progress:
            self.page_progress.close()
            self.page_progress = None

        if self.document_progress:
            self.document_progress.close()
            self.document_progress = None

    def get_stats(self) -> dict[str, Any]:
        """Get progress statistics.

        Returns:
            Dictionary with progress statistics
        """
        return {
            **self.stats,
            "success_rate": (
                self.stats["completed_pages"] / self.stats["total_pages"] * 100
                if self.stats["total_pages"] > 0
                else 0
            ),
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
