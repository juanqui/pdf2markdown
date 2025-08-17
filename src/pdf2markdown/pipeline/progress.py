"""Progress tracking for pipeline processing."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks progress of pipeline processing."""

    def __init__(self, enable: bool = True):
        """Initialize the progress tracker.

        Args:
            enable: Whether to enable progress tracking (kept for compatibility)
        """
        self.enable = enable
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
        self.stats["total_documents"] = total_documents
        if self.enable:
            logger.info(f"Starting to process {total_documents} document(s)")

    def start_page_processing(self, total_pages: int, document_name: str = "") -> None:
        """Start tracking page processing.

        Args:
            total_pages: Total number of pages to process
            document_name: Name of the current document
        """
        self.stats["total_pages"] += total_pages
        self.current_document = document_name
        if self.enable:
            if document_name:
                logger.info(f"Processing {total_pages} pages from {document_name}")
            else:
                logger.info(f"Processing {total_pages} pages")

    def update_document_progress(self, count: int = 1) -> None:
        """Update document processing progress.

        Args:
            count: Number of documents processed
        """
        self.stats["completed_documents"] += count
        if self.enable:
            logger.debug(
                f"Document progress: {self.stats['completed_documents']}/{self.stats['total_documents']}"
            )

    def update_page_progress(self, count: int = 1, failed: bool = False) -> None:
        """Update page processing progress.

        Args:
            count: Number of pages processed
            failed: Whether the page(s) failed processing
        """
        if failed:
            self.stats["failed_pages"] += count
        else:
            self.stats["completed_pages"] += count

        if self.enable:
            total_processed = self.stats["completed_pages"] + self.stats["failed_pages"]
            logger.debug(
                f"Page progress: {total_processed}/{self.stats['total_pages']} "
                f"(completed: {self.stats['completed_pages']}, failed: {self.stats['failed_pages']})"
            )

    def set_document_description(self, description: str) -> None:
        """Update the document progress description.

        Args:
            description: New description
        """
        if self.enable:
            logger.debug(f"Document: {description}")

    def set_page_description(self, description: str) -> None:
        """Update the page progress description.

        Args:
            description: New description
        """
        if self.enable:
            logger.debug(f"Page: {description}")

    def close_page_progress(self) -> None:
        """Close the page progress tracking."""
        # No-op without tqdm
        pass

    def close(self) -> None:
        """Close all progress tracking."""
        if self.enable and self.stats["total_pages"] > 0:
            success_rate = (
                self.stats["completed_pages"] / self.stats["total_pages"] * 100
                if self.stats["total_pages"] > 0
                else 0
            )
            logger.info(
                f"Processing complete: {self.stats['completed_pages']}/{self.stats['total_pages']} pages "
                f"({success_rate:.1f}% success rate)"
            )

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
