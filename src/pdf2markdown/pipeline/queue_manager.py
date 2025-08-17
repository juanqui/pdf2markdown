"""Queue management for pipeline processing."""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pdf2markdown.core import Document, Page, QueueOverflowError

logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Priority levels for queue items."""

    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class QueueItem:
    """Wrapper for queue items with priority and metadata."""

    data: Any
    priority: QueuePriority = QueuePriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """Compare queue items by priority for priority queue."""
        return self.priority.value < other.priority.value


class QueueManager:
    """Manages multiple queues for pipeline processing."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the queue manager.

        Args:
            config: Configuration dictionary with queue sizes
        """
        self.config = config

        # Initialize queues
        self.document_queue = asyncio.PriorityQueue(maxsize=config.get("document_queue_size", 100))
        self.page_queue = asyncio.PriorityQueue(maxsize=config.get("page_queue_size", 1000))
        self.output_queue = asyncio.Queue(maxsize=config.get("output_queue_size", 500))
        self.error_queue = asyncio.Queue()

        # Statistics
        self.stats = {
            "documents_queued": 0,
            "documents_processed": 0,
            "pages_queued": 0,
            "pages_processed": 0,
            "errors": 0,
        }

        logger.debug(
            f"Initialized QueueManager with sizes: doc={config.get('document_queue_size', 100)}, "
            f"page={config.get('page_queue_size', 1000)}, output={config.get('output_queue_size', 500)}"
        )

    async def add_document(
        self, document: Document, priority: QueuePriority = QueuePriority.NORMAL
    ) -> None:
        """Add a document to the processing queue.

        Args:
            document: Document to add
            priority: Priority level for the document

        Raises:
            QueueOverflowError: If the queue is full
        """
        try:
            item = QueueItem(data=document, priority=priority)
            await self.document_queue.put((priority.value, item))
            self.stats["documents_queued"] += 1
            logger.debug(f"Added document {document.id} to queue with priority {priority.name}")
        except asyncio.QueueFull as e:
            raise QueueOverflowError("Document queue is full") from e

    async def get_next_document(self) -> QueueItem | None:
        """Get the next document from the queue.

        Returns:
            QueueItem with document or None if queue is empty
        """
        try:
            priority, item = await asyncio.wait_for(self.document_queue.get(), timeout=0.1)
            self.stats["documents_processed"] += 1
            return item
        except asyncio.TimeoutError:
            return None

    async def requeue_document(self, item: QueueItem) -> None:
        """Requeue a document item, preserving its retry count.

        Args:
            item: QueueItem to requeue with preserved retry count
        """
        try:
            # Use negative priority to ensure it's processed after new items
            await self.document_queue.put((-item.priority.value, item))
            logger.debug(f"Requeued document with retry_count={item.retry_count}")
        except asyncio.QueueFull as e:
            raise QueueOverflowError("Document queue is full") from e

    async def add_page(self, page: Page, priority: QueuePriority = QueuePriority.NORMAL) -> None:
        """Add a page to the processing queue.

        Args:
            page: Page to add
            priority: Priority level for the page

        Raises:
            QueueOverflowError: If the queue is full
        """
        try:
            item = QueueItem(data=page, priority=priority)
            # Use page number as secondary priority to ensure pages are processed in order
            # Lower page numbers will be processed first
            queue_priority = (priority.value, page.page_number)
            await self.page_queue.put((queue_priority, item))
            self.stats["pages_queued"] += 1
            logger.debug(f"Added page {page.page_number} from document {page.document_id} to queue")
        except asyncio.QueueFull as e:
            raise QueueOverflowError("Page queue is full") from e

    async def get_next_page(self) -> QueueItem | None:
        """Get the next page from the queue.

        Returns:
            QueueItem with page or None if queue is empty
        """
        try:
            queue_priority, item = await asyncio.wait_for(self.page_queue.get(), timeout=0.1)
            self.stats["pages_processed"] += 1
            return item
        except asyncio.TimeoutError:
            return None

    async def requeue_page(self, item: QueueItem) -> None:
        """Requeue a page item, preserving its retry count.

        Args:
            item: QueueItem to requeue with preserved retry count
        """
        try:
            # Use negative priority to ensure it's processed after new items
            # Use same tuple format as add_page: (priority, page_number)
            page = item.data
            queue_priority = (-item.priority.value, page.page_number)
            await self.page_queue.put((queue_priority, item))
            logger.debug(f"Requeued page with retry_count={item.retry_count}")
        except asyncio.QueueFull as e:
            raise QueueOverflowError("Page queue is full") from e

    async def add_output(self, data: Any) -> None:
        """Add processed output to the output queue.

        Args:
            data: Output data to add
        """
        try:
            await self.output_queue.put(data)
            logger.debug("Added output to queue")
        except asyncio.QueueFull as e:
            raise QueueOverflowError("Output queue is full") from e

    async def get_next_output(self) -> Any | None:
        """Get the next output from the queue.

        Returns:
            Output data or None if queue is empty
        """
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    async def add_error(self, error_info: dict[str, Any]) -> None:
        """Add error information to the error queue.

        Args:
            error_info: Dictionary with error details
        """
        await self.error_queue.put(error_info)
        self.stats["errors"] += 1
        logger.error(f"Added error to queue: {error_info.get('error', 'Unknown error')}")

    async def get_next_error(self) -> dict[str, Any] | None:
        """Get the next error from the queue.

        Returns:
            Error information or None if queue is empty
        """
        try:
            return await asyncio.wait_for(self.error_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        return {
            **self.stats,
            "document_queue_size": self.document_queue.qsize(),
            "page_queue_size": self.page_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
            "error_queue_size": self.error_queue.qsize(),
        }

    def is_empty(self) -> bool:
        """Check if all queues are empty.

        Returns:
            True if all queues are empty
        """
        return self.document_queue.empty() and self.page_queue.empty() and self.output_queue.empty()
