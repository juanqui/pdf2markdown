"""Worker implementation for pipeline processing."""

import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pdf2markdown.core import Document, Page
from pdf2markdown.parsers import DocumentParser, PageParser

from .queue_manager import QueueItem, QueueManager

logger = logging.getLogger(__name__)


class WorkerType(Enum):
    """Types of workers in the pipeline."""

    DOCUMENT = "document"
    PAGE = "page"
    OUTPUT = "output"


class Worker(ABC):
    """Base worker class for pipeline processing."""

    def __init__(self, worker_id: str, worker_type: WorkerType, queue_manager: QueueManager):
        """Initialize the worker.

        Args:
            worker_id: Unique identifier for the worker
            worker_type: Type of worker
            queue_manager: Queue manager instance
        """
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.queue_manager = queue_manager
        self.is_active = False
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{worker_id}")
        self.stats = {"tasks_processed": 0, "tasks_failed": 0}

    @abstractmethod
    async def process_task(self, task: Any) -> Any:
        """Process a single task.

        Args:
            task: Task to process

        Returns:
            Processed result
        """
        pass

    async def run(self) -> None:
        """Main worker loop."""
        self.is_active = True
        self.logger.info(f"Worker {self.worker_id} started")

        while self.is_active:
            try:
                task = await self._get_next_task()
                if task:
                    result = await self.process_task(task.data)
                    await self._handle_result(result, task)
                    self.stats["tasks_processed"] += 1
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error processing task: {e}")
                if task:
                    await self._handle_error(task, e)
                self.stats["tasks_failed"] += 1

    async def stop(self) -> None:
        """Stop the worker."""
        self.is_active = False
        self.logger.info(f"Worker {self.worker_id} stopped. Stats: {self.stats}")

    @abstractmethod
    async def _get_next_task(self) -> QueueItem | None:
        """Get the next task from the appropriate queue.

        Returns:
            QueueItem or None if no task available
        """
        pass

    @abstractmethod
    async def _handle_result(self, result: Any, task: QueueItem) -> None:
        """Handle the processing result.

        Args:
            result: Processing result
            task: Original task item
        """
        pass

    async def _handle_error(self, task: QueueItem, error: Exception) -> None:
        """Handle processing errors.

        Args:
            task: Failed task
            error: Exception that occurred
        """
        if task and task.retry_count < task.max_retries:
            task.retry_count += 1
            self.logger.warning(f"Retrying task (attempt {task.retry_count}/{task.max_retries})")
            await self._requeue_task(task)
        else:
            # Log that we've exhausted retries
            if task:
                self.logger.error(
                    f"Task failed after {task.max_retries} retries. Moving to error queue. Error: {str(error)[:200]}"
                )
            else:
                self.logger.error(
                    f"Task failed with no retry information. Error: {str(error)[:200]}"
                )

            await self.queue_manager.add_error(
                {
                    "task": task,
                    "error": str(error),
                    "worker_id": self.worker_id,
                    "worker_type": self.worker_type.value,
                }
            )

    @abstractmethod
    async def _requeue_task(self, task: QueueItem) -> None:
        """Requeue a task for retry.

        Args:
            task: Task to requeue
        """
        pass


class DocumentWorker(Worker):
    """Worker for processing documents."""

    def __init__(self, worker_id: str, queue_manager: QueueManager, parser: DocumentParser):
        """Initialize the document worker.

        Args:
            worker_id: Unique identifier for the worker
            queue_manager: Queue manager instance
            parser: Document parser instance
        """
        super().__init__(worker_id, WorkerType.DOCUMENT, queue_manager)
        self.parser = parser

    async def process_task(self, task: Document) -> Document:
        """Process a document.

        Args:
            task: Document to process

        Returns:
            Processed document with pages
        """
        self.logger.info(f"Processing document {task.id}")

        # Parse the document
        document = await self.parser.parse(task.source_path)

        # IMPORTANT: Keep the document ID from the parser (deterministic hash)
        # This is essential for cache consistency - do not overwrite with task.id
        # The parser uses a deterministic hash based on file content and config,
        # which is required for proper caching
        parser_document_id = document.id

        # Preserve the output path from the original task
        document.output_path = task.output_path

        # Update the coordinator's tracking to use the parser's document ID
        if hasattr(self, "coordinator") and self.coordinator:
            # Move the document from the old UUID to the new deterministic hash
            if task.id in self.coordinator.active_documents:
                self.coordinator.active_documents[parser_document_id] = (
                    self.coordinator.active_documents.pop(task.id)
                )
                self.logger.debug(
                    f"Updated coordinator tracking: {task.id} -> {parser_document_id}"
                )

        # Pages already have the correct document_id from the parser
        # Add pages to processing queue
        for page in document.pages:
            await self.queue_manager.add_page(page)

        self.logger.info(
            f"Document {parser_document_id} processed with {len(document.pages)} pages"
        )
        return document

    async def _get_next_task(self) -> QueueItem | None:
        """Get the next document from the queue."""
        return await self.queue_manager.get_next_document()

    async def _handle_result(self, result: Document, task: QueueItem) -> None:
        """Handle the document processing result."""
        # Document is processed, pages are in the page queue
        # Update the coordinator's active_documents with the parsed document that has pages
        if hasattr(self, "coordinator") and self.coordinator:
            if result.id in self.coordinator.active_documents:
                self.coordinator.active_documents[result.id] = result
        self.logger.debug(f"Document {result.id} added to output")

    async def _requeue_task(self, task: QueueItem) -> None:
        """Requeue a document for retry."""
        # IMPORTANT: Requeue the actual task item to preserve retry_count
        await self.queue_manager.requeue_document(task)


class PageWorker(Worker):
    """Worker for processing pages."""

    def __init__(self, worker_id: str, queue_manager: QueueManager, parser: PageParser):
        """Initialize the page worker.

        Args:
            worker_id: Unique identifier for the worker
            queue_manager: Queue manager instance
            parser: Page parser instance
        """
        super().__init__(worker_id, WorkerType.PAGE, queue_manager)
        self.parser = parser

    async def process_task(self, task: Page) -> Page:
        """Process a page.

        Args:
            task: Page to process

        Returns:
            Processed page with markdown content
        """
        self.logger.info(f"Processing page {task.page_number} from document {task.document_id}")

        # Parse the page
        page = await self.parser.parse(task)

        self.logger.info(f"Page {task.page_number} processed successfully")
        return page

    async def _get_next_task(self) -> QueueItem | None:
        """Get the next page from the queue."""
        return await self.queue_manager.get_next_page()

    async def _handle_result(self, result: Page, task: QueueItem) -> None:
        """Handle the page processing result."""
        # Add processed page to output queue
        await self.queue_manager.add_output(result)
        self.logger.debug(f"Page {result.page_number} added to output")

    async def _requeue_task(self, task: QueueItem) -> None:
        """Requeue a page for retry."""
        # IMPORTANT: Requeue the actual task item to preserve retry_count
        await self.queue_manager.requeue_page(task)


class OutputWorker(Worker):
    """Worker for handling output."""

    def __init__(self, worker_id: str, queue_manager: QueueManager, output_handler: Any):
        """Initialize the output worker.

        Args:
            worker_id: Unique identifier for the worker
            queue_manager: Queue manager instance
            output_handler: Handler for output processing
        """
        super().__init__(worker_id, WorkerType.OUTPUT, queue_manager)
        self.output_handler = output_handler
        self.documents = {}  # Track documents being assembled

    async def process_task(self, task: Page) -> None:
        """Process output data.

        Args:
            task: Page to add to output
        """
        # Assemble pages into documents
        doc_id = task.document_id

        if doc_id not in self.documents:
            self.documents[doc_id] = []

        self.documents[doc_id].append(task)
        self.logger.debug(f"Added page {task.page_number} to document {doc_id}")

        # Call the output handler to notify about the completed page
        if self.output_handler:
            await self.output_handler(task)

    async def _get_next_task(self) -> QueueItem | None:
        """Get the next output from the queue."""
        output = await self.queue_manager.get_next_output()
        if output:
            return QueueItem(data=output)
        return None

    async def _handle_result(self, result: Any, task: QueueItem) -> None:
        """Handle the output processing result."""
        # Output is already processed
        pass

    async def _requeue_task(self, task: QueueItem) -> None:
        """Requeue output for retry."""
        await self.queue_manager.add_output(task.data)
