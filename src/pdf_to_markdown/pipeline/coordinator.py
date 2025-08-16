"""Pipeline coordinator for managing the conversion process."""

import asyncio
import logging
from pathlib import Path
from typing import Any

from pdf_to_markdown.core import (
    Document,
    Page,
    PDFToMarkdownError,
    Pipeline,
    ProcessingStatus,
)
from pdf_to_markdown.llm_providers import create_llm_provider
from pdf_to_markdown.parsers import (
    SimpleDocumentParser,
    SimpleLLMPageParser,
)

from .progress import ProgressTracker
from .queue_manager import QueueManager, QueuePriority
from .worker import DocumentWorker, OutputWorker, PageWorker

logger = logging.getLogger(__name__)


class PipelineCoordinator(Pipeline):
    """Coordinates the pipeline processing of PDF documents."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the pipeline coordinator.

        Args:
            config: Configuration dictionary with pipeline settings
        """
        self.config = config

        # Initialize queue manager
        queue_config = config.get("queues", {})
        self.queue_manager = QueueManager(queue_config)

        # Initialize progress tracker
        self.progress = ProgressTracker(enable=config.get("enable_progress", True))

        # Initialize parsers
        doc_parser_config = config.get("document_parser", {})
        # Pass page_limit to document parser if set
        if "page_limit" in config:
            doc_parser_config["page_limit"] = config["page_limit"]
        self.document_parser = SimpleDocumentParser(doc_parser_config)

        # Initialize page parser with LLM provider
        page_parser_config = config.get("page_parser", {})

        # Get LLM provider from top-level config
        llm_provider_config = config.get("llm_provider")
        if llm_provider_config:
            if isinstance(llm_provider_config, dict):
                llm_provider = create_llm_provider(llm_provider_config)
            else:
                # Assume it's already a config object
                llm_provider = create_llm_provider(llm_provider_config.model_dump())
            self.page_parser = SimpleLLMPageParser(page_parser_config, llm_provider)
        else:
            # No provider configured - this will fail in SimpleLLMPageParser
            raise ValueError("LLM provider configuration is required")

        # Worker configuration
        self.document_workers: list[DocumentWorker] = []
        self.page_workers: list[PageWorker] = []
        self.output_workers: list[OutputWorker] = []

        # Tracking
        self.active_documents: dict[str, Document] = {}
        self.completed_documents: dict[str, Document] = {}

        logger.info("Initialized PipelineCoordinator")

    async def _start_workers(self) -> None:
        """Start all worker tasks."""
        # Create document workers (only 1 as per requirement)
        doc_worker = DocumentWorker(
            worker_id="doc_worker_1", queue_manager=self.queue_manager, parser=self.document_parser
        )
        # Give the worker a reference to the coordinator so it can update active_documents
        doc_worker.coordinator = self
        self.document_workers.append(doc_worker)

        # Create page workers
        page_worker_count = self.config.get("page_workers", 10)
        for i in range(page_worker_count):
            page_worker = PageWorker(
                worker_id=f"page_worker_{i+1}",
                queue_manager=self.queue_manager,
                parser=self.page_parser,
            )
            self.page_workers.append(page_worker)

        # Create output worker
        output_worker = OutputWorker(
            worker_id="output_worker_1",
            queue_manager=self.queue_manager,
            output_handler=self._handle_output,
        )
        self.output_workers.append(output_worker)

        # Start all workers
        self.worker_tasks = []
        for worker in self.document_workers + self.page_workers + self.output_workers:
            task = asyncio.create_task(worker.run())
            self.worker_tasks.append(task)

        logger.info(f"Started {len(self.worker_tasks)} workers")

    async def _stop_workers(self) -> None:
        """Stop all worker tasks."""
        # Stop all workers
        for worker in self.document_workers + self.page_workers + self.output_workers:
            await worker.stop()

        # Cancel all tasks
        for task in self.worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        logger.info("All workers stopped")

    async def _handle_output(self, page: Page) -> None:
        """Handle output from page processing.

        Args:
            page: Processed page
        """
        doc_id = page.document_id

        if doc_id in self.active_documents:
            document = self.active_documents[doc_id]

            # Update the page in the document
            for i, doc_page in enumerate(document.pages):
                if doc_page.page_number == page.page_number:
                    document.pages[i] = page
                    break

            # Check if all pages are complete
            if all(p.is_processed() for p in document.pages):
                document.mark_complete()
                self.completed_documents[doc_id] = document
                del self.active_documents[doc_id]
                self.progress.update_document_progress()
                logger.info(f"Document {doc_id} completed")

    async def process(self, document_path: Path) -> Document:
        """Process a complete document through the pipeline.

        Args:
            document_path: Path to the PDF document

        Returns:
            Processed Document with markdown content
        """
        logger.info(f"Starting pipeline processing for {document_path}")

        try:
            # Validate the document
            if not self.document_parser.validate_document(document_path):
                raise PDFToMarkdownError(f"Invalid document: {document_path}")

            # Start workers
            await self._start_workers()

            # Create initial document with a proper ID
            import uuid

            doc_id = str(uuid.uuid4())
            document = Document(
                id=doc_id, source_path=document_path, status=ProcessingStatus.PENDING
            )

            # Track this document
            self.active_documents[doc_id] = document

            # Add document to queue
            await self.queue_manager.add_document(document, QueuePriority.NORMAL)

            # Start progress tracking
            self.progress.start_document_processing(1)

            # Wait for processing to complete
            while True:
                # Check if document is complete
                if document.id in self.completed_documents:
                    result = self.completed_documents[document.id]
                    break

                # Check for errors
                error = await self.queue_manager.get_next_error()
                if error:
                    logger.error(f"Pipeline error: {error}")

                # Wait a bit
                await asyncio.sleep(0.5)

                # Update document reference if it's being processed
                if not document.id and self.active_documents:
                    # Get the document ID from active documents
                    for doc_id, doc in self.active_documents.items():
                        if doc.source_path == document_path:
                            document.id = doc_id
                            break

            # Stop workers
            await self._stop_workers()

            # Close progress
            self.progress.close()

            # Save markdown output
            if result.output_path:
                await self._save_markdown(result)

            logger.info(f"Pipeline processing complete for {document_path}")
            return result

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            await self._stop_workers()
            self.progress.close()
            raise

    async def _save_markdown(self, document: Document) -> None:
        """Save document pages as markdown.

        Args:
            document: Document with processed pages
        """
        if not document.output_path:
            # Generate output path
            document.output_path = document.source_path.with_suffix(".md")

        # Combine all page content
        markdown_content = []
        for page in sorted(document.pages, key=lambda p: p.page_number):
            if page.markdown_content:
                markdown_content.append(page.markdown_content)
                if page.page_number < len(document.pages):
                    markdown_content.append("\n---\n")  # Page separator

        # Write to file
        with open(document.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_content))

        logger.info(f"Saved markdown to {document.output_path}")

    async def get_status(self, document_id: str) -> dict[str, Any]:
        """Get processing status for a document.

        Args:
            document_id: Document ID

        Returns:
            Status information
        """
        if document_id in self.completed_documents:
            document = self.completed_documents[document_id]
            status = "completed"
        elif document_id in self.active_documents:
            document = self.active_documents[document_id]
            status = "processing"
        else:
            return {"status": "not_found"}

        return {
            "status": status,
            "document_id": document_id,
            "total_pages": len(document.pages),
            "processed_pages": sum(1 for p in document.pages if p.is_processed()),
            "queue_stats": self.queue_manager.get_stats(),
            "progress_stats": self.progress.get_stats(),
        }

    async def cancel(self, document_id: str) -> bool:
        """Cancel processing for a document.

        Args:
            document_id: Document ID

        Returns:
            True if cancelled successfully
        """
        if document_id in self.active_documents:
            document = self.active_documents[document_id]
            document.mark_failed("Cancelled by user")
            del self.active_documents[document_id]
            logger.info(f"Cancelled processing for document {document_id}")
            return True
        return False
