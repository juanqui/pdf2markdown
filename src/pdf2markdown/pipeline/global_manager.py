"""Global pipeline manager for batch processing multiple documents with shared worker pools."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pdf2markdown.core import (
    Document,
    Page,
    PDFToMarkdownError,
    ProcessingStatus,
)
from pdf2markdown.llm_providers import create_llm_provider
from pdf2markdown.parsers import (
    SimpleDocumentParser,
    SimpleLLMPageParser,
)
from pdf2markdown.utils.statistics import get_statistics_tracker

from .progress import ProgressTracker
from .queue_manager import QueueManager, QueuePriority
from .worker import DocumentWorker, OutputWorker, PageWorker

logger = logging.getLogger(__name__)


class GlobalPipelineManager:
    """Global pipeline manager that maintains a single worker pool for processing multiple documents."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the global pipeline manager.

        Args:
            config: Configuration dictionary with pipeline settings
        """
        self.config = config

        # Document processing queue and tracking
        self.document_queue: asyncio.Queue[tuple[Path, Path | None]] = asyncio.Queue()
        self.active_documents: dict[str, Document] = {}
        self.completed_documents: dict[str, Document] = {}

        # Global worker pools (shared across all documents)
        self.queue_manager = QueueManager(config.get("queues", {}))
        self.progress = ProgressTracker(enable=config.get("enable_progress", True))

        # Initialize parsers (shared instances)
        doc_parser_config = config.get("document_parser", {})
        if "page_limit" in config:
            doc_parser_config["page_limit"] = config["page_limit"]
        self.document_parser = SimpleDocumentParser(doc_parser_config, full_config=config)

        # Initialize page parser with LLM provider
        page_parser_config = config.get("page_parser", {})
        llm_provider_config = config.get("llm_provider")
        if llm_provider_config:
            if isinstance(llm_provider_config, dict):
                llm_provider = create_llm_provider(llm_provider_config)
            else:
                llm_provider = create_llm_provider(llm_provider_config.model_dump())
            self.page_parser = SimpleLLMPageParser(
                page_parser_config, llm_provider, full_config=config
            )
            self.page_parser.set_cache_manager(self.document_parser.cache_manager)
        else:
            raise ValueError("LLM provider configuration is required")

        # Worker management
        self.workers_started = False
        self.document_workers: list[DocumentWorker] = []
        self.page_workers: list[PageWorker] = []
        self.output_workers: list[OutputWorker] = []
        self.worker_tasks: list[asyncio.Task] = []

        # Configuration
        pipeline_config = config.get("pipeline", {})
        self.document_concurrency = pipeline_config.get("document_workers", 1)
        self.page_worker_count = pipeline_config.get("page_workers", 10)

        logger.info(
            f"Initialized GlobalPipelineManager with {self.page_worker_count} page workers, "
            f"document concurrency: {self.document_concurrency}"
        )

    async def _start_workers(self) -> None:
        """Start the global worker pool."""
        if self.workers_started:
            return

        logger.info("Starting global worker pool")

        # Create document workers (limited by document_concurrency)
        for i in range(self.document_concurrency):
            doc_worker = DocumentWorker(
                worker_id=f"global_doc_worker_{i+1}",
                queue_manager=self.queue_manager,
                parser=self.document_parser,
            )
            doc_worker.coordinator = self  # Set reference for document tracking
            self.document_workers.append(doc_worker)

        # Create page workers (shared across all active documents)
        for i in range(self.page_worker_count):
            page_worker = PageWorker(
                worker_id=f"global_page_worker_{i+1}",
                queue_manager=self.queue_manager,
                parser=self.page_parser,
            )
            self.page_workers.append(page_worker)

        # Create output worker
        output_worker = OutputWorker(
            worker_id="global_output_worker_1",
            queue_manager=self.queue_manager,
            output_handler=self._handle_output,
        )
        self.output_workers.append(output_worker)

        # Start all workers
        self.worker_tasks = []
        for worker in self.document_workers + self.page_workers + self.output_workers:
            task = asyncio.create_task(worker.run())
            self.worker_tasks.append(task)

        self.workers_started = True
        logger.info(
            f"Started global worker pool: {len(self.document_workers)} document workers, "
            f"{len(self.page_workers)} page workers, {len(self.output_workers)} output workers"
        )

    async def _stop_workers(self) -> None:
        """Stop all workers."""
        if not self.workers_started:
            return

        logger.info("Stopping global worker pool")

        # Stop all workers
        for worker in self.document_workers + self.page_workers + self.output_workers:
            await worker.stop()

        # Cancel all tasks
        for task in self.worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        self.workers_started = False
        logger.info("Global worker pool stopped")

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
            if document.pages and all(p.is_processed() for p in document.pages):
                document.mark_complete()
                self.completed_documents[doc_id] = document
                del self.active_documents[doc_id]
                logger.info(f"Document {doc_id} completed")

    async def process_document(
        self, document_path: Path, output_path: Path | None = None
    ) -> Document:
        """Process a single document through the global pipeline.

        Args:
            document_path: Path to the PDF document
            output_path: Optional output path for the document

        Returns:
            Processed Document with markdown content
        """
        logger.info(f"Processing document: {document_path}")

        # Ensure workers are started
        await self._start_workers()

        # Validate the document
        if not self.document_parser.validate_document(document_path):
            raise PDFToMarkdownError(f"Invalid document: {document_path}")

        # Create document with deterministic ID (using cache manager approach)
        from pdf2markdown.utils.cache_manager import ConfigHasher, DocumentHasher

        doc_config_hash = ConfigHasher.hash_document_config(self.config)
        doc_hash = DocumentHasher.hash_document(document_path, doc_config_hash)

        document = Document(
            id=doc_hash,
            source_path=document_path,
            output_path=output_path,
            status=ProcessingStatus.PENDING,
        )

        # Track this document
        self.active_documents[doc_hash] = document

        # Add document to queue for processing
        await self.queue_manager.add_document(document, QueuePriority.NORMAL)

        # Wait for processing to complete
        while True:
            # Check if document is complete
            if document.id in self.completed_documents:
                result = self.completed_documents[document.id]
                del self.completed_documents[document.id]  # Clean up
                break

            # Check for errors
            error = await self.queue_manager.get_next_error()
            if error:
                logger.error(f"Pipeline error: {error}")

            # Wait a bit
            await asyncio.sleep(0.1)

        # Save markdown output if path is specified
        logger.debug(f"Document result output_path: {result.output_path}")
        if result.output_path:
            logger.info(f"Saving markdown to: {result.output_path}")
            await self._save_markdown(result)
        else:
            logger.warning(f"No output path set for document {result.id}, markdown not saved")

        logger.info(f"Document processing complete: {document_path}")
        return result

    async def process_documents_batch(
        self,
        documents: list[tuple[Path, Path | None]],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[Document]:
        """Process multiple documents sequentially using the global worker pool.

        Args:
            documents: List of (document_path, output_path) tuples
            progress_callback: Optional progress callback function

        Returns:
            List of processed Documents
        """
        logger.info(f"Starting batch processing of {len(documents)} documents")

        # Initialize statistics
        stats = get_statistics_tracker()
        stats.start_process()

        results = []

        try:
            # Process documents sequentially to maintain order
            for i, (doc_path, output_path) in enumerate(documents, 1):
                if progress_callback:
                    progress_callback(i, len(documents), f"Processing {doc_path.name}")

                try:
                    result = await self.process_document(doc_path, output_path)
                    results.append(result)
                    logger.info(f"Completed {i}/{len(documents)}: {doc_path.name}")
                except Exception as e:
                    logger.error(f"Failed to process {doc_path}: {e}")
                    # Create a failed document entry
                    failed_doc = Document(
                        id=f"failed_{i}",
                        source_path=doc_path,
                        output_path=output_path,
                        status=ProcessingStatus.FAILED,
                    )
                    failed_doc.mark_failed(str(e))
                    results.append(failed_doc)

        finally:
            # Stop workers when done
            await self._stop_workers()
            stats.end_process()

        logger.info(f"Batch processing complete: {len(results)} documents processed")
        return results

    async def _save_markdown(self, document: Document) -> None:
        """Save document pages as markdown.

        Args:
            document: Document with processed pages
        """
        if not document.output_path:
            # Generate output path
            document.output_path = document.source_path.with_suffix(".md")

        # Get page separator from config
        page_separator = self.config.get("page_separator", "\n\n--[PAGE: {page_number}]--\n\n")

        # Combine all page content
        markdown_content = []
        sorted_pages = sorted(document.pages, key=lambda p: p.page_number)

        for page in sorted_pages:
            if page.markdown_content:
                # Add page separator before each page (except the first)
                if page.page_number > 1:
                    separator = page_separator.format(page_number=page.page_number)
                    markdown_content.append(separator)

                markdown_content.append(page.markdown_content)

        # Write to file
        document.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(document.output_path, "w", encoding="utf-8") as f:
            f.write("".join(markdown_content))

        logger.info(f"Saved markdown to {document.output_path}")

    async def get_status(self) -> dict[str, Any]:
        """Get processing status for the global pipeline.

        Returns:
            Status information
        """
        return {
            "workers_started": self.workers_started,
            "active_documents": len(self.active_documents),
            "completed_documents": len(self.completed_documents),
            "document_workers": len(self.document_workers),
            "page_workers": len(self.page_workers),
            "queue_stats": self.queue_manager.get_stats(),
        }

    async def cleanup(self) -> None:
        """Cleanup all resources."""
        logger.info("Cleaning up global pipeline manager")

        await self._stop_workers()

        # Cleanup parsers
        await self.document_parser.cleanup()
        await self.page_parser.cleanup()

        # Clear tracking dictionaries
        self.active_documents.clear()
        self.completed_documents.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
