"""Simple document parser using PyMuPDF."""

import logging
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf

from pdf_to_markdown.core import (
    Document,
    DocumentParser,
    DocumentParsingError,
    InvalidFileFormatError,
    Page,
    PageMetadata,
    ProcessingStatus,
)

logger = logging.getLogger(__name__)


class SimpleDocumentParser(DocumentParser):
    """Simple document parser that uses PyMuPDF to render pages to images."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the parser with configuration.

        Args:
            config: Configuration dictionary with the following keys:
                - resolution (int): DPI for rendering pages (default: 300)
                - cache_dir (Path): Directory for caching images
                - max_page_size (int): Maximum page size in bytes
                - timeout (int): Timeout in seconds for page rendering
                - page_limit (int): Optional limit on number of pages to process
        """
        super().__init__(config)
        self.resolution = config.get("resolution", 300)
        self.cache_dir = Path(
            config.get("cache_dir", tempfile.gettempdir()) / "pdf_to_markdown_cache"
        )
        self.max_page_size = config.get("max_page_size", 50_000_000)  # 50MB
        self.timeout = config.get("timeout", 30)
        self.page_limit = config.get("page_limit", None)

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Initialized SimpleDocumentParser with resolution={self.resolution}, cache_dir={self.cache_dir}"
        )

    def validate_document(self, document_path: Path) -> bool:
        """Validate if the document can be parsed.

        Args:
            document_path: Path to the PDF document

        Returns:
            True if the document is valid and can be parsed

        Raises:
            InvalidFileFormatError: If the file is not a valid PDF
        """
        if not document_path.exists():
            raise InvalidFileFormatError(f"File not found: {document_path}")

        if not document_path.suffix.lower() == ".pdf":
            raise InvalidFileFormatError(f"File is not a PDF: {document_path}")

        try:
            # Try to open the document to validate it
            with pymupdf.open(document_path) as doc:
                if doc.page_count == 0:
                    raise InvalidFileFormatError(f"PDF has no pages: {document_path}")
            return True
        except Exception as e:
            raise InvalidFileFormatError(f"Invalid PDF file: {e}")

    async def parse(self, document_path: Path) -> Document:
        """Parse a PDF document into a Document object with Pages.

        Args:
            document_path: Path to the PDF document

        Returns:
            Document object with all pages rendered as images

        Raises:
            DocumentParsingError: If there's an error parsing the document
        """
        logger.info(f"Starting to parse document: {document_path}")

        # Validate document first
        self.validate_document(document_path)

        # Create document object
        doc_id = str(uuid.uuid4())
        document = Document(
            id=doc_id, source_path=document_path, status=ProcessingStatus.PROCESSING
        )

        try:
            # Open the PDF document
            pdf_doc = pymupdf.open(document_path)
            document.metadata["page_count"] = pdf_doc.page_count
            document.metadata["metadata"] = pdf_doc.metadata

            # Create cache directory for this document
            doc_cache_dir = self.cache_dir / doc_id
            doc_cache_dir.mkdir(parents=True, exist_ok=True)

            # Determine number of pages to process
            pages_to_process = pdf_doc.page_count
            if self.page_limit and self.page_limit < pages_to_process:
                pages_to_process = self.page_limit
                logger.info(
                    f"Limiting processing to {pages_to_process} pages (out of {pdf_doc.page_count})"
                )

            # Process each page
            for page_num in range(pages_to_process):
                logger.debug(f"Processing page {page_num + 1}/{pages_to_process}")

                # Load the page
                pdf_page = pdf_doc.load_page(page_num)

                # Get page dimensions
                rect = pdf_page.rect
                width = int(rect.width)
                height = int(rect.height)

                # Create page metadata
                metadata = PageMetadata(
                    page_number=page_num + 1,
                    total_pages=pdf_doc.page_count,
                    width=width,
                    height=height,
                    dpi=self.resolution,
                    rotation=pdf_page.rotation,
                    extraction_timestamp=datetime.now(),
                )

                # Render page to image
                mat = pymupdf.Matrix(self.resolution / 72.0, self.resolution / 72.0)
                pix = pdf_page.get_pixmap(matrix=mat, alpha=False)

                # Save image to cache
                image_path = doc_cache_dir / f"page_{page_num + 1:04d}.png"
                pix.save(str(image_path))

                # Check file size
                file_size = image_path.stat().st_size
                if file_size > self.max_page_size:
                    logger.warning(f"Page {page_num + 1} exceeds max size: {file_size} bytes")

                # Create page object
                page = Page(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    page_number=page_num + 1,
                    image_path=image_path,
                    metadata=metadata,
                    status=ProcessingStatus.PENDING,
                )

                # Add page to document
                document.add_page(page)

                # Clean up pixmap
                pix = None

            # Close the PDF document
            pdf_doc.close()

            # Update document status
            document.status = ProcessingStatus.PENDING  # Ready for page parsing
            logger.info(
                f"Successfully parsed document with {document.metadata['page_count']} pages"
            )

            return document

        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            document.mark_failed(str(e))
            raise DocumentParsingError(f"Failed to parse document: {e}")

    async def parse_page(self, document_path: Path, page_number: int) -> Page:
        """Parse a single page from a document.

        Args:
            document_path: Path to the PDF document
            page_number: Page number to parse (1-indexed)

        Returns:
            Page object with the rendered image

        Raises:
            DocumentParsingError: If there's an error parsing the page
        """
        logger.info(f"Parsing single page {page_number} from {document_path}")

        try:
            # Open the PDF document
            pdf_doc = pymupdf.open(document_path)

            if page_number < 1 or page_number > pdf_doc.page_count:
                raise DocumentParsingError(f"Invalid page number: {page_number}")

            # Load the page (0-indexed)
            pdf_page = pdf_doc.load_page(page_number - 1)

            # Get page dimensions
            rect = pdf_page.rect
            width = int(rect.width)
            height = int(rect.height)

            # Create page metadata
            metadata = PageMetadata(
                page_number=page_number,
                total_pages=pdf_doc.page_count,
                width=width,
                height=height,
                dpi=self.resolution,
                rotation=pdf_page.rotation,
                extraction_timestamp=datetime.now(),
            )

            # Render page to image
            mat = pymupdf.Matrix(self.resolution / 72.0, self.resolution / 72.0)
            pix = pdf_page.get_pixmap(matrix=mat, alpha=False)

            # Save image to cache
            doc_id = str(uuid.uuid4())
            doc_cache_dir = self.cache_dir / doc_id
            doc_cache_dir.mkdir(parents=True, exist_ok=True)
            image_path = doc_cache_dir / f"page_{page_number:04d}.png"
            pix.save(str(image_path))

            # Create page object
            page = Page(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                page_number=page_number,
                image_path=image_path,
                metadata=metadata,
                status=ProcessingStatus.PENDING,
            )

            # Clean up
            pix = None
            pdf_doc.close()

            return page

        except Exception as e:
            logger.error(f"Error parsing page {page_number}: {e}")
            raise DocumentParsingError(f"Failed to parse page {page_number}: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources and cache."""
        logger.info("Cleaning up document parser resources")

        try:
            # Clean up cache directory if it exists
            if self.cache_dir.exists():
                # Only clean up old cache files (older than 24 hours)
                import time

                current_time = time.time()
                for item in self.cache_dir.iterdir():
                    if item.is_dir():
                        # Check if directory is older than 24 hours
                        dir_time = item.stat().st_mtime
                        if current_time - dir_time > 86400:  # 24 hours
                            logger.debug(f"Removing old cache directory: {item}")
                            shutil.rmtree(item, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
