"""Simple document parser using PyMuPDF."""

import logging
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf

from pdf2markdown.core import (
    Document,
    DocumentParser,
    DocumentParsingError,
    InvalidFileFormatError,
    Page,
    PageMetadata,
    ProcessingStatus,
)
from pdf2markdown.utils.cache_manager import CacheManager, ConfigHasher, DocumentHasher
from pdf2markdown.utils.statistics import get_statistics_tracker

logger = logging.getLogger(__name__)


class SimpleDocumentParser(DocumentParser):
    """Simple document parser that uses PyMuPDF to render pages to images."""

    def __init__(self, config: dict[str, Any], full_config: dict[str, Any] | None = None):
        """Initialize the parser with configuration.

        Args:
            config: Configuration dictionary with the following keys:
                - resolution (int): DPI for rendering pages (default: 300)
                - max_dimension (int): Optional maximum pixels for longest side
                - cache_dir (Path): Directory for caching images
                - max_page_size (int): Maximum page size in bytes
                - timeout (int): Timeout in seconds for page rendering
                - page_limit (int): Optional limit on number of pages to process
                - use_cache (bool): Whether to use caching (default: True)
            full_config: Full application configuration for cache hashing
        """
        super().__init__(config)
        self.resolution = config.get("resolution", 300)
        self.max_dimension = config.get("max_dimension", None)
        self.cache_dir = Path(
            config.get("cache_dir", Path(tempfile.gettempdir()) / "pdf2markdown_cache")
        )
        self.max_page_size = config.get("max_page_size", 50_000_000)  # 50MB
        self.timeout = config.get("timeout", 30)
        self.page_limit = config.get("page_limit", None)
        self.use_cache = config.get("use_cache", True)

        # Store full config for cache hashing
        self.full_config = full_config or {"document_parser": config}

        # Initialize cache manager
        self.cache_manager = CacheManager(self.cache_dir)

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Initialized SimpleDocumentParser with resolution={self.resolution}, "
            f"max_dimension={self.max_dimension}, cache_dir={self.cache_dir}, "
            f"use_cache={self.use_cache}"
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
            raise InvalidFileFormatError(f"Invalid PDF file: {e}") from e

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

        # Get statistics tracker
        stats = get_statistics_tracker()
        stats.start_parsing()

        # Validate document first
        self.validate_document(document_path)

        try:
            # Generate configuration hashes for cache validation
            doc_config_hash = ConfigHasher.hash_document_config(self.full_config)

            # Generate document hash for deterministic identification
            doc_hash = DocumentHasher.hash_document(document_path, doc_config_hash)

            # Check if we should use cache
            image_cache = self.cache_manager.get_image_cache(doc_hash)
            use_cached_images = self.use_cache and image_cache.is_valid(doc_config_hash)

            # Create document object with deterministic ID
            document = Document(
                id=doc_hash, source_path=document_path, status=ProcessingStatus.PROCESSING
            )

            # Open the PDF document to get metadata
            pdf_doc = pymupdf.open(document_path)
            document.metadata["page_count"] = pdf_doc.page_count
            document.metadata["metadata"] = pdf_doc.metadata

            # Determine number of pages to process
            pages_to_process = pdf_doc.page_count
            if self.page_limit and self.page_limit < pages_to_process:
                pages_to_process = self.page_limit
                logger.info(
                    f"Limiting processing to {pages_to_process} pages (out of {pdf_doc.page_count})"
                )

            # Record total pages in statistics
            stats.total_pages = pages_to_process

            if use_cached_images:
                logger.info(f"Using cached images for document {doc_hash}")
                image_cache.get_cached_images()  # Trigger cache validation

                # Load pages from cache
                for page_num in range(1, pages_to_process + 1):
                    image_path = image_cache.get_image_path(page_num)

                    if image_path.exists():
                        # Create minimal metadata (we don't need to re-render)
                        metadata = PageMetadata(
                            page_number=page_num,
                            total_pages=pdf_doc.page_count,
                            width=0,  # Will be filled from cache if needed
                            height=0,
                            dpi=self.resolution,
                            rotation=0,
                            extraction_timestamp=datetime.fromtimestamp(image_path.stat().st_mtime),
                        )

                        # Create page object
                        page = Page(
                            id=str(uuid.uuid4()),
                            document_id=doc_hash,
                            page_number=page_num,
                            image_path=image_path,
                            metadata=metadata,
                            status=ProcessingStatus.PENDING,
                        )

                        # Add page to document
                        document.add_page(page)

                        # Track cached page
                        stats.start_page_parsing(page_num)
                        stats.end_page_parsing(page_num)
                    else:
                        logger.warning(f"Expected cached image missing: {image_path}")
                        use_cached_images = False
                        break

            if not use_cached_images:
                logger.info(f"Rendering pages for document {doc_hash}")

                # Save cache metadata
                image_cache.save_config(doc_config_hash, pages_to_process)

                # Process each page
                for page_num in range(pages_to_process):
                    logger.debug(f"Processing page {page_num + 1}/{pages_to_process}")

                    # Track page parsing time
                    stats.start_page_parsing(page_num + 1)

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

                    # Apply max_dimension resizing if specified
                    if self.max_dimension:
                        # Get current dimensions
                        current_width = pix.width
                        current_height = pix.height
                        max_current = max(current_width, current_height)

                        # Check if resizing is needed
                        if max_current > self.max_dimension:
                            # Calculate scaling factor
                            scale_factor = self.max_dimension / max_current
                            new_width = int(current_width * scale_factor)
                            new_height = int(current_height * scale_factor)

                            # Create scaled pixmap using PyMuPDF's scaling constructor
                            # Pixmap(src, width, height) - creates a scaled copy
                            scaled_pix = pymupdf.Pixmap(pix, new_width, new_height)

                            logger.debug(
                                f"Resized page {page_num + 1} from {current_width}x{current_height} "
                                f"to {new_width}x{new_height} (max_dimension={self.max_dimension})"
                            )

                            # Replace original pixmap with scaled one
                            pix = scaled_pix

                    # Save image to cache
                    image_path = image_cache.get_image_path(page_num + 1)
                    pix.save(str(image_path))

                    # Check file size
                    file_size = image_path.stat().st_size
                    if file_size > self.max_page_size:
                        logger.warning(f"Page {page_num + 1} exceeds max size: {file_size} bytes")

                    # Create page object
                    page = Page(
                        id=str(uuid.uuid4()),
                        document_id=doc_hash,
                        page_number=page_num + 1,
                        image_path=image_path,
                        metadata=metadata,
                        status=ProcessingStatus.PENDING,
                    )

                    # Add page to document
                    document.add_page(page)

                    # Mark page parsing complete
                    stats.end_page_parsing(page_num + 1)

                    # Clean up pixmap
                    pix = None

            # Close the PDF document
            pdf_doc.close()

            # Save document metadata for cache management
            page_config_hash = ConfigHasher.hash_page_config(self.full_config)
            self.cache_manager.save_document_metadata(
                doc_hash, document_path, doc_config_hash, page_config_hash
            )

            # Mark parsing phase complete
            stats.end_parsing()

            # Update document status
            document.status = ProcessingStatus.PENDING  # Ready for page parsing

            cache_status = "from cache" if use_cached_images else "newly rendered"
            logger.info(
                f"Successfully parsed document with {document.metadata['page_count']} pages ({cache_status})"
            )

            return document

        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            # Create a dummy document to mark as failed
            dummy_doc = Document(
                id=str(uuid.uuid4()), source_path=document_path, status=ProcessingStatus.FAILED
            )
            dummy_doc.mark_failed(str(e))
            raise DocumentParsingError(f"Failed to parse document: {e}") from e

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

            # Apply max_dimension resizing if specified
            if self.max_dimension:
                # Get current dimensions
                current_width = pix.width
                current_height = pix.height
                max_current = max(current_width, current_height)

                # Check if resizing is needed
                if max_current > self.max_dimension:
                    # Calculate scaling factor
                    scale_factor = self.max_dimension / max_current
                    new_width = int(current_width * scale_factor)
                    new_height = int(current_height * scale_factor)

                    # Create scaled pixmap using PyMuPDF's scaling constructor
                    # Pixmap(src, width, height) - creates a scaled copy
                    scaled_pix = pymupdf.Pixmap(pix, new_width, new_height)

                    logger.debug(
                        f"Resized page {page_number} from {current_width}x{current_height} "
                        f"to {new_width}x{new_height} (max_dimension={self.max_dimension})"
                    )

                    # Replace original pixmap with scaled one
                    pix = scaled_pix

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
            raise DocumentParsingError(f"Failed to parse page {page_number}: {e}") from e

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
