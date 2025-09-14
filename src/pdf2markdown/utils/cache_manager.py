"""Cache management utilities for caching and resuming document processing."""

import hashlib
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DocumentHasher:
    """Generates deterministic hashes for documents based on file and configuration."""

    @staticmethod
    def hash_document(document_path: Path, config_hash: str) -> str:
        """Generate a deterministic hash for a document.

        Args:
            document_path: Path to the PDF document
            config_hash: Hash of the configuration that affects processing

        Returns:
            Deterministic hash string for the document
        """
        if not document_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")

        # Get file metadata
        stat = document_path.stat()
        file_size = stat.st_size
        modified_time = stat.st_mtime

        # Create hash input
        hash_input = f"{document_path.absolute()}:{file_size}:{modified_time}:{config_hash}"

        # Generate SHA-256 hash
        document_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        logger.debug(f"Generated document hash {document_hash} for {document_path}")
        return document_hash


class ConfigHasher:
    """Generates hashes for configuration parameters that affect cache validity."""

    # Configuration fields that affect image rendering (invalidate image cache)
    DOCUMENT_CONFIG_FIELDS = ["resolution", "max_dimension", "timeout", "type"]

    # Configuration fields that affect markdown generation (invalidate markdown cache)
    PAGE_CONFIG_FIELDS = [
        "model",
        "temperature",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "table_format",
        "additional_instructions",
        "validate_content",
        "validation",
        "prompt_template",
    ]

    @classmethod
    def hash_document_config(cls, config: dict[str, Any]) -> str:
        """Hash configuration that affects document parsing (image rendering).

        Args:
            config: Document parser configuration

        Returns:
            Hash of configuration affecting image rendering
        """
        relevant_config = {}

        # Extract relevant document parser config
        doc_config = config.get("document_parser", {})
        for field in cls.DOCUMENT_CONFIG_FIELDS:
            if field in doc_config:
                relevant_config[f"doc.{field}"] = doc_config[field]

        return cls._hash_config_dict(relevant_config)

    @classmethod
    def hash_page_config(cls, config: dict[str, Any]) -> str:
        """Hash configuration that affects page parsing (markdown generation).

        Args:
            config: Full configuration including LLM provider and page parser

        Returns:
            Hash of configuration affecting markdown generation
        """
        relevant_config = {}

        # Extract relevant LLM provider config
        llm_config = config.get("llm_provider", {})
        for field in cls.PAGE_CONFIG_FIELDS:
            if field in llm_config:
                relevant_config[f"llm.{field}"] = llm_config[field]

        # Extract relevant page parser config
        page_config = config.get("page_parser", {})
        for field in cls.PAGE_CONFIG_FIELDS:
            if field in page_config:
                relevant_config[f"page.{field}"] = page_config[field]

        # Include validation configuration
        if "validation" in page_config:
            validation_config = page_config["validation"]
            relevant_config["page.validation"] = validation_config

        return cls._hash_config_dict(relevant_config)

    @classmethod
    def _hash_config_dict(cls, config_dict: dict[str, Any]) -> str:
        """Generate hash from configuration dictionary.

        Args:
            config_dict: Dictionary of configuration values

        Returns:
            SHA-256 hash of the configuration
        """
        # Sort keys for deterministic hashing
        sorted_items = sorted(config_dict.items())
        config_str = json.dumps(sorted_items, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]


class ImageCache:
    """Manages caching of rendered PDF page images."""

    def __init__(self, cache_dir: Path):
        """Initialize image cache.

        Args:
            cache_dir: Base directory for caching
        """
        self.cache_dir = cache_dir
        self.images_dir = cache_dir / "images"
        self.config_file = self.images_dir / ".render_config.json"

    def is_valid(self, config_hash: str) -> bool:
        """Check if cached images are valid for the given configuration.

        Args:
            config_hash: Hash of current document parser configuration

        Returns:
            True if cache is valid for this configuration
        """
        if not self.config_file.exists():
            return False

        try:
            with open(self.config_file) as f:
                cached_config = json.load(f)

            return (
                cached_config.get("config_hash") == config_hash
                and self.images_dir.exists()
                and len(list(self.images_dir.glob("page_*.png"))) > 0
            )
        except (json.JSONDecodeError, OSError):
            return False

    def save_config(self, config_hash: str, page_count: int) -> None:
        """Save configuration metadata for cache validation.

        Args:
            config_hash: Hash of document parser configuration
            page_count: Number of pages in the document
        """
        self.images_dir.mkdir(parents=True, exist_ok=True)

        config_data = {
            "config_hash": config_hash,
            "page_count": page_count,
            "created_at": datetime.now().isoformat(),
        }

        with open(self.config_file, "w") as f:
            json.dump(config_data, f, indent=2)

    def get_cached_images(self) -> list[Path]:
        """Get list of cached image files.

        Returns:
            List of cached image file paths, sorted by page number
        """
        if not self.images_dir.exists():
            return []

        images = list(self.images_dir.glob("page_*.png"))
        return sorted(images, key=lambda p: int(p.stem.split("_")[1]))

    def get_image_path(self, page_number: int) -> Path:
        """Get path for a specific page image.

        Args:
            page_number: Page number (1-indexed)

        Returns:
            Path where the page image should be stored
        """
        self.images_dir.mkdir(parents=True, exist_ok=True)
        return self.images_dir / f"page_{page_number:04d}.png"

    def invalidate(self) -> None:
        """Remove all cached images and configuration."""
        if self.images_dir.exists():
            shutil.rmtree(self.images_dir)
        logger.debug(f"Invalidated image cache at {self.images_dir}")


class MarkdownCache:
    """Manages caching of LLM-generated markdown content."""

    def __init__(self, cache_dir: Path):
        """Initialize markdown cache.

        Args:
            cache_dir: Base directory for caching
        """
        self.cache_dir = cache_dir
        self.markdown_dir = cache_dir / "markdown"
        self.config_file = self.markdown_dir / ".llm_config.json"

    def is_valid(self, config_hash: str) -> bool:
        """Check if cached markdown is valid for the given configuration.

        Args:
            config_hash: Hash of current page parser configuration

        Returns:
            True if cache is valid for this configuration
        """
        if not self.config_file.exists():
            logger.debug(f"Markdown cache config file not found: {self.config_file}")
            return False

        try:
            with open(self.config_file) as f:
                cached_config = json.load(f)

            cached_hash = cached_config.get("config_hash")
            has_files = (
                self.markdown_dir.exists() and len(list(self.markdown_dir.glob("page_*.md"))) > 0
            )

            is_valid = cached_hash == config_hash and has_files

            if not is_valid:
                if cached_hash != config_hash:
                    logger.debug(
                        f"Markdown cache invalid: config hash mismatch. Expected {config_hash}, got {cached_hash}"
                    )
                if not has_files:
                    logger.debug(
                        f"Markdown cache invalid: no cached files found in {self.markdown_dir}"
                    )
            else:
                cached_pages = len(list(self.markdown_dir.glob("page_*.md")))
                logger.debug(f"Markdown cache valid: {cached_pages} cached pages found")

            return is_valid
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Error reading markdown cache config: {e}")
            return False

    def save_config(self, config_hash: str, page_count: int) -> None:
        """Save configuration metadata for cache validation.

        Args:
            config_hash: Hash of page parser configuration
            page_count: Number of pages in the document
        """
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        config_data = {
            "config_hash": config_hash,
            "page_count": page_count,
            "created_at": datetime.now().isoformat(),
        }

        with open(self.config_file, "w") as f:
            json.dump(config_data, f, indent=2)

    def get_cached_markdown(self, page_number: int) -> str | None:
        """Get cached markdown content for a specific page.

        Args:
            page_number: Page number (1-indexed)

        Returns:
            Cached markdown content or None if not found
        """
        markdown_file = self.get_markdown_path(page_number)
        if markdown_file.exists():
            try:
                with open(markdown_file, encoding="utf-8") as f:
                    return f.read()
            except OSError:
                return None
        return None

    def save_markdown(self, page_number: int, content: str) -> None:
        """Save markdown content for a specific page.

        Args:
            page_number: Page number (1-indexed)
            content: Markdown content to save
        """
        markdown_file = self.get_markdown_path(page_number)
        markdown_file.parent.mkdir(parents=True, exist_ok=True)

        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(content)

    def get_markdown_path(self, page_number: int) -> Path:
        """Get path for a specific page markdown file.

        Args:
            page_number: Page number (1-indexed)

        Returns:
            Path where the page markdown should be stored
        """
        return self.markdown_dir / f"page_{page_number:04d}.md"

    def get_cached_pages(self) -> list[int]:
        """Get list of page numbers that have cached markdown.

        Returns:
            List of page numbers with cached markdown
        """
        if not self.markdown_dir.exists():
            return []

        markdown_files = list(self.markdown_dir.glob("page_*.md"))
        page_numbers = []
        for file in markdown_files:
            try:
                page_num = int(file.stem.split("_")[1])
                page_numbers.append(page_num)
            except (ValueError, IndexError):
                continue

        return sorted(page_numbers)

    def invalidate(self) -> None:
        """Remove all cached markdown and configuration."""
        if self.markdown_dir.exists():
            shutil.rmtree(self.markdown_dir)
        logger.debug(f"Invalidated markdown cache at {self.markdown_dir}")


class CacheManager:
    """Central manager for document processing caches."""

    def __init__(self, base_cache_dir: Path):
        """Initialize cache manager.

        Args:
            base_cache_dir: Base directory for all caches
        """
        self.base_cache_dir = base_cache_dir
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_document_cache_dir(self, document_hash: str) -> Path:
        """Get cache directory for a specific document.

        Args:
            document_hash: Hash of the document

        Returns:
            Cache directory for the document
        """
        return self.base_cache_dir / document_hash

    def get_image_cache(self, document_hash: str) -> ImageCache:
        """Get image cache for a document.

        Args:
            document_hash: Hash of the document

        Returns:
            ImageCache instance for the document
        """
        cache_dir = self.get_document_cache_dir(document_hash)
        return ImageCache(cache_dir)

    def get_markdown_cache(self, document_hash: str) -> MarkdownCache:
        """Get markdown cache for a document.

        Args:
            document_hash: Hash of the document

        Returns:
            MarkdownCache instance for the document
        """
        cache_dir = self.get_document_cache_dir(document_hash)
        return MarkdownCache(cache_dir)

    def save_document_metadata(
        self, document_hash: str, document_path: Path, doc_config_hash: str, page_config_hash: str
    ) -> None:
        """Save document metadata for cache management.

        Args:
            document_hash: Hash of the document
            document_path: Path to the source document
            doc_config_hash: Hash of document parser configuration
            page_config_hash: Hash of page parser configuration
        """
        cache_dir = self.get_document_cache_dir(document_hash)
        cache_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "document_hash": document_hash,
            "document_path": str(document_path.absolute()),
            "document_config_hash": doc_config_hash,
            "page_config_hash": page_config_hash,
            "created_at": datetime.now().isoformat(),
        }

        metadata_file = cache_dir / ".doc_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_document_metadata(self, document_hash: str) -> dict[str, Any] | None:
        """Load document metadata from cache.

        Args:
            document_hash: Hash of the document

        Returns:
            Document metadata dictionary or None if not found
        """
        cache_dir = self.get_document_cache_dir(document_hash)
        metadata_file = cache_dir / ".doc_metadata.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def cleanup_old_caches(self, max_age_days: int = 7) -> int:
        """Clean up old cache directories.

        Args:
            max_age_days: Maximum age of cache directories to keep

        Returns:
            Number of cache directories removed
        """
        if not self.base_cache_dir.exists():
            return 0

        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        removed_count = 0

        for cache_dir in self.base_cache_dir.iterdir():
            if cache_dir.is_dir():
                try:
                    # Check directory modification time
                    if cache_dir.stat().st_mtime < cutoff_time:
                        shutil.rmtree(cache_dir)
                        removed_count += 1
                        logger.debug(f"Removed old cache directory: {cache_dir}")
                except OSError as e:
                    logger.warning(f"Failed to remove cache directory {cache_dir}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old cache directories")

        return removed_count

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the cache.

        Returns:
            Dictionary with cache statistics
        """
        if not self.base_cache_dir.exists():
            return {"total_size": 0, "document_count": 0, "directories": []}

        total_size = 0
        document_count = 0
        directories = []

        for cache_dir in self.base_cache_dir.iterdir():
            if cache_dir.is_dir():
                document_count += 1
                dir_size = 0

                # Calculate directory size
                for file_path in cache_dir.rglob("*"):
                    if file_path.is_file():
                        try:
                            dir_size += file_path.stat().st_size
                        except OSError:
                            pass

                total_size += dir_size

                directories.append(
                    {
                        "hash": cache_dir.name,
                        "size": dir_size,
                        "modified": cache_dir.stat().st_mtime,
                    }
                )

        return {
            "total_size": total_size,
            "document_count": document_count,
            "directories": sorted(directories, key=lambda d: d["modified"], reverse=True),
        }

    def clear_cache(self, document_hash: str | None = None) -> None:
        """Clear cache for a specific document or all documents.

        Args:
            document_hash: Hash of specific document to clear, or None to clear all
        """
        if document_hash:
            cache_dir = self.get_document_cache_dir(document_hash)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared cache for document {document_hash}")
        else:
            if self.base_cache_dir.exists():
                shutil.rmtree(self.base_cache_dir)
                self.base_cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all caches")
