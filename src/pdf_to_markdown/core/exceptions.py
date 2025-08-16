"""Custom exceptions for PDF to Markdown converter."""


class PDFToMarkdownError(Exception):
    """Base exception for PDF to Markdown converter."""

    pass


class DocumentParsingError(PDFToMarkdownError):
    """Error during document parsing."""

    pass


class PageParsingError(PDFToMarkdownError):
    """Error during page parsing."""

    pass


class LLMConnectionError(PDFToMarkdownError):
    """Error connecting to LLM API."""

    pass


class QueueOverflowError(PDFToMarkdownError):
    """Queue has reached maximum capacity."""

    pass


class ConfigurationError(PDFToMarkdownError):
    """Invalid configuration."""

    pass


class FileNotFoundError(PDFToMarkdownError):
    """File not found."""

    pass


class InvalidFileFormatError(PDFToMarkdownError):
    """Invalid file format."""

    pass
