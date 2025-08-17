"""Library-specific exceptions for pdf2markdown."""

from typing import Any


class PDFConversionError(Exception):
    """Base exception for conversion errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(PDFConversionError):
    """Invalid configuration."""

    pass


class ParsingError(PDFConversionError):
    """PDF parsing failed."""

    def __init__(
        self,
        message: str,
        page_number: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.page_number = page_number


class LLMError(PDFConversionError):
    """LLM provider error."""

    def __init__(
        self, message: str, provider: str | None = None, details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)
        self.provider = provider


class ValidationError(PDFConversionError):
    """Content validation failed."""

    def __init__(
        self, message: str, issues: list | None = None, details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)
        self.issues = issues or []
