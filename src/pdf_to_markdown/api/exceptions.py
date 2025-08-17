"""Library-specific exceptions for pdf-to-markdown."""

from typing import Optional, Any, Dict


class PDFConversionError(Exception):
    """Base exception for conversion errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(PDFConversionError):
    """Invalid configuration."""
    pass


class ParsingError(PDFConversionError):
    """PDF parsing failed."""
    
    def __init__(self, message: str, page_number: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.page_number = page_number


class LLMError(PDFConversionError):
    """LLM provider error."""
    
    def __init__(self, message: str, provider: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.provider = provider


class ValidationError(PDFConversionError):
    """Content validation failed."""
    
    def __init__(self, message: str, issues: Optional[list] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.issues = issues or []