"""Public API for pdf2markdown library."""

from .config import Config, ConfigBuilder
from .converter import PDFConverter
from .exceptions import (
    ConfigurationError,
    LLMError,
    ParsingError,
    PDFConversionError,
    ValidationError,
)
from .types import (
    AsyncProgressCallback,
    ConfigDict,
    ConversionStatus,
    DocumentResult,
    PageResult,
    ProgressCallback,
)

# Public API exports
__all__ = [
    # Main converter
    "PDFConverter",
    # Configuration
    "Config",
    "ConfigBuilder",
    # Types
    "DocumentResult",
    "PageResult",
    "ConversionStatus",
    "ProgressCallback",
    "AsyncProgressCallback",
    "ConfigDict",
    # Exceptions
    "PDFConversionError",
    "ConfigurationError",
    "ParsingError",
    "LLMError",
    "ValidationError",
]
