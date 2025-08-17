"""Public API for pdf-to-markdown library."""

from .converter import PDFConverter
from .config import Config, ConfigBuilder
from .types import (
    DocumentResult,
    PageResult,
    ConversionStatus,
    ProgressCallback,
    AsyncProgressCallback,
    ConfigDict
)
from .exceptions import (
    PDFConversionError,
    ConfigurationError,
    ParsingError,
    LLMError,
    ValidationError
)

# Public API exports
__all__ = [
    # Main converter
    'PDFConverter',
    
    # Configuration
    'Config',
    'ConfigBuilder',
    
    # Types
    'DocumentResult',
    'PageResult',
    'ConversionStatus',
    'ProgressCallback',
    'AsyncProgressCallback',
    'ConfigDict',
    
    # Exceptions
    'PDFConversionError',
    'ConfigurationError',
    'ParsingError',
    'LLMError',
    'ValidationError',
]