"""PDF to Markdown converter using LLMs.

This package provides both a command-line interface and a Python library API
for converting PDF documents to Markdown format using Large Language Models.

Library Usage:
    from pdf2markdown import PDFConverter

    # Simple usage
    converter = PDFConverter()
    markdown = converter.convert_sync("document.pdf")

    # With configuration
    from pdf2markdown import ConfigBuilder

    config = ConfigBuilder() \\
        .with_openai(api_key="sk-...") \\
        .with_resolution(400) \\
        .with_page_workers(20) \\
        .build()

    converter = PDFConverter(config=config)
    markdown = converter.convert_sync("document.pdf", "output.md")
"""

__version__ = "0.3.0"

# Export the public API
from .api import (
    Config,
    ConfigBuilder,
    ConfigurationError,
    ConversionStatus,
    DocumentResult,
    LLMError,
    PageResult,
    ParsingError,
    PDFConversionError,
    PDFConverter,
    ValidationError,
)

__all__ = [
    # Version
    "__version__",
    # Main converter
    "PDFConverter",
    # Configuration
    "Config",
    "ConfigBuilder",
    # Types
    "DocumentResult",
    "PageResult",
    "ConversionStatus",
    # Exceptions
    "PDFConversionError",
    "ConfigurationError",
    "ParsingError",
    "LLMError",
    "ValidationError",
]
