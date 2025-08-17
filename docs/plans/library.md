# pdf2markdown Library Interface Design

## Executive Summary

This document outlines the plan to refactor the pdf2markdown project to be consumable as both a Python library and a CLI tool. The primary goal is to enable other applications to integrate pdf2markdown's powerful PDF processing capabilities programmatically while maintaining backward compatibility with the existing CLI interface.

## Current State Analysis

### Existing Architecture
- **CLI-focused**: Primary interface through `__main__.py` using Click
- **Configuration**: YAML-based configuration with environment variable support
- **Pipeline**: Async pipeline with queue-based processing
- **Components**: Modular parsers, LLM providers, and validators

### Key Strengths to Preserve
- Modular architecture with clear separation of concerns
- Async/await pattern for efficient processing
- Pluggable LLM provider system
- Sophisticated validation and correction pipeline
- Queue-based parallel processing

### Current Limitations for Library Use
1. Configuration tightly coupled to YAML files and environment variables
2. No programmatic API for pipeline initialization
3. Output primarily designed for file writing
4. Progress tracking tied to CLI display
5. Error handling optimized for CLI reporting

## Design Goals

1. **Zero Breaking Changes**: Maintain 100% backward compatibility with existing CLI
2. **Pythonic API**: Provide intuitive, well-documented Python interfaces
3. **Flexible Configuration**: Support both dictionary-based and YAML configurations
4. **Multiple Output Formats**: Support string returns, streaming, and file writing
5. **Library-Friendly Logging**: Configurable logging that doesn't interfere with host applications
6. **Async and Sync APIs**: Provide both for different use cases

## Proposed Architecture

### 1. Core API Module Structure

```
src/pdf2markdown/
├── api/                      # New library API module
│   ├── __init__.py          # Public API exports
│   ├── converter.py         # Main converter class
│   ├── config.py           # Configuration builder
│   ├── exceptions.py       # API-specific exceptions
│   └── types.py           # Type definitions
├── cli/                     # Refactored CLI (moved from __main__.py)
│   └── main.py
└── [existing modules...]
```

### 2. Public API Design

#### 2.1 Simple High-Level API

```python
from pdf2markdown import PDFConverter

# Simple usage with defaults
converter = PDFConverter()
markdown = await converter.convert("document.pdf")

# With configuration
converter = PDFConverter(config={
    "llm_provider": {
        "provider_type": "openai",
        "api_key": "sk-...",
        "model": "gpt-4o-mini"
    },
    "document_parser": {
        "resolution": 300
    }
})
markdown = await converter.convert("document.pdf")

# Sync wrapper for convenience
markdown = converter.convert_sync("document.pdf")
```

#### 2.2 Advanced Component-Based API

```python
from pdf2markdown import (
    DocumentParser,
    PageParser,
    LLMProvider,
    Pipeline,
    Config
)

# Build configuration programmatically
config = Config.builder() \
    .with_llm_provider("openai", api_key="sk-...") \
    .with_resolution(300) \
    .with_page_workers(10) \
    .build()

# Or from dictionary
config = Config.from_dict({...})

# Or from YAML (backward compatibility)
config = Config.from_yaml("config.yaml")

# Initialize components
doc_parser = DocumentParser.create(config.document_parser)
page_parser = PageParser.create(config.page_parser)
llm_provider = LLMProvider.create(config.llm_provider)

# Create pipeline
pipeline = Pipeline(
    document_parser=doc_parser,
    page_parser=page_parser,
    llm_provider=llm_provider,
    config=config.pipeline
)

# Process document
document = await pipeline.process("document.pdf")

# Access results
for page in document.pages:
    print(f"Page {page.page_number}: {len(page.content)} characters")

# Get full markdown
markdown = document.to_markdown()

# Or write to file
document.save("output.md")
```

#### 2.3 Streaming API

```python
from pdf2markdown import PDFConverter

converter = PDFConverter(config={...})

# Stream pages as they're processed
async for page in converter.stream_pages("document.pdf"):
    print(f"Processed page {page.page_number}")
    # Process page content immediately
    
# Stream with progress callback
async def progress_callback(current, total, message):
    print(f"Progress: {current}/{total} - {message}")

markdown = await converter.convert(
    "document.pdf",
    progress_callback=progress_callback
)
```

### 3. Configuration Management

#### 3.1 Configuration Builder Pattern

```python
class ConfigBuilder:
    def with_llm_provider(self, provider_type: str, **kwargs) -> 'ConfigBuilder':
        """Configure LLM provider."""
        
    def with_openai(self, api_key: str, model: str = "gpt-4o-mini", **kwargs) -> 'ConfigBuilder':
        """Convenience method for OpenAI configuration."""
        
    def with_transformers(self, model_name: str, device: str = "auto", **kwargs) -> 'ConfigBuilder':
        """Convenience method for Transformers configuration."""
        
    def with_resolution(self, dpi: int) -> 'ConfigBuilder':
        """Set PDF rendering resolution."""
        
    def with_page_workers(self, workers: int) -> 'ConfigBuilder':
        """Set number of parallel page workers."""
        
    def with_validators(self, validators: List[str]) -> 'ConfigBuilder':
        """Configure validation pipeline."""
        
    def build(self) -> Config:
        """Build and validate configuration."""
```

#### 3.2 Configuration Sources Priority

1. Programmatic (dictionary/builder)
2. YAML file
3. Environment variables
4. Defaults

### 4. Error Handling

```python
# Library-specific exceptions
class PDFConversionError(Exception):
    """Base exception for conversion errors."""

class ConfigurationError(PDFConversionError):
    """Invalid configuration."""

class ParsingError(PDFConversionError):
    """PDF parsing failed."""
    
class LLMError(PDFConversionError):
    """LLM provider error."""
    
class ValidationError(PDFConversionError):
    """Content validation failed."""
```

### 5. Logging Strategy

```python
# Library uses named loggers
logger = logging.getLogger("pdf2markdown.converter")

# Host application can configure as needed
logging.getLogger("pdf2markdown").setLevel(logging.WARNING)

# Or suppress entirely
logging.getLogger("pdf2markdown").addHandler(logging.NullHandler())
```

## Implementation Plan

### Phase 1: Core Refactoring (Week 1)
1. Create `api/` module structure
2. Extract configuration logic from `Settings` class
3. Implement `ConfigBuilder` and dictionary-based configuration
4. Create base `PDFConverter` class with async methods

### Phase 2: API Development (Week 2)
1. Implement high-level converter API
2. Add sync wrappers using `asyncio.run()`
3. Implement streaming methods
4. Create comprehensive type hints

### Phase 3: CLI Refactoring (Week 3)
1. Move CLI logic to `cli/main.py`
2. Update CLI to use new library API
3. Ensure backward compatibility
4. Update entry points in `pyproject.toml`

### Phase 4: Testing & Documentation (Week 4)
1. Create library usage examples
2. Write API documentation
3. Add integration tests for library usage
4. Update README with library usage section

## Testing Strategy

### Unit Tests
- Test each API method independently
- Mock LLM providers and file I/O
- Verify configuration handling

### Integration Tests
```python
def test_library_basic_usage():
    converter = PDFConverter()
    result = converter.convert_sync("test.pdf")
    assert isinstance(result, str)
    assert "expected_content" in result

def test_library_with_config():
    config = {"llm_provider": {...}}
    converter = PDFConverter(config=config)
    # ...

def test_streaming_api():
    converter = PDFConverter()
    pages = []
    async for page in converter.stream_pages("test.pdf"):
        pages.append(page)
    assert len(pages) > 0
```

### Backward Compatibility Tests
- Ensure all existing CLI commands work
- Verify YAML configuration still works
- Test environment variable overrides

## Documentation Requirements

### API Reference
- Comprehensive docstrings for all public methods
- Type hints for all parameters and returns
- Usage examples in docstrings

### User Guide
```markdown
# Using pdf2markdown as a Library

## Installation
pip install pdf2markdown

## Quick Start
[Basic usage example]

## Configuration
[Configuration options and examples]

## Advanced Usage
[Component-based usage, streaming, etc.]

## API Reference
[Auto-generated from docstrings]
```

## Migration Guide for CLI Users

No changes required. The CLI interface remains exactly the same:

```bash
pdf2markdown input.pdf -o output.md --config config.yaml
```

## Performance Considerations

1. **Memory Management**: Stream large documents page-by-page
2. **Connection Pooling**: Reuse LLM provider connections
3. **Async Processing**: Maintain async throughout for efficiency
4. **Resource Cleanup**: Proper context managers for all resources

## Security Considerations

1. **API Key Handling**: Never log or expose API keys
2. **Input Validation**: Validate all configuration inputs
3. **File Access**: Respect file system permissions
4. **Dependency Security**: Regular security updates

---

## Appendix: Adding Support for PDFKB-MCP

### Integration Overview

The PDFKB-MCP project can leverage pdf2markdown as a high-quality PDF parser to replace or supplement its existing parser implementations. Here's how to integrate it:

### 1. Installation

Add pdf2markdown as a dependency in PDFKB-MCP's `pyproject.toml`:

```toml
[project.dependencies]
pdf2markdown = "^1.0.0"
```

### 2. Create Parser Adapter

Create a new parser at `src/pdfkb/parsers/parser_pdf2markdown.py`:

```python
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio

from pdf2markdown import PDFConverter, Config
from ..base_parser import BaseParser, ParsedDocument

class PDFToMarkdownParser(BaseParser):
    """High-quality PDF parser using pdf2markdown library."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Build pdf2markdown configuration
        self.config = self._build_config(config or {})
        self.converter = PDFConverter(config=self.config)
    
    def _build_config(self, pdfkb_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert PDFKB config to pdf2markdown config."""
        
        # Map PDFKB config to pdf2markdown config
        config = {
            "llm_provider": {
                "provider_type": "openai",
                "endpoint": pdfkb_config.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
                "api_key": pdfkb_config.get("openrouter_api_key"),
                "model": pdfkb_config.get("llm_model", "google/gemini-2.0-flash-exp:free"),
                "max_tokens": pdfkb_config.get("max_tokens", 8192),
                "temperature": 0.1,
            },
            "document_parser": {
                "resolution": pdfkb_config.get("resolution", 300),
                "cache_dir": pdfkb_config.get("cache_dir", "/tmp/pdf2markdown_cache"),
            },
            "page_parser": {
                "validate_content": True,
                "validation": {
                    "validators": ["markdown", "repetition"],
                    "max_correction_attempts": 2,
                }
            },
            "pipeline": {
                "page_workers": pdfkb_config.get("page_workers", 10),
                "enable_progress": False,  # Disable for library usage
            }
        }
        
        # Use local provider if specified
        if pdfkb_config.get("use_local_llm", False):
            config["llm_provider"] = {
                "provider_type": "transformers",
                "model_name": pdfkb_config.get("local_model", "microsoft/Phi-3.5-vision-instruct"),
                "device": "cuda" if pdfkb_config.get("use_gpu", False) else "cpu",
                "torch_dtype": "float16",
                "max_tokens": 4096,
            }
        
        return config
    
    async def parse_async(self, file_path: Path) -> ParsedDocument:
        """Parse PDF file asynchronously."""
        try:
            # Convert PDF to markdown
            markdown_content = await self.converter.convert(str(file_path))
            
            # Extract metadata during conversion
            metadata = {
                "parser": "pdf2markdown",
                "llm_provider": self.config["llm_provider"]["provider_type"],
                "model": self.config["llm_provider"].get("model", "N/A"),
                "resolution": self.config["document_parser"]["resolution"],
            }
            
            # Extract title from first H1 or filename
            title = self._extract_title(markdown_content) or file_path.stem
            
            return ParsedDocument(
                content=markdown_content,
                metadata=metadata,
                title=title,
                page_count=self._count_pages(markdown_content),
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF with pdf2markdown: {e}")
    
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse PDF file synchronously."""
        return asyncio.run(self.parse_async(file_path))
    
    def _extract_title(self, markdown: str) -> Optional[str]:
        """Extract title from markdown content."""
        lines = markdown.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if line.startswith('# '):
                return line[2:].strip()
        return None
    
    def _count_pages(self, markdown: str) -> int:
        """Count pages based on page markers."""
        # pdf2markdown uses page separators
        return markdown.count('--[PAGE:') + 1
    
    @property
    def supported_extensions(self) -> list[str]:
        """Return supported file extensions."""
        return ['.pdf']
    
    @property
    def name(self) -> str:
        """Return parser name."""
        return "pdf2markdown"
    
    @property
    def description(self) -> str:
        """Return parser description."""
        return "High-quality LLM-based PDF parser with validation and correction"
```

### 3. Register Parser in Factory

Update `src/pdfkb/parser_factory.py`:

```python
from .parsers.parser_pdf2markdown import PDFToMarkdownParser

AVAILABLE_PARSERS = {
    "pymupdf4llm": PyMuPDF4LLMParser,
    "marker": MarkerParser,
    "mineru": MinerUParser,
    "docling": DoclingParser,
    "llm": LLMParser,
    "pdf2markdown": PDFToMarkdownParser,  # Add new parser
    "markdown": MarkdownParser,
}

# Set as default for highest quality
DEFAULT_PDF_PARSER = "pdf2markdown"
```

### 4. Configuration Integration

Add configuration options to PDFKB-MCP's environment variables:

```python
# In src/pdfkb/config.py

# pdf2markdown Parser Settings
PDF2MARKDOWN_PROVIDER: str = Field(
    default="openai",
    description="LLM provider for pdf2markdown (openai/transformers)"
)
PDF2MARKDOWN_MODEL: str = Field(
    default="gpt-4o-mini",
    description="Model to use for pdf2markdown"
)
PDF2MARKDOWN_RESOLUTION: int = Field(
    default=300,
    description="DPI resolution for PDF rendering"
)
PDF2MARKDOWN_PAGE_WORKERS: int = Field(
    default=10,
    description="Number of parallel page workers"
)
PDF2MARKDOWN_VALIDATE: bool = Field(
    default=True,
    description="Enable content validation and correction"
)
```

### 5. Usage in PDFKB-MCP

The parser will automatically be available through the existing document processing pipeline:

```python
# In document processor
parser = parser_factory.get_parser("pdf2markdown", config)
parsed_doc = await parser.parse_async(pdf_path)
```

### 6. Performance Optimization

For PDFKB-MCP's batch processing needs:

```python
class BatchPDFToMarkdownParser(PDFToMarkdownParser):
    """Optimized for batch processing multiple PDFs."""
    
    async def parse_batch(self, file_paths: List[Path]) -> List[ParsedDocument]:
        """Parse multiple PDFs efficiently."""
        # Reuse converter instance for connection pooling
        tasks = [self.parse_async(path) for path in file_paths]
        return await asyncio.gather(*tasks)
```

### 7. Caching Integration

Leverage PDFKB-MCP's intelligent cache:

```python
# Configure pdf2markdown to use PDFKB's cache directory
config = {
    "document_parser": {
        "cache_dir": str(PDFKB_CACHE_DIR / "pdf2markdown"),
    }
}
```

### Benefits for PDFKB-MCP

1. **Superior Quality**: LLM-based parsing with validation ensures high-quality markdown output
2. **Flexibility**: Support for both cloud (OpenAI-compatible) and local (Transformers) models
3. **Performance**: Parallel page processing and intelligent caching
4. **Reliability**: Built-in validation and correction pipeline
5. **Maintenance**: Leverages actively maintained pdf2markdown library

### Migration Strategy

1. **Phase 1**: Add as optional parser alongside existing ones
2. **Phase 2**: Run quality comparisons on test documents
3. **Phase 3**: Set as default parser if quality meets requirements
4. **Phase 4**: Optionally deprecate lower-quality parsers

### Configuration Example for PDFKB-MCP

```bash
# .env file for PDFKB-MCP with pdf2markdown
PDFKB_DEFAULT_PDF_PARSER=pdf2markdown
PDFKB_PDF2MARKDOWN_PROVIDER=openai
PDFKB_PDF2MARKDOWN_MODEL=gpt-4o-mini
PDFKB_PDF2MARKDOWN_RESOLUTION=300
PDFKB_PDF2MARKDOWN_PAGE_WORKERS=10
PDFKB_PDF2MARKDOWN_VALIDATE=true
OPENROUTER_API_KEY=sk-or-v1-...
```

This integration provides PDFKB-MCP with a best-in-class PDF parsing solution while maintaining its existing architecture and flexibility.