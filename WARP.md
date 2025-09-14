# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

`pdf2markdown` is a Python library and CLI application that leverages Large Language Models (LLMs) to accurately convert technical PDF documents into well-structured Markdown. The project provides both a programmatic Python API and a command-line interface, designed specifically for technical documents like semiconductor datasheets.

## Development Commands

### Environment Setup
```bash
# Install Hatch (project dependency manager)
pipx install hatch

# Create development environment
hatch env create

# Activate the development environment
hatch shell
```

### Running the Application
```bash
# Basic usage
pdf2markdown input.pdf -o output.md

# With custom configuration
pdf2markdown input.pdf --config config/default.yaml

# Save configuration for reuse
pdf2markdown input.pdf --save-config my-config.yaml

# Process multiple files
pdf2markdown *.pdf -o /output/directory/

# Using specific model and resolution
pdf2markdown input.pdf --model gpt-4o --resolution 400

# Resume interrupted processing
pdf2markdown input.pdf --resume
```

### Testing
```bash
# Run all tests
hatch run test

# Run with coverage
hatch run test-cov

# Coverage report
hatch run cov-report

# Full coverage workflow
hatch run cov

# Run specific test
hatch run pytest tests/test_library_api.py
```

### Code Quality
```bash
# Format code (Black + Ruff)
hatch run format

# Lint code
hatch run lint

# Type checking
hatch run typecheck
```

### Version Management
**CRITICAL**: Never manually change version numbers. Always use bump-my-version:

```bash
# For bug fixes and small changes
hatch run bump-patch    # 0.1.0 → 0.1.1

# For new features  
hatch run bump-minor    # 0.1.0 → 0.2.0

# For breaking changes
hatch run bump-major    # 0.1.0 → 1.0.0
```

## Architecture Overview

The application uses a modular, pipeline-based architecture with several key components:

### Core Architecture
- **Two-Phase Processing**: PDF → Images (Document Parser) → Markdown (Page Parser)
- **Queue-Based Pipeline**: Asynchronous processing with configurable worker pools
- **Pluggable LLM Providers**: Abstract interface supporting OpenAI-compatible APIs and local models
- **Validation Pipeline**: Extensible system with markdown syntax and repetition validators

### Key Components

#### 1. LLM Provider System (`src/pdf2markdown/llm_providers/`)
- **Base Interface**: `LLMProvider` abstract class with methods like `invoke_with_image()`
- **OpenAI Provider**: Supports any OpenAI-compatible endpoint (OpenAI, Azure, local servers)
- **Transformers Provider**: Runs models locally using HuggingFace Transformers
- **Factory Pattern**: `create_llm_provider()` for easy instantiation

#### 2. Parser System (`src/pdf2markdown/parsers/`)
- **Document Parser**: Converts PDF to page images using PyMuPDF (`SimpleDocumentParser`)
- **Page Parser**: Converts images to Markdown using LLM (`SimpleLLMPageParser`)
- **Abstract Base Classes**: `DocumentParser` and `PageParser` for extensibility

#### 3. Pipeline System (`src/pdf2markdown/pipeline/`)
- **Queue Manager**: Handles document, page, and output queues with priority support
- **Workers**: Document workers (sequential) and page workers (parallel processing)
- **Coordinator**: Orchestrates the entire processing pipeline
- **Progress Tracking**: Real-time progress with Rich terminal output

#### 4. Validation System (`src/pdf2markdown/validators/`)
- **Markdown Validator**: Syntax validation using PyMarkdown with LLM correction
- **Repetition Validator**: Detects and corrects various types of content repetition
- **Extensible Framework**: Easy to add custom validators

#### 5. Configuration System (`src/pdf2markdown/config/`)
- **Pydantic Models**: Type-safe configuration with validation
- **Hierarchical Loading**: Default values → YAML config → Environment variables → CLI args
- **Builder Pattern**: `ConfigBuilder` for programmatic configuration

#### 6. Library API (`src/pdf2markdown/api/`)
- **PDFConverter**: Main entry point with sync/async methods
- **Streaming Support**: Process large documents page-by-page
- **Batch Processing**: Handle multiple PDFs efficiently
- **Context Managers**: Proper resource cleanup

### Data Flow
1. **Input**: PDF document and configuration
2. **Document Parsing**: PDF pages rendered to high-resolution images (cached)
3. **Queue Distribution**: Pages distributed to worker pool
4. **Page Processing**: Each page image sent to LLM for Markdown conversion
5. **Validation**: Generated content validated and optionally corrected
6. **Assembly**: Processed pages combined into final Markdown document
7. **Output**: Clean Markdown file with configurable page separators

## Configuration

### Initial Setup
```bash
# Copy sample configuration
cp config/default.sample.yaml config/default.yaml

# Edit configuration (heavily documented with examples)
nano config/default.yaml

# Set API key via environment variable (recommended)
export OPENAI_API_KEY="your-api-key-here"
```

### Key Configuration Sections
- **llm_provider**: LLM API settings (endpoint, model, API key, penalties)
- **document_parser**: PDF rendering settings (resolution, caching, timeouts)  
- **page_parser**: Markdown conversion settings (table format, validation, templates)
- **pipeline**: Worker counts and queue sizes
- **cache**: Caching system configuration
- **validation**: Content validation and correction settings

### Environment Variables
- `OPENAI_API_KEY`: Your LLM API key (required)
- `PDF2MARKDOWN_CACHE_DIR`: Cache directory override
- `PDF2MARKDOWN_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Performance & Caching

### Smart Caching System
The application includes sophisticated caching that dramatically improves performance:
- **Image Cache**: PDF page renders cached based on content and settings
- **Markdown Cache**: LLM-generated content cached based on configuration
- **Smart Invalidation**: Automatic cache invalidation when configs change
- **Resume Support**: Interrupted processing can resume using cached data

### Performance Tuning
- **Page Workers**: Default 10, increase for faster processing (20+ on powerful systems)
- **Resolution**: 300 DPI default, lower (200) for speed, higher (400) for quality
- **Max Dimension**: Optional pixel limit to control memory usage
- **Local Models**: Set `page_workers: 1` for memory-intensive local models

## Testing Strategy

### Test Organization
- **Unit Tests**: Individual component testing (`tests/unit/`)
- **Integration Tests**: Parser and pipeline integration (`tests/integration/`)
- **Library API Tests**: End-to-end API testing (`tests/test_library_api.py`)
- **Fixtures**: Shared test data and configurations (`tests/fixtures/`)

### Running Specific Tests
```bash
# Test specific component
hatch run pytest tests/test_config.py

# Test with verbose output
hatch run pytest -v tests/test_models.py

# Test with coverage for specific file
hatch run pytest --cov=src/pdf2markdown/api tests/test_library_api.py
```

## Code Style & Conventions

### Standards
- **Type Hints**: Required for all functions and methods
- **Docstrings**: Google-style docstrings for all public APIs
- **Formatting**: Black formatter with 100-character line length
- **Linting**: Ruff with comprehensive rule set
- **Error Handling**: Custom exception hierarchy with detailed messages

### Key Rules from CLAUDE.md
- Never manually edit version numbers - use `hatch run bump-*` commands
- Always update HLD and LLD documents in `docs/` when making design changes
- Update README.md when adding/changing features or usage patterns
- Use Hatch for all dependency management and application execution
- Maintain the modular architecture with clear separation of concerns

## Development Workflows

### Adding New Features
1. Update relevant documentation in `docs/` if architectural changes
2. Implement feature following existing patterns
3. Add comprehensive tests
4. Update README.md if user-facing changes
5. Use `hatch run bump-minor` for version increment

### Adding New LLM Providers
1. Inherit from `LLMProvider` base class
2. Implement all abstract methods (`invoke_with_image`, `validate_config`, etc.)
3. Add to factory in `llm_providers/factory.py`
4. Add configuration schema to `config/schemas.py`
5. Update documentation with provider-specific instructions

### Adding New Validators
1. Inherit from `BaseValidator` in `validators/base.py`
2. Implement `validate()` and `create_correction_instructions()` methods
3. Add to validator factory
4. Add configuration options to schemas
5. Include in default validation pipeline if generally useful

## Deployment Considerations

### Local Development
- Use `hatch shell` for isolated development environment
- Configure local LLM servers using OpenAI-compatible endpoints
- See `LOCAL.md` for vLLM and llama.cpp server configurations

### Production Deployment
- Docker containerization recommended
- Environment-based configuration (never hardcode API keys)
- Configure log rotation and monitoring
- Consider using local models for sensitive document processing
- Set appropriate worker counts based on available resources

### CI/CD Pipeline
- **CI Workflow**: Runs on push/PR with multi-version Python testing
- **Release Workflow**: Automatic PyPI publication on version tags
- **Version Tags**: Created automatically by bump-my-version commands

This architecture enables high-quality PDF to Markdown conversion while maintaining flexibility, performance, and extensibility for various deployment scenarios and LLM providers.