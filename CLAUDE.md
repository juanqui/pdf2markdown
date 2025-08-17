# pdf2markdown Project Development Guide

## Project Overview
`pdf2markdown` is a Python application that leverages Large Language Models (LLMs) to accurately convert technical PDF documents (such as semiconductor datasheets) into well-structured Markdown documents.

## Architecture Summary

### Core Components
1. **Document Parser** (`SimpleDocumentParser`)
   - Uses PyMuPDF to render PDF pages as PNG images
   - Configurable resolution (default 300 DPI)
   - Caches rendered images in temporary directory
   - Sequential processing (one document at a time)

2. **Page Parser** (`SimpleLLMPageParser`)
   - Uses pluggable LLM providers to convert images to Markdown
   - Accepts any `LLMProvider` implementation
   - Uses Jinja2 templates for prompts
   - Handles tables, equations, images, watermarks, and formatting

3. **LLM Provider System** (`LLMProvider`)
   - Abstract interface for LLM communication
   - `OpenAILLMProvider`: Supports any OpenAI-compatible API endpoint
   - Supports GPT-4o-mini and other vision models
   - Extensible for future providers (Transformers, Ollama, Anthropic, etc.)

4. **Pipeline System**
   - Queue-based architecture with multiple worker types
   - 1 document worker (sequential requirement)
   - N page workers (parallel processing, default 10)
   - Progress tracking with tqdm
   - Automatic retry with exponential backoff

## Technology Stack
- **Python 3.10+**: Core language
- **Hatch**: Project and dependency management
- **PyMuPDF**: PDF rendering to images
- **OpenAI Python SDK**: LLM integration
- **Pydantic**: Configuration validation
- **asyncio**: Asynchronous processing
- **Click**: CLI framework
- **Rich**: Enhanced terminal output

## Project Structure
```
pdf2markdown/
├── src/pdf2markdown/
│   ├── core/              # Data models and interfaces
│   ├── parsers/            # Document and page parsers
│   ├── llm_providers/      # LLM provider implementations
│   ├── pipeline/           # Queue-based processing
│   ├── config/             # Configuration management
│   ├── templates/prompts/  # LLM prompt templates
│   └── utils/              # Utilities
├── config/                 # Configuration files
├── tests/                  # Test suites
└── pyproject.toml          # Project configuration
```

## Key Design Decisions

### 1. Modular Parser Architecture
- Abstract base classes for `DocumentParser` and `PageParser`
- Easy to extend with new implementations
- Current implementations: `SimpleDocumentParser`, `SimpleLLMPageParser`
- Abstract `LLMProvider` interface for pluggable LLM backends

### 2. Queue-Based Pipeline
- Separate queues for documents, pages, and output
- Priority queue support for processing order
- Configurable worker counts
- Error queue for failed tasks

### 3. Configuration Management
- Pydantic models for type-safe configuration
- YAML configuration files
- Environment variable overrides
- Hierarchical configuration structure

### 4. LLM Integration
- Pluggable LLM provider system
- Base64 image encoding for API calls
- Retry logic with exponential backoff
- Configurable prompts via Jinja2 templates
- Support for multiple provider types:
  - OpenAI-compatible endpoints
  - Future: Local models (Transformers/HuggingFace)
  - Future: Ollama, Anthropic, etc.

### 5. Validation Pipeline
- Extensible validator system with base `BaseValidator` class
- Multiple validators running in sequence:
  - **MarkdownValidator**: Syntax and formatting validation
  - **RepetitionValidator**: Detects various types of unwanted repetition
- Automatic correction by re-prompting LLM with specific instructions
- Configurable validators via YAML
- Easy to extend with custom validators

## Development Commands

### Environment Setup
```bash
# Install Hatch
pipx install hatch

# Install dependencies
hatch env create

# Activate environment
hatch shell
```

### Running the Application
```bash
# Basic usage
pdf2markdown input.pdf -o output.md

# With custom configuration
pdf2markdown input.pdf --config config.yaml

# With specific model
pdf2markdown input.pdf --model gpt-4o --resolution 400

# Save configuration
pdf2markdown input.pdf --save-config my-config.yaml
```

### Testing
```bash
# Run all tests
hatch run test

# Run with coverage
hatch run test-cov

# Run specific test
hatch run pytest tests/test_parsers.py
```

### Code Quality
```bash
# Format code
hatch run format

# Lint code
hatch run lint

# Type checking
hatch run typecheck
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for LLM API access
- `OPENAI_API_ENDPOINT`: Optional custom endpoint
- `OPENAI_MODEL`: Model to use (default: gpt-4o-mini)
- `PDF2MARKDOWN_CACHE_DIR`: Cache directory for images
- `PDF2MARKDOWN_OUTPUT_DIR`: Default output directory
- `PDF2MARKDOWN_LOG_LEVEL`: Logging level

### Configuration File (YAML)
```yaml
document_parser:
  resolution: 300
  cache_dir: /tmp/pdf2markdown/cache
  
page_parser:
  llm_provider:
    provider_type: openai
    endpoint: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o-mini
    temperature: 0.1
    max_tokens: 4096
  prompt_template: templates/prompts/ocr_extraction.j2
  
pipeline:
  page_workers: 10
  enable_progress: true
```

## LLM Prompt Template
The prompt template (`ocr_extraction.j2`) instructs the LLM to:
- Extract tables as HTML
- Format equations in LaTeX
- Preserve document structure (headings, lists, emphasis)
- Handle special elements (watermarks, page numbers, checkboxes)
- Describe images when captions are absent

## Performance Considerations

### Optimization Strategies
1. **Image Caching**: Rendered pages cached to avoid re-rendering
2. **Parallel Page Processing**: Multiple workers process pages concurrently
3. **Batch Processing**: Pages processed in configurable batches
4. **Connection Pooling**: Reuse HTTP connections to LLM API
5. **Automatic Cleanup**: Old cache files removed after 24 hours

### Resource Management
- Memory-aware queue sizes
- Configurable timeouts for API calls
- Maximum page size limits
- Graceful error handling and recovery

## Error Handling
- Custom exception hierarchy
- Retry logic for transient failures
- Error queue for failed tasks
- Detailed logging at multiple levels
- Progress preservation on failure

## Validation System Details

### BaseValidator Interface
All validators inherit from `BaseValidator` and implement:
- `validate(content, page)`: Validate content and return issues
- `create_correction_instructions(issues)`: Generate LLM correction prompts
- `get_rule_prefix()`: Return rule ID prefix (e.g., "MD", "REP")

### RepetitionValidator Detection Strategies
1. **Consecutive Duplicates**: Lines repeated N times in a row
2. **Window Duplicates**: Lines appearing multiple times within sliding window
3. **Normalized Duplicates**: Similar lines ignoring whitespace/punctuation
4. **Paragraph Duplicates**: Entire paragraphs that repeat
5. **Pattern Detection**: Repetitive structural patterns

### Validation Pipeline Flow
1. Initial content extraction from LLM
2. Run all configured validators
3. Collect all issues from validators
4. If issues found and correction enabled:
   - Create combined correction prompt
   - Re-prompt LLM with specific instructions
   - Validate corrected content
   - Repeat up to max_correction_attempts
5. Use best version (fewest issues)

## Future Enhancements
1. **Additional LLM Provider Implementations**
   - Local LLM support using Transformers/HuggingFace
   - Ollama integration for local models
   - Anthropic Claude API support
   - Google Gemini API support
   - Custom inference endpoints

2. **Additional Parser Implementations**
   - OCR-based parsers (Tesseract, EasyOCR)
   - Hybrid approaches (OCR + LLM)

3. **Additional Validators**
   - **StructureValidator**: Ensure headers follow logical hierarchy
   - **CompletionValidator**: Detect if content seems truncated
   - **LanguageValidator**: Detect if LLM switched languages mid-document
   - **FormattingConsistencyValidator**: Ensure consistent formatting styles
   - **ContentAccuracyValidator**: Compare against source for accuracy

4. **Advanced Features**
   - Document structure analysis
   - Table of contents generation
   - Cross-reference resolution
   - Multi-document processing

5. **Output Formats**
   - HTML export
   - LaTeX export
   - JSON structured data

## Debugging Tips
1. Enable debug logging: `--log-level DEBUG`
2. Disable progress logging: `--no-progress`
3. Check cache directory for rendered images
4. Monitor queue statistics in logs
5. Use smaller PDFs for testing

## Common Issues and Solutions

### API Key Issues
- Ensure `OPENAI_API_KEY` is set in environment
- Check API key permissions and quotas
- Verify endpoint URL if using custom provider

### Memory Issues
- Reduce page worker count
- Lower rendering resolution
- Increase queue size limits
- Process smaller documents

### Performance Issues
- Increase page worker count
- Use faster LLM model (gpt-4o-mini)
- Enable caching
- Reduce rendering resolution

## Testing Strategy
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Parser and pipeline integration
3. **End-to-End Tests**: Full document conversion
4. **Performance Tests**: Throughput and latency
5. **Error Scenario Tests**: Failure recovery

## Dependencies to Note
- **PyMuPDF**: Requires system libraries for PDF rendering
- **OpenAI SDK**: Handles API communication and retries
- **Pydantic**: Provides runtime type checking
- **asyncio**: Core to the concurrent processing model

## Code Style and Conventions
- Type hints for all functions
- Docstrings in Google style
- Black formatting (100 char line length)
- Comprehensive error handling
- Logging at appropriate levels

## Deployment Considerations
1. Docker containerization recommended
2. Environment-based configuration
3. Health check endpoints for monitoring
4. Metrics collection for observability
5. Secure API key management

## Maintenance Tasks
- Regular dependency updates
- Cache cleanup automation
- Log rotation configuration
- Performance monitoring
- API usage tracking
- We should always update the HLD and LLD when making design changes in our application. These documents are located in the `docs/` directory.
- We should always update the README.md when adding/changing/removing features from our application. Also when we change how the application is used (i.e. new command lines, changed command lines, updated configuration, etc.)
- We need to use hatch when running our application, installing dependencies, etc. This is what the project was configured with/for.

## Version Management
**IMPORTANT**: Version numbers should NEVER be manually changed. Always use `bump-my-version` for version management:

```bash
# For bug fixes and small changes
hatch run bump-patch    # 0.1.0 → 0.1.1

# For new features
hatch run bump-minor    # 0.1.0 → 0.2.0

# For breaking changes
hatch run bump-major    # 0.1.0 → 1.0.0
```

This automatically:
- Updates the version in `src/pdf2markdown/__init__.py`
- Creates a git commit with the version change
- Creates a git tag (e.g., `v0.1.1`)
- Triggers the GitHub Actions release workflow when pushed

## GitHub Actions Workflows
The project includes CI/CD workflows:

- **CI Workflow** (`.github/workflows/ci.yml`): Runs on push/PR to main/develop
  - Tests on Python 3.10, 3.11, 3.12, 3.13
  - Linting, formatting, and type checking
  - Test coverage reporting
  - Package building and validation

- **Release Workflow** (`.github/workflows/release.yml`): Runs on version tags
  - Builds and validates the package
  - Creates GitHub releases with auto-generated notes
  - Publishes to PyPI automatically

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.