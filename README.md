# PDF to Markdown Converter

A Python application that leverages Large Language Models (LLMs) to accurately convert technical PDF documents into well-structured Markdown documents.

## Features

- 🚀 **High-Quality Conversion**: Uses state-of-the-art LLMs for accurate text extraction
- 📊 **Table Preservation**: Converts tables to HTML or Markdown format (configurable)
- 🔢 **Equation Support**: Preserves mathematical equations in LaTeX format
- 🖼️ **Image Handling**: Describes images and preserves captions
- ⚡ **Parallel Processing**: Processes multiple pages concurrently for speed
- 📈 **Progress Tracking**: Clear logging of processing status
- 🔧 **Configurable**: Extensive configuration options via YAML or CLI
- 🔄 **Retry Logic**: Automatic retry with exponential backoff for reliability
- ✅ **Validation Pipeline**: Extensible validation system with multiple validators
- 🔍 **Repetition Detection**: Automatically detects and corrects content repetition
- ✔️ **Markdown Validation**: Built-in syntax validation and correction using PyMarkdown
- 🎯 **Pure Output**: Generates only document content without additional commentary
- 🧹 **Smart Cleaning**: Automatically removes markdown code fences that LLMs sometimes add
- 📄 **Configurable Page Separators**: Customize how pages are separated in the output
- 📁 **Batch Processing**: Process multiple files and directories with optional output organization
- 🔄 **Flexible I/O**: Optional output paths with smart defaults (same name, .md extension)
- 💾 **Smart Caching**: Automatic caching of rendered images and LLM outputs for fast re-processing
- ⏯️ **Resume Support**: Resume interrupted processing using cached data to save time and costs

## Installation

### From PyPI (Coming Soon)

```bash
pip install pdf2markdown
```

### Using Hatch (Development)

```bash
# Install Hatch
pipx install hatch

# Clone the repository
git clone https://github.com/juanqui/pdf2markdown.git
cd pdf2markdown

# Install dependencies
hatch env create

# Activate environment
hatch shell
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/juanqui/pdf2markdown.git
cd pdf2markdown

# Install the package
pip install -e .

# Optional: Install with transformers support for local models
pip install -e ".[transformers]"
```

## Quick Start

1. **Set up configuration:**
```bash
# Copy the sample configuration file
cp config/default.sample.yaml config/default.yaml

# Edit the configuration file with your settings
# At minimum, update the llm_provider section with your API details
nano config/default.yaml  # or use your preferred editor
```

2. **Set your API key (recommended via environment variable):**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. **Convert a PDF:**
```bash
# Output defaults to input filename with .md extension
pdf2markdown input.pdf

# Or specify a custom output file
pdf2markdown input.pdf -o output.md
```

## Library Usage

`pdf2markdown` can be used as a Python library in your own applications. This is useful for integrating PDF conversion into larger systems, web applications, or data processing pipelines.

### Simple Usage

```python
from pdf2markdown import PDFConverter

# Create converter with default settings
converter = PDFConverter()

# Convert a PDF to markdown
markdown_text = converter.convert_sync("document.pdf")
print(markdown_text)

# Save to a file
markdown_text = converter.convert_sync("document.pdf", "output.md")
```

### Configuration Options

```python
from pdf2markdown import PDFConverter, ConfigBuilder

# Build configuration programmatically
config = ConfigBuilder() \
    .with_openai(api_key="your-api-key", model="gpt-4o") \
    .with_resolution(400) \
    .with_page_workers(20) \
    .with_cache_dir("/tmp/my_cache") \
    .build()

converter = PDFConverter(config=config)
markdown = converter.convert_sync("document.pdf")
```

### Table Format Configuration

```python
from pdf2markdown import ConfigBuilder, PDFConverter

# Configure for HTML tables (better for complex layouts)
config = ConfigBuilder() \
    .with_openai(api_key="your-api-key") \
    .build()

# Set table format in the configuration
config['page_parser']['table_format'] = 'html'  # Default

converter = PDFConverter(config=config)

# Or configure for Markdown tables (simpler format)
config['page_parser']['table_format'] = 'markdown'
```

### Using Different LLM Providers

```python
from pdf2markdown import ConfigBuilder, PDFConverter

# OpenAI (or compatible endpoints)
config = ConfigBuilder() \
    .with_openai(
        api_key="your-key",
        model="gpt-4o-mini",
        endpoint="https://api.openai.com/v1"  # or your custom endpoint
    ) \
    .build()

# Local models with Transformers
config = ConfigBuilder() \
    .with_transformers(
        model_name="microsoft/Phi-3.5-vision-instruct",
        device="cuda",  # or "cpu"
        torch_dtype="float16"
    ) \
    .build()

converter = PDFConverter(config=config)
```

### Async Usage

```python
import asyncio
from pdf2markdown import PDFConverter

async def convert_pdf():
    converter = PDFConverter()
    
    # Async conversion
    markdown = await converter.convert("document.pdf")
    
    # With progress callback
    async def progress(current, total, message):
        print(f"Progress: {current}/{total} - {message}")
    
    markdown = await converter.convert(
        "document.pdf",
        progress_callback=progress
    )
    
    return markdown

# Run async function
markdown = asyncio.run(convert_pdf())
```

### Streaming Pages

Process large documents page by page as they complete:

```python
import asyncio
from pdf2markdown import PDFConverter

async def stream_conversion():
    converter = PDFConverter()
    
    async for page in converter.stream_pages("large_document.pdf"):
        print(f"Page {page.page_number}: {len(page.content)} characters")
        # Process each page as it completes
        # e.g., save to database, send to queue, etc.

asyncio.run(stream_conversion())
```

### Batch Processing

Convert multiple PDFs efficiently:

```python
import asyncio
from pdf2markdown import PDFConverter

async def batch_convert():
    converter = PDFConverter()
    
    pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    results = await converter.process_batch(
        pdf_files,
        output_dir="./output"
    )
    
    for result in results:
        if result.status == ConversionStatus.COMPLETED:
            print(f"✓ {result.source_path}")
        else:
            print(f"✗ {result.source_path}: {result.error_message}")

asyncio.run(batch_convert())
```

### Loading Configuration from Files

```python
from pdf2markdown import PDFConverter, Config

# From YAML file
config = Config.from_yaml("config.yaml")
converter = PDFConverter(config=config)

# From dictionary
config_dict = {
    "llm_provider": {
        "provider_type": "openai",
        "api_key": "your-key",
        "model": "gpt-4o-mini"
    },
    "pipeline": {
        "page_workers": 15
    }
}
converter = PDFConverter(config=config_dict)
```

### Error Handling

```python
from pdf2markdown import (
    PDFConverter,
    PDFConversionError,
    ConfigurationError,
    ParsingError
)

try:
    converter = PDFConverter()
    markdown = converter.convert_sync("document.pdf")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except ParsingError as e:
    print(f"Failed to parse PDF: {e}")
    if e.page_number:
        print(f"Error on page {e.page_number}")
except PDFConversionError as e:
    print(f"Conversion failed: {e}")
```

### Context Manager

Properly clean up resources using context managers:

```python
import asyncio
from pdf2markdown import PDFConverter

async def convert_with_cleanup():
    async with PDFConverter() as converter:
        markdown = await converter.convert("document.pdf")
        # Converter automatically cleaned up after this block
    return markdown

markdown = asyncio.run(convert_with_cleanup())
```

### Integration Examples

#### Flask Web Application

```python
from flask import Flask, request, jsonify
from pdf2markdown import PDFConverter

app = Flask(__name__)
converter = PDFConverter()

@app.route('/convert', methods=['POST'])
def convert_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    file.save('/tmp/upload.pdf')
    
    try:
        markdown = converter.convert_sync('/tmp/upload.pdf')
        return jsonify({'markdown': markdown})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

#### Celery Task Queue

```python
from celery import Celery
from pdf2markdown import PDFConverter

app = Celery('tasks', broker='redis://localhost:6379')
converter = PDFConverter()

@app.task
def convert_pdf_task(pdf_path):
    """Background task to convert PDF"""
    return converter.convert_sync(pdf_path)
```

#### Document Processing Pipeline

```python
from pdf2markdown import PDFConverter, ConfigBuilder
import sqlite3

# Configure for high-quality conversion
config = ConfigBuilder() \
    .with_openai(api_key="your-key", model="gpt-4o") \
    .with_resolution(400) \
    .with_validators(['markdown', 'repetition']) \
    .build()

converter = PDFConverter(config=config)

def process_document(pdf_path, doc_id):
    """Process document and store in database"""
    # Convert PDF
    markdown = converter.convert_sync(pdf_path)
    
    # Store in database
    conn = sqlite3.connect('documents.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (id, content) VALUES (?, ?)",
        (doc_id, markdown)
    )
    conn.commit()
    conn.close()
    
    return doc_id
```

## CLI Usage

### Basic Usage

```bash
# Convert single file (output defaults to same name with .md extension)
pdf2markdown document.pdf                    # Creates document.md

# Specify output file
pdf2markdown document.pdf -o converted.md

# Use a specific model
pdf2markdown document.pdf --model gpt-4o

# Adjust rendering resolution
pdf2markdown document.pdf --resolution 400

# Limit maximum image dimension
pdf2markdown document.pdf --max-dimension 2048
```

### Multiple Files and Directories

```bash
# Convert multiple files (each creates its own .md file)
pdf2markdown file1.pdf file2.pdf file3.pdf

# Convert all PDFs in a directory
pdf2markdown /path/to/pdfs/

# Convert multiple files to a specific output directory
pdf2markdown *.pdf -o /output/directory/

# Convert directory to another directory
pdf2markdown /input/docs/ -o /output/docs/

# Mix files and directories
pdf2markdown doc1.pdf /more/docs/ doc2.pdf

# Concatenate multiple files into single output
pdf2markdown file1.pdf file2.pdf -o combined.md

# Resume interrupted processing
pdf2markdown document.pdf --resume

# Clear cache and force fresh processing
pdf2markdown document.pdf --clear-cache

# View cache statistics
pdf2markdown --cache-stats
```

### Caching and Resume

The application includes a sophisticated caching system that dramatically improves performance for repeated processing:

```bash
# Automatic caching (enabled by default)
pdf2markdown document.pdf                     # Caches images and markdown

# Resume interrupted processing
pdf2markdown document.pdf --resume            # Uses cached data where available

# Force fresh processing
pdf2markdown document.pdf --clear-cache       # Ignores all cached data

# Monitor cache usage
pdf2markdown --cache-stats                    # Shows cache size and contents

# Process with specific cache settings
pdf2markdown document.pdf --cache-dir /my/cache
```

**How caching works:**
- **Image Cache**: PDF pages rendered to images are cached based on file content and rendering settings (resolution, max_dimension)
- **Markdown Cache**: LLM-generated content is cached based on LLM configuration (model, temperature, prompts, validation settings)
- **Smart Invalidation**: Caches are automatically invalidated when relevant configurations change
- **Deterministic IDs**: Documents get consistent cache IDs based on file content and configuration
- **Cost Savings**: Avoid re-processing expensive LLM calls for unchanged content

### Advanced Usage

```bash
# Use custom configuration file
pdf2markdown document.pdf --config my-config.yaml

# Parallel processing with more workers
pdf2markdown document.pdf --page-workers 20

# Disable progress logging for automation
pdf2markdown document.pdf --no-progress

# Save configuration for reuse
pdf2markdown document.pdf --save-config my-settings.yaml

# Specify table format (html or markdown)
pdf2markdown document.pdf --table-format html  # For complex tables
pdf2markdown document.pdf --table-format markdown  # For simple tables
```

### Configuration

#### Initial Setup

The application uses a YAML configuration file to manage settings. To get started:

1. **Copy the sample configuration:**
   ```bash
   cp config/default.sample.yaml config/default.yaml
   ```

2. **Review and edit the configuration:**
   The sample file (`config/default.sample.yaml`) is heavily documented with explanations for every setting. Key sections to configure:
   - `llm_provider`: Your LLM API settings (endpoint, API key, model)
   - `document_parser`: PDF rendering settings
   - `pipeline`: Worker and processing settings

3. **Set sensitive values via environment variables:**
   Instead of hardcoding API keys in the config file, use environment variables:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Then reference it in your config as: `${OPENAI_API_KEY}`

#### Configuration File Structure

Here's an overview of the configuration structure:

```yaml
# Cache Configuration (optional, but recommended)
cache:
  enabled: true                          # Enable caching system
  base_dir: /tmp/pdf2markdown/cache      # Cache directory
  max_size_gb: 10                        # Maximum cache size
  cleanup_after_days: 7                  # Auto-cleanup old caches
  resume_by_default: false               # Resume by default

# LLM Provider Configuration (required)
llm_provider:
  provider_type: openai  # Provider type (currently supports "openai")
  endpoint: https://api.openai.com/v1  # API endpoint URL
  api_key: ${OPENAI_API_KEY}  # Can reference environment variables
  model: gpt-4o-mini  # Model to use
  max_tokens: 4096  # Maximum tokens in response
  temperature: 0.1  # Generation temperature (0.0-2.0)
  timeout: 60  # Request timeout in seconds
  
  # Penalty parameters to reduce repetition (all optional)
  presence_penalty: 0.0  # Penalize tokens based on presence (-2.0 to 2.0)
  frequency_penalty: 0.0  # Penalize tokens based on frequency (-2.0 to 2.0)
  repetition_penalty: null  # Alternative repetition penalty (0.0 to 2.0, some providers only)

# Document Parser Configuration
document_parser:
  type: simple  # Parser type
  resolution: 300  # DPI for rendering PDF pages to images
  max_dimension: null  # Optional: maximum pixels for longest side of rendered image
  cache_dir: /tmp/pdf2markdown/cache  # Cache directory for rendered images
  max_page_size: 50000000  # Maximum page size in bytes (50MB)
  timeout: 30  # Timeout for rendering operations
  use_cache: true  # Enable caching of rendered images (recommended)

# Page Parser Configuration
page_parser:
  type: simple_llm  # Parser type
  prompt_template: null  # Optional custom prompt template path
  additional_instructions: null  # Optional additional LLM instructions
  
  # Table format configuration
  table_format: html  # 'html' for complex layouts, 'markdown' for simple tables
  
  # Content validation pipeline configuration
  validate_content: true  # Enable content validation
  use_cache: true  # Enable caching of LLM-generated markdown (recommended)
  
  validation:
    # List of validators to run (in order)
    validators: ["markdown", "repetition"]
    
    # Maximum number of correction attempts
    max_correction_attempts: 2
    
    # Markdown validator - checks syntax and formatting
    markdown:
      enabled: true  # Enable this validator
      attempt_correction: true  # Try to fix issues by re-prompting LLM
      strict_mode: false  # Use relaxed mode for LLM-generated content
      max_line_length: 1000  # Max line length (MD013 rule)
      disabled_rules: []  # Additional rules to disable
      enabled_rules: []  # Specific rules to enable
      # Note: Common overly-strict rules are disabled by default including:
      # MD033 (Inline HTML) - common in technical documents and tables
      # MD026 (Trailing punctuation in headings) - common in PDF headings
      # MD042 (No empty links) - LLMs may generate placeholder links during extraction
      # MD036 (Emphasis used instead of heading) - LLMs may use bold/italic for headings
      # MD041, MD022, MD031, MD032, MD025, MD024, MD013, MD047, MD040
    
    # Repetition validator - detects and corrects unwanted repetition
    repetition:
      enabled: true  # Enable this validator
      attempt_correction: true  # Try to fix repetition issues
      consecutive_threshold: 3  # Flag 3+ consecutive duplicate lines
      window_size: 10  # Check within 10-line windows
      window_threshold: 3  # Flag 3+ occurrences in window
      check_exact_lines: true  # Check for exact duplicates
      check_normalized_lines: true  # Check ignoring whitespace/punctuation
      check_paragraphs: true  # Check for duplicate paragraphs
      check_patterns: true  # Detect repetitive patterns
      min_pattern_length: 20  # Minimum chars for pattern detection
      pattern_similarity_threshold: 0.9  # Similarity threshold (0-1)
      min_line_length: 5  # Minimum line length to check

# Pipeline Configuration
pipeline:
  document_workers: 1  # Must be 1 for sequential document processing
  page_workers: 10  # Number of parallel page processing workers
  queues:
    document_queue_size: 100
    page_queue_size: 1000
    output_queue_size: 500
  enable_progress: true  # Show progress bars
  log_level: INFO  # Logging level

# Output Configuration
output_dir: ./output  # Default output directory
temp_dir: /tmp/pdf2markdown  # Temporary file directory
page_separator: "\n\n--[PAGE: {page_number}]--\n\n"  # Separator between pages
```

#### Configuration Hierarchy

Configuration values are loaded in the following order (later values override earlier ones):

1. Default values in code
2. Configuration file (`config/default.yaml` or file specified via `--config`)
3. Environment variables
4. Command-line arguments

**Note:** The application looks for `config/default.yaml` in the current working directory by default. You can specify a different configuration file using the `--config` option:
```bash
pdf2markdown input.pdf --config /path/to/my-config.yaml
```

#### LLM Provider Configuration

The `llm_provider` section is shared across all components that need LLM access. This centralized configuration makes it easy to:

- Switch between different LLM providers
- Use the same provider settings for multiple components
- Override settings globally via environment variables or CLI

**Supported Providers:**
- `openai`: Any OpenAI-compatible API (OpenAI, Azure OpenAI, local servers with OpenAI-compatible endpoints)
- `transformers`: Local models using HuggingFace Transformers (requires optional dependencies)

**Future Providers (planned):**
- `ollama`: Local models via Ollama
- `anthropic`: Anthropic Claude API
- `google`: Google Gemini API

##### Penalty Parameters for Reducing Repetition

To avoid repetitive text in the generated markdown, you can configure penalty parameters:

- **presence_penalty** (-2.0 to 2.0): Penalizes tokens that have already appeared in the text. Positive values discourage repetition.
- **frequency_penalty** (-2.0 to 2.0): Penalizes tokens based on their frequency in the text so far. Positive values reduce repetition of common phrases.
- **repetition_penalty** (0.0 to 2.0): Alternative parameter used by some providers (e.g., local models). Values > 1.0 reduce repetition.

**Recommended settings for reducing repetition:**
```yaml
llm_provider:
  presence_penalty: 0.5
  frequency_penalty: 0.5
  # OR for providers that use repetition_penalty:
  repetition_penalty: 1.15
```

#### Custom OpenAI-Compatible Endpoints

To use a custom OpenAI-compatible endpoint (e.g., local LLM server, vLLM, etc.):

```yaml
llm_provider:
  provider_type: openai
  endpoint: http://localhost:8080/v1  # Your custom endpoint
  api_key: dummy-key  # Some endpoints require a placeholder
  model: your-model-name
  max_tokens: 8192
  temperature: 0.7
  timeout: 120
```

#### Using Local Models with Transformers

The Transformers provider allows you to run models locally using HuggingFace Transformers. This is useful for:
- Running without API costs
- Processing sensitive documents locally
- Using specialized models not available via APIs
- Running on systems with GPU acceleration

**Installation:**
```bash
# Install with transformers support
pip install -e ".[transformers]"
```

**Configuration Example:**
```yaml
llm_provider:
  provider_type: transformers
  model_name: "openbmb/MiniCPM-V-4"  # HuggingFace model ID
  device: "auto"  # or "cuda", "cpu", "cuda:0", etc.
  torch_dtype: "bfloat16"  # or "float16", "float32", "auto"
  max_tokens: 4096
  temperature: 0.1
  do_sample: false
  
  # Optional: Use 4-bit quantization to save memory
  load_in_4bit: true
  
  # Optional: For models with .chat() method
  use_chat_method: true
```

**Supported Models (examples):**
- **MiniCPM-V series**: `openbmb/MiniCPM-V-4`, `openbmb/MiniCPM-V-2_6`
- **Nanonets OCR**: `nanonets/Nanonets-OCR-s`
- **Other vision models**: Any model supporting image-text-to-text generation

**Performance Tips:**
- Use `load_in_4bit: true` or `load_in_8bit: true` to reduce memory usage
- Set `page_workers: 1` in pipeline config for local models (they use more memory)
- Use `device_map: "auto"` for multi-GPU systems
- Consider using `attn_implementation: "flash_attention_2"` for faster inference (if supported)

See `config/transformers_example.yaml` for a complete configuration example.

## Environment Variables

### LLM Provider Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_API_ENDPOINT`: Custom API endpoint URL (optional)
- `OPENAI_MODEL`: Model to use (default: gpt-4o-mini)

### Application Variables
- `PDF2MARKDOWN_CACHE_DIR`: Cache directory for rendered images
- `PDF2MARKDOWN_OUTPUT_DIR`: Default output directory
- `PDF2MARKDOWN_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `PDF2MARKDOWN_TEMP_DIR`: Temporary file directory

## How It Works

1. **Document Parsing**: PDF pages are rendered as high-resolution images using PyMuPDF
2. **LLM Provider**: The configured LLM provider handles communication with the AI model
3. **Image Processing**: Each page image is sent to the LLM with vision capabilities
4. **Content Extraction**: The LLM extracts and formats content as Markdown
5. **Validation Pipeline**: Content passes through multiple validators:
   - **Markdown Validator**: Checks syntax and formatting
   - **Repetition Validator**: Detects unwanted repetition patterns
6. **Correction** (optional): If issues are found, the LLM is re-prompted with specific instructions to fix them
7. **Assembly**: Processed pages are combined into a single Markdown document

### Architecture Overview

The application uses a modular architecture with these key components:

- **LLM Provider**: Abstraction layer for different LLM services (OpenAI, local models, etc.)
- **Document Parser**: Converts PDF pages to images
- **Page Parser**: Converts images to Markdown using LLM
- **Validation Pipeline**: Extensible system with multiple validators:
  - **Markdown Validator**: Validates and corrects syntax issues
  - **Repetition Validator**: Detects and corrects unwanted repetition
  - Easily extensible for additional validators
- **Pipeline**: Orchestrates the conversion process with parallel workers
- **Queue System**: Manages work distribution across workers

## Output Format

The converter preserves:
- **Headers**: Converted to appropriate Markdown heading levels
- **Tables**: Rendered as HTML tables or Markdown tables (configurable)
- **Lists**: Both ordered and unordered lists
- **Equations**: LaTeX format for mathematical expressions ($inline$ and $$display$$)
- **Images**: Descriptions or captions preserved
- **Formatting**: Bold, italic, code, and other text styling
- **Technical Elements**: Pin diagrams, electrical characteristics, timing specifications
- **Special Notations**: Notes, warnings, footnotes, and cross-references

### Table Format Options

The converter supports two table formats, configurable via the `table_format` setting:

#### HTML Tables (Default)
HTML tables are recommended for complex layouts with:
- Merged cells (colspan/rowspan)
- Nested tables
- Complex alignments
- Multi-line cell content

Example configuration:
```yaml
page_parser:
  table_format: html  # Default setting
```

Output example:
```html
<table>
  <thead>
    <tr>
      <th rowspan="2">Parameter</th>
      <th colspan="3">Conditions</th>
      <th>Unit</th>
    </tr>
    <tr>
      <th>Min</th>
      <th>Typ</th>
      <th>Max</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Operating Voltage</td>
      <td>1.7</td>
      <td>3.3</td>
      <td>3.6</td>
      <td>V</td>
    </tr>
  </tbody>
</table>
```

#### Markdown Tables
Markdown tables are simpler and more readable in plain text, best for:
- Simple tabular data
- Tables without merged cells
- Basic alignment needs

Example configuration:
```yaml
page_parser:
  table_format: markdown
```

Output example:
```markdown
| Parameter | Min | Typ | Max | Unit |
|-----------|----:|----:|----:|------|
| Voltage   | 1.7 | 3.3 | 3.6 | V    |
| Current   | 0.1 | 0.5 | 1.0 | mA   |
```

### Output Quality

The converter ensures high-quality output through multiple mechanisms:

#### Output Purity
- Outputs **ONLY** the content from the PDF document
- No explanatory text or comments
- No "Here is the content" preambles
- No additional formatting suggestions
- Automatically removes markdown code fences if LLM wraps output
- Just clean, accurate Markdown representing the original document

#### Validation Pipeline
- **Syntax Validation**: Ensures proper markdown formatting
- **Repetition Detection**: Identifies and corrects various types of repetition:
  - Consecutive duplicate lines
  - Near-duplicates within sliding windows
  - Duplicate paragraphs
  - Repetitive patterns
- **Extensible System**: Easy to add custom validators for specific needs

### Page Separation

Pages are separated using a configurable separator (default: `--[PAGE: N]--`). You can customize this in the configuration:
```yaml
# Examples of page separators:
page_separator: "\n---\n"                           # Simple horizontal rule
page_separator: "\n\n<!-- Page {page_number} -->\n\n"  # HTML comment (invisible)
page_separator: "\n\n# Page {page_number}\n\n"         # Markdown heading
page_separator: "\n\n--[PAGE: {page_number}]--\n\n"    # Default format
```

## Performance

- Processes pages in parallel (default: 10 workers)
- Automatic caching of rendered images
- Typical processing: 5-10 seconds per page

## Requirements

- Python 3.10+
- OpenAI API key (or compatible endpoint)
- System dependencies for PyMuPDF

## Configuration Examples

### Using Azure OpenAI

```yaml
llm_provider:
  provider_type: openai
  endpoint: https://your-resource.openai.azure.com/
  api_key: ${AZURE_OPENAI_KEY}
  model: gpt-4-vision
  max_tokens: 4096
```

### Using Local LLM Server

```yaml
llm_provider:
  provider_type: openai
  endpoint: http://localhost:11434/v1  # Ollama with OpenAI compatibility
  api_key: not-needed
  model: llava:13b
  max_tokens: 8192
  timeout: 300  # Longer timeout for local models
  # Many local servers use repetition_penalty instead
  repetition_penalty: 1.15
```

### High-Performance Configuration

```yaml
llm_provider:
  provider_type: openai
  endpoint: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  max_tokens: 8192
  temperature: 0.1
  # Reduce repetition for better quality output
  presence_penalty: 0.5
  frequency_penalty: 0.5

pipeline:
  page_workers: 20  # More parallel workers for faster processing

document_parser:
  resolution: 400  # Higher quality images
  # max_dimension: 3000  # Optional: limit max dimension if memory is a concern
```

## Troubleshooting

### API Key Issues
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Set in .env file
echo "OPENAI_API_KEY=your-key" > .env

# Check configuration
pdf2markdown document.pdf --save-config debug-config.yaml
# Then inspect debug-config.yaml
```

### Memory Issues
```bash
# Reduce worker count
pdf2markdown large.pdf --page-workers 5

# Lower resolution
pdf2markdown large.pdf --resolution 200

# Limit maximum image dimension (pixels)
pdf2markdown large.pdf --max-dimension 1536
```

### Debugging
```bash
# Enable debug logging
pdf2markdown document.pdf --log-level DEBUG

# Check cache directory
ls /tmp/pdf2markdown/cache/
```

## Development

### Running Tests
```bash
hatch run test
```

### Code Formatting
```bash
hatch run format
```

### Type Checking
```bash
hatch run typecheck
```

## License

MIT License - see LICENSE file for details
