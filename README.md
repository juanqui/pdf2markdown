# PDF to Markdown Converter

A Python application that leverages Large Language Models (LLMs) to accurately convert technical PDF documents into well-structured Markdown documents.

## Features

- 🚀 **High-Quality Conversion**: Uses state-of-the-art LLMs for accurate text extraction
- 📊 **Table Preservation**: Converts tables to clean Markdown format
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

## Installation

### Using Hatch (Recommended)

```bash
# Install Hatch
pipx install hatch

# Clone the repository
git clone https://github.com/yourusername/pdf-to-markdown.git
cd pdf-to-markdown

# Install dependencies
hatch env create

# Activate environment
hatch shell
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/pdf-to-markdown.git
cd pdf-to-markdown

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
pdf-to-markdown input.pdf -o output.md
```

## Usage

### Basic Usage

```bash
# Convert with default settings
pdf-to-markdown document.pdf

# Specify output file
pdf-to-markdown document.pdf -o converted.md

# Use a specific model
pdf-to-markdown document.pdf --model gpt-4o

# Adjust rendering resolution
pdf-to-markdown document.pdf --resolution 400
```

### Advanced Usage

```bash
# Use custom configuration file
pdf-to-markdown document.pdf --config my-config.yaml

# Parallel processing with more workers
pdf-to-markdown document.pdf --page-workers 20

# Disable progress logging for automation
pdf-to-markdown document.pdf --no-progress

# Save configuration for reuse
pdf-to-markdown document.pdf --save-config my-settings.yaml
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
  cache_dir: /tmp/pdf_to_markdown/cache  # Cache directory for rendered images
  max_page_size: 50000000  # Maximum page size in bytes (50MB)
  timeout: 30  # Timeout for rendering operations

# Page Parser Configuration
page_parser:
  type: simple_llm  # Parser type
  prompt_template: null  # Optional custom prompt template path
  additional_instructions: null  # Optional additional LLM instructions
  
  # Content validation pipeline configuration
  validate_content: true  # Enable content validation
  
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
      # Note: Common overly-strict rules are disabled by default
    
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
temp_dir: /tmp/pdf_to_markdown  # Temporary file directory
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
pdf-to-markdown input.pdf --config /path/to/my-config.yaml
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
- `PDF_TO_MARKDOWN_CACHE_DIR`: Cache directory for rendered images
- `PDF_TO_MARKDOWN_OUTPUT_DIR`: Default output directory
- `PDF_TO_MARKDOWN_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `PDF_TO_MARKDOWN_TEMP_DIR`: Temporary file directory

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
- **Tables**: Rendered as clean Markdown tables with pipe syntax
- **Lists**: Both ordered and unordered lists
- **Equations**: LaTeX format for mathematical expressions ($inline$ and $$display$$)
- **Images**: Descriptions or captions preserved
- **Formatting**: Bold, italic, code, and other text styling
- **Technical Elements**: Pin diagrams, electrical characteristics, timing specifications
- **Special Notations**: Notes, warnings, footnotes, and cross-references

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
```

## Troubleshooting

### API Key Issues
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Set in .env file
echo "OPENAI_API_KEY=your-key" > .env

# Check configuration
pdf-to-markdown document.pdf --save-config debug-config.yaml
# Then inspect debug-config.yaml
```

### Memory Issues
```bash
# Reduce worker count
pdf-to-markdown large.pdf --page-workers 5

# Lower resolution
pdf-to-markdown large.pdf --resolution 200
```

### Debugging
```bash
# Enable debug logging
pdf-to-markdown document.pdf --log-level DEBUG

# Check cache directory
ls /tmp/pdf_to_markdown/cache/
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

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

## Acknowledgments

- PyMuPDF for PDF rendering
- OpenAI for LLM capabilities
- The open-source community