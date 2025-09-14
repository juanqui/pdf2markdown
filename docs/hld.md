# High-level Design Document - pdf2markdown Converter

## Summary

`pdf2markdown` is a Python library and command-line application that leverages Large Language Models (LLMs) to accurately convert technical PDF documents, such as semiconductor datasheets, to well structured Markdown documents. The project provides both a high-level Python API for programmatic use and a CLI interface for command-line operations.

Key architectural features include:
- **Smart Caching System**: Deterministic caching of both rendered images and LLM-generated markdown for fast resume functionality
- **Multiple Input Support**: Process single files, multiple files, directories, or mixed inputs with flexible output options
- **Batch Processing**: Efficient processing of multiple documents with shared worker pools
- **Validation Pipeline**: Extensible validation system with markdown syntax validation and repetition detection

## Requirements

### Core Architecture Requirements
* **Dual Interface Design**: Provide both a Python library API and command-line interface
    * Library API for programmatic integration into other Python applications
    * CLI for standalone usage and automation scripts
    * Zero breaking changes to existing CLI functionality

### Library API Requirements
* **High-Level API**: Simple, intuitive interface for common use cases
    * `PDFConverter` class as main entry point
    * Synchronous and asynchronous methods
    * Streaming support for processing large documents
    * Batch processing capabilities
* **Configuration Management**: Flexible configuration system
    * `ConfigBuilder` pattern for programmatic configuration
    * Support for YAML files, dictionaries, and environment variables
    * Configuration hierarchy with proper override precedence
* **Error Handling**: Library-specific exception hierarchy
    * `PDFConversionError` base exception
    * Specific exceptions for configuration, parsing, LLM, and validation errors

### Parser Requirements
* Modular architecture that supports multiple `DocumentParser` implementations.
    * A Document Parser is an implementation that converts a complete PDF document into multiple `Page` objects which are generally just image renders of the page.
    * First version should implement a `SimpleDocumentParser` that uses the `PyMuPDF` to render each page into a PNG. This implementation should accept parameters such as the resolution to render pages to. Each `Page` resource should contain not only the path to the rendered image, but also any metadata available for that page as well. Images should be rendered to a temporary/cache location due to their size.
* Modular architecture that supports multiple `PageParser` implementations.
    * A Page Parser is an implementation that converts a `Page` resource, specifically the image render, into markdown.
    * First version should implement a parser called `SimpleLLMPageParser`. This parser will accept an `LLMProvider` instance which handles the actual LLM communication. The `Page` resource needs to support the markdown content. This implementation should use a Jinja2 template to define the prompt that will be used to invoke the LLM to perform the conversion to Markdown.
        * For this first version, let's use `gpt-4o-mini` as the default model. The prompt template has been simplified and emphasizes outputting ONLY the markdown content from the PDF.

### LLM Provider Requirements
* Modular architecture that supports multiple `LLMProvider` implementations.
    * An LLM Provider is an abstraction that handles communication with Large Language Models.
    * Implemented providers:
        * `OpenAILLMProvider`: Supports any OpenAI-compatible API endpoint.
        * `TransformersLLMProvider`: Runs models locally using HuggingFace Transformers.
    * The provider interface supports methods like `invoke_with_image(prompt, image_path)` to process images with text prompts.
    * Future implementations could include Ollama, Anthropic, etc.

### Pipeline Requirements
* Implements a robust pipeline-based approach to processing a PDF. It should use multiple queues to support N number of workers for each phase of the work. For example, we might want to have 5 workers converting PDFs to image renders and 10 workers converting each page to markdown.
    * NOTE: A `PageParser` can only parse one document at a time. We can't parallelize the process of generating page renders.
* Provides clear progress logging.

### Validation Requirements
* Implements comprehensive validation pipeline with multiple validators:
    * **MarkdownValidator**: Uses PyMarkdown (pymarkdownlnt) for syntax validation with configurable rules and LLM-based correction
    * **RepetitionValidator**: Detects and corrects various types of content repetition (consecutive lines, paragraph patterns, character repetition)
    * Extensible framework allowing easy addition of custom validators
    * Smart correction system that re-prompts LLM with detailed validation errors

### Caching and Resume Requirements
* **Smart Caching System**: Deterministic caching based on file content and configuration hashes
    * **Image Cache**: Caches rendered PDF page images with automatic invalidation when rendering settings change
    * **Markdown Cache**: Caches LLM-generated content with automatic invalidation when LLM configuration changes
    * **Resume Functionality**: Interrupted processing can be resumed using cached data, dramatically reducing time and costs
    * **Cache Management**: Automatic cleanup, statistics reporting, and manual cache clearing

### Multiple Input Processing Requirements
* **Flexible Input Handling**: Support for various input patterns
    * Single files, multiple files, directories, or mixed combinations
    * Automatic PDF discovery in directories with recursive search
    * Smart output path resolution with configurable separators and concatenation
* **Batch Processing**: Efficient processing of multiple documents
    * Shared worker pools across multiple documents
    * Global pipeline manager for optimal resource utilization
    * Progress tracking and error reporting for batch operations

## Reference - Simplified Prompt

The prompt template has been simplified to emphasize clarity and prevent additional text generation:

```markdown
**CRITICAL**: Output ONLY the markdown content from the document. Do not add any explanations, comments, or text that is not present in the original PDF.

Convert the document image to Markdown following these rules:
- Tables: Use Markdown pipe syntax
- Headers: Use # for sections  
- Formatting: **Bold**, *Italic*, `Code`
- Math: Use $LaTeX$ notation
- Preserve ALL numbers, units, and conditions exactly
- For diagrams/graphs: **[Type: Brief description]**
- Start directly with the document content
```