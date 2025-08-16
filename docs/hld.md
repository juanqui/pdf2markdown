# High-level Design Document - PDF-to-Markdown Converter

## Summary

`pdf-to-markdown` is a Python application that leverages LLMs to accurately convert technical PDF documents, such as semiconductor datasheets, to well structured Markdown documents.

## Requirements

* Modular architecture that supports multiple `DocumentParser` implementations.
    * A Document Parser is an implementation that converts a complete PDF document into multiple `Page` objects which are generally just image renders of the page.
    * First version should implement a `SimpleDocumentParser` that uses the `PyMuPDF` to render each page into a PNG. This implementation should accept parameters such as the resolution to render pages to. Each `Page` resource should contain not only the path to the rendered image, but also any metadata available for that page as well. Images should be rendered to a temporary/cache location due to their size.
* Modular architecture that supports multiple `PageParser` implementations.
    * A Page Parser is an implementation that converts a `Page` resource, specifically the image render, into markdown.
    * First version should implement a parser called `SimpleLLMPageParser`. This parser will accept an `LLMProvider` instance which handles the actual LLM communication. The `Page` resource needs to support the markdown content. This implementation should use a Jinja2 template to define the prompt that will be used to invoke the LLM to perform the conversion to Markdown.
        * For this first version, let's use `gpt-4o-mini` as the default model. The prompt template has been simplified and emphasizes outputting ONLY the markdown content from the PDF.
* Modular architecture that supports multiple `LLMProvider` implementations.
    * An LLM Provider is an abstraction that handles communication with Large Language Models.
    * First version should implement an `OpenAILLMProvider` that supports any OpenAI-compatible API endpoint.
    * The provider interface should support methods like `invoke_with_image(prompt, image_path)` to process images with text prompts.
    * Future implementations could include local providers (using transformers/HuggingFace), Ollama, Anthropic, etc.
* Implements a robust pipeline-based approach to processing a PDF. It should use multiple queues to support N number of workers for each phase of the work. For example, we might want to have 5 workers converting PDFs to image renders and 10 workers converting each page to markdown.
    * NOTE: A `PageParser` can only parse one document at a time. We can't parallelize the process of generating page renders.
* Leverages `tqdm` to render progress.
* Implements `MarkdownValidator` using PyMarkdown (pymarkdownlnt) for validation.
    * Validates generated markdown for syntax correctness.
    * Can optionally attempt to correct issues by re-prompting the LLM with validation errors.
    * Configurable rules with sensible defaults for LLM-generated content (ignores overly strict rules like MD013 line length and MD047 trailing newline).


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