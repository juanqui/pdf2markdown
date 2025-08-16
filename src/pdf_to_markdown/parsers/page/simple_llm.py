"""Simple LLM-based page parser using LLM providers."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template

from pdf_to_markdown.core import (
    Page,
    PageParser,
    PageParsingError,
    ProcessingStatus,
)
from pdf_to_markdown.llm_providers import LLMProvider
from pdf_to_markdown.validators import MarkdownValidator

logger = logging.getLogger(__name__)


class SimpleLLMPageParser(PageParser):
    """Simple page parser that uses LLM providers to convert images to markdown."""

    def __init__(self, config: dict[str, Any], llm_provider: LLMProvider):
        """Initialize the parser with configuration.

        Args:
            config: Configuration dictionary with the following keys:
                - prompt_template (Path): Path to Jinja2 template
                - additional_instructions (str): Additional instructions for the LLM
                - validate_markdown (bool): Whether to validate generated markdown (default: True)
                - markdown_validator (dict): Configuration for markdown validator
            llm_provider: Pre-configured LLM provider instance (required)
        """
        super().__init__(config)

        # Initialize LLM provider
        if not llm_provider:
            raise ValueError("LLM provider is required for SimpleLLMPageParser")
        self.llm_provider = llm_provider

        # Initialize markdown validator if enabled
        self.validate_markdown = config.get("validate_markdown", True)
        self.markdown_validator = None

        if self.validate_markdown:
            validator_config = config.get("markdown_validator", {})
            # Set default configuration for validator
            validator_config.setdefault("attempt_correction", True)
            validator_config.setdefault("max_line_length", 1000)
            self.markdown_validator = MarkdownValidator(validator_config)
            logger.info("Markdown validation enabled")

        # Load prompt template
        template_path = config.get("prompt_template")
        if template_path is None:
            template_path = (
                Path(__file__).parent.parent.parent / "templates" / "prompts" / "ocr_extraction.j2"
            )
        elif isinstance(template_path, str):
            template_path = Path(template_path)
        self.prompt_template = self._load_template(template_path)

        logger.info(
            f"Initialized SimpleLLMPageParser with provider={self.llm_provider.__class__.__name__}"
        )

    def _load_template(self, template_path: Path) -> Template:
        """Load Jinja2 template from file.

        Args:
            template_path: Path to the template file

        Returns:
            Loaded Jinja2 template
        """
        if not template_path.exists():
            # Try to load from package templates directory
            env = Environment(
                loader=FileSystemLoader(
                    Path(__file__).parent.parent.parent / "templates" / "prompts"
                )
            )
            return env.get_template("ocr_extraction.j2")
        else:
            with open(template_path) as f:
                return Template(f.read())

    async def parse(self, page: Page) -> Page:
        """Convert a page image to markdown.

        Args:
            page: Page object with image path

        Returns:
            Page object with markdown content

        Raises:
            PageParsingError: If there's an error parsing the page
        """
        logger.info(f"Parsing page {page.page_number} with LLM")

        if not page.image_path or not page.image_path.exists():
            raise PageParsingError(f"Image not found for page {page.page_number}")

        try:
            # Update page status
            page.status = ProcessingStatus.PROCESSING

            # Render prompt template
            prompt = self.prompt_template.render(
                page_number=page.page_number,
                total_pages=page.metadata.total_pages if page.metadata else None,
                additional_instructions=self.config.get("additional_instructions"),
            )

            # Call LLM provider to extract text
            response = await self.llm_provider.invoke_with_image(prompt, page.image_path)
            markdown_content = response.content

            # Validate and potentially correct the markdown if enabled
            if self.validate_markdown and self.markdown_validator:
                logger.debug(f"Validating markdown for page {page.page_number}")

                validation_result = await self.markdown_validator.validate_and_correct(
                    markdown_content, page, self.llm_provider, self.prompt_template
                )

                if not validation_result.is_valid:
                    issues_summary = validation_result.get_issues_summary()
                    logger.warning(f"Page {page.page_number} validation issues:\n{issues_summary}")

                    # Use corrected markdown if available and better
                    if validation_result.corrected_markdown:
                        # Check if correction actually improved things
                        corrected_validation = self.markdown_validator.validate(
                            validation_result.corrected_markdown
                        )

                        if corrected_validation.is_valid or (
                            len(corrected_validation.issues) < len(validation_result.issues)
                        ):
                            logger.info(
                                f"Using corrected markdown for page {page.page_number} "
                                f"(issues: {len(validation_result.issues)} -> {len(corrected_validation.issues)})"
                            )
                            markdown_content = validation_result.corrected_markdown
                        else:
                            logger.warning(
                                f"Correction did not improve page {page.page_number}, using original"
                            )
                else:
                    logger.debug(f"Page {page.page_number} markdown validation passed")

            # Update page with markdown content
            page.markdown_content = markdown_content
            page.status = ProcessingStatus.COMPLETED

            # Update metadata
            if page.metadata:
                page.metadata.extraction_timestamp = datetime.now()

            logger.info(
                f"Successfully parsed page {page.page_number}, content length: {len(markdown_content) if markdown_content else 0}"
            )
            return page

        except Exception as e:
            logger.error(f"Error parsing page {page.page_number}: {e}")
            page.status = ProcessingStatus.FAILED
            page.error_message = str(e)
            raise PageParsingError(f"Failed to parse page {page.page_number}: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up LLM page parser resources")

        # Cleanup the provider
        await self.llm_provider.cleanup()
