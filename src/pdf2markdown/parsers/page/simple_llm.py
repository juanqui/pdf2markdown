"""Simple LLM-based page parser using LLM providers."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template

from pdf2markdown.core import (
    Page,
    PageParser,
    PageParsingError,
    ProcessingStatus,
)
from pdf2markdown.llm_providers import LLMProvider
from pdf2markdown.utils.cache_manager import CacheManager, ConfigHasher
from pdf2markdown.utils.markdown_cleaner import clean_llm_output
from pdf2markdown.utils.statistics import get_statistics_tracker
from pdf2markdown.validators import BaseValidator, create_validators

logger = logging.getLogger(__name__)


class SimpleLLMPageParser(PageParser):
    """Simple page parser that uses LLM providers to convert images to markdown."""

    def __init__(
        self,
        config: dict[str, Any],
        llm_provider: LLMProvider,
        full_config: dict[str, Any] | None = None,
    ):
        """Initialize the parser with configuration.

        Args:
            config: Configuration dictionary with the following keys:
                - prompt_template (Path): Path to Jinja2 template
                - additional_instructions (str): Additional instructions for the LLM
                - validate_markdown (bool): Whether to validate generated markdown (default: True)
                - markdown_validator (dict): Configuration for markdown validator
                - use_cache (bool): Whether to use caching (default: True)
            llm_provider: Pre-configured LLM provider instance (required)
            full_config: Full application configuration for cache hashing
        """
        super().__init__(config)

        # Initialize LLM provider
        if not llm_provider:
            raise ValueError("LLM provider is required for SimpleLLMPageParser")
        self.llm_provider = llm_provider

        # Store full config for cache hashing
        self.full_config = full_config or {"page_parser": config}
        self.use_cache = config.get("use_cache", True)

        # Initialize cache manager (will be set by coordinator)
        self.cache_manager = None

        # Initialize validators from configuration
        self.validators = self._initialize_validators(config)
        # Get max_correction_attempts from validation config
        validation_config = config.get("validation", {})
        self.max_correction_attempts = validation_config.get("max_correction_attempts", 2)

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
            f"Initialized SimpleLLMPageParser with provider={self.llm_provider.__class__.__name__}, "
            f"validators={[v.name for v in self.validators]}, use_cache={self.use_cache}"
        )

    def set_cache_manager(self, cache_manager: CacheManager) -> None:
        """Set the cache manager for this parser.

        Args:
            cache_manager: CacheManager instance to use
        """
        self.cache_manager = cache_manager

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

    def _initialize_validators(self, config: dict[str, Any]) -> list[BaseValidator]:
        """Initialize validators from configuration.

        Args:
            config: Configuration dictionary with optional 'validation' section

        Returns:
            List of configured validators
        """
        # Check if validation is enabled at all
        if not config.get("validate_content", True):
            logger.info("Content validation disabled")
            return []

        # Get validation configuration
        validation_config = config.get("validation", {})

        # If old-style configuration, convert it (only if the new config isn't present)
        # Check for actual values, not just presence of keys (which might be None)
        has_legacy_config = (
            config.get("validate_markdown") is not None
            or config.get("markdown_validator") is not None
        )
        has_new_config = bool(validation_config)

        if has_legacy_config and not has_new_config:
            logger.warning("Using legacy validation configuration format")
            # Create markdown validator config from old format
            markdown_config = config.get("markdown_validator", {})
            if markdown_config is None:
                markdown_config = {}
            markdown_config["enabled"] = config.get("validate_markdown", True)
            validation_config = {
                "validators": ["markdown"],
                "markdown": markdown_config,
            }

        # Default configuration if not specified
        if not validation_config:
            validation_config = {
                "validators": ["markdown", "repetition"],
                "markdown": {"enabled": True, "attempt_correction": True},
                "repetition": {"enabled": True, "attempt_correction": True},
            }

        # Create validators
        validators = create_validators(validation_config)
        return validators

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

        # Track conversion time
        stats = get_statistics_tracker()
        stats.start_page_conversion(page.page_number)

        if not page.image_path or not page.image_path.exists():
            raise PageParsingError(f"Image not found for page {page.page_number}")

        try:
            # Update page status
            page.status = ProcessingStatus.PROCESSING

            # Check if we can use cached markdown
            cached_content = None
            if self.use_cache and self.cache_manager:
                # Generate page config hash for cache validation
                page_config_hash = ConfigHasher.hash_page_config(self.full_config)
                logger.debug(f"Page config hash for page {page.page_number}: {page_config_hash}")

                # Get markdown cache for this document
                markdown_cache = self.cache_manager.get_markdown_cache(page.document_id)
                logger.debug(
                    f"Checking markdown cache for document {page.document_id}, page {page.page_number}"
                )

                # Check if cache is valid and has this page
                if markdown_cache.is_valid(page_config_hash):
                    cached_content = markdown_cache.get_cached_markdown(page.page_number)

                    if cached_content:
                        logger.info(f"Using cached markdown for page {page.page_number}")

                        # Update page with cached content
                        page.markdown_content = cached_content
                        page.status = ProcessingStatus.COMPLETED

                        # Update metadata
                        if page.metadata:
                            page.metadata.extraction_timestamp = datetime.now()

                        # Mark conversion complete
                        stats.end_page_conversion(page.page_number)

                        logger.info(
                            f"Successfully loaded cached page {page.page_number}, content length: {len(cached_content)}"
                        )
                        return page
                    else:
                        logger.debug(f"No cached markdown found for page {page.page_number}")
                else:
                    logger.debug(f"Markdown cache invalid for page {page.page_number}")

            # No cached content available, generate new content
            logger.info(f"Generating new markdown for page {page.page_number}")

            # Render prompt template
            prompt = self.prompt_template.render(
                additional_instructions=self.config.get("additional_instructions"),
                table_format=self.config.get("table_format", "html"),  # Default to HTML
            )

            # Call LLM provider to extract text
            response = await self.llm_provider.invoke_with_image(prompt, page.image_path)

            # Clean the LLM output (remove code fences, etc.)
            markdown_content = clean_llm_output(response.content)

            # Run validation pipeline if validators are configured
            if self.validators:
                markdown_content = await self._run_validation_pipeline(markdown_content, page)

            # Cache the generated content
            if self.use_cache and self.cache_manager and markdown_content:
                page_config_hash = ConfigHasher.hash_page_config(self.full_config)
                markdown_cache = self.cache_manager.get_markdown_cache(page.document_id)

                # Always ensure configuration is saved (not just for first page)
                # This fixes the issue where cache config might not be saved if processing
                # is interrupted or pages are processed out of order
                if not markdown_cache.config_file.exists():
                    # Get total pages from document metadata if available
                    total_pages = getattr(page, "total_pages", 1)
                    if hasattr(page, "metadata") and page.metadata:
                        total_pages = page.metadata.total_pages
                    markdown_cache.save_config(page_config_hash, total_pages)
                    logger.debug(f"Saved markdown cache config for document {page.document_id}")

                # Save the markdown content
                markdown_cache.save_markdown(page.page_number, markdown_content)
                logger.debug(f"Cached markdown for page {page.page_number}")

            # Update page with markdown content
            page.markdown_content = markdown_content
            page.status = ProcessingStatus.COMPLETED

            # Update metadata
            if page.metadata:
                page.metadata.extraction_timestamp = datetime.now()

            # Mark conversion complete
            stats.end_page_conversion(page.page_number)

            logger.info(
                f"Successfully parsed page {page.page_number}, content length: {len(markdown_content) if markdown_content else 0}"
            )
            return page

        except Exception as e:
            logger.error(f"Error parsing page {page.page_number}: {e}")
            page.status = ProcessingStatus.FAILED
            page.error_message = str(e)
            stats.record_page_failure(page.page_number)
            raise PageParsingError(f"Failed to parse page {page.page_number}: {e}") from e

    async def _run_validation_pipeline(self, content: str, page: Page) -> str:
        """Run the validation pipeline on content.

        Args:
            content: The content to validate
            page: The page object

        Returns:
            Validated and potentially corrected content
        """
        logger.debug(f"Running validation pipeline for page {page.page_number}")

        # Track validation statistics
        stats = get_statistics_tracker()
        total_issues_found = 0
        total_corrections = 0

        current_content = content
        correction_attempt = 0

        while correction_attempt < self.max_correction_attempts:
            all_issues = []
            needs_correction = False

            # Run each validator and collect issues
            for validator in self.validators:
                logger.debug(f"Checking validator {validator.name}: enabled={validator.enabled}")
                if not validator.enabled:
                    logger.debug(f"Skipping disabled validator: {validator.name}")
                    continue

                logger.debug(f"Running {validator.name} validator on page {page.page_number}")
                result = await validator.validate(current_content, page)

                if not result.is_valid:
                    all_issues.extend(result.issues)
                    if validator.attempt_correction:
                        needs_correction = True

                    # Log issues from this validator
                    if result.issues:
                        logger.info(
                            f"{validator.name} found {len(result.issues)} issues on page {page.page_number}"
                        )

            # Track issues found
            if all_issues and correction_attempt == 0:
                total_issues_found = len(all_issues)

            # If all valid or no correction needed, we're done
            if not needs_correction or len(all_issues) == 0:
                if len(all_issues) == 0:
                    logger.debug(f"Page {page.page_number} passed all validators")
                else:
                    logger.info(
                        f"Page {page.page_number} has {len(all_issues)} issues but correction disabled"
                    )
                    # Log sample of issues found (max 5)
                    sample_issues = all_issues[:5]
                    for issue in sample_issues:
                        logger.info(
                            f"  - {issue.rule_id}: {issue.description[:100]}..."
                            if len(issue.description) > 100
                            else f"  - {issue.rule_id}: {issue.description}"
                        )
                    if len(all_issues) > 5:
                        logger.info(f"  ... and {len(all_issues) - 5} more issues")
                break

            # Log all issues found
            logger.info(
                f"Page {page.page_number} validation found {len(all_issues)} total issues, attempting correction"
            )

            # Log sample of issues to be corrected (max 5)
            sample_issues = all_issues[:5]
            for issue in sample_issues:
                logger.debug(
                    f"  - {issue.rule_id}: {issue.description[:100]}..."
                    if len(issue.description) > 100
                    else f"  - {issue.rule_id}: {issue.description}"
                )
            if len(all_issues) > 5:
                logger.debug(f"  ... and {len(all_issues) - 5} more issues")

            # Create combined correction prompt (include the previous attempt for context)
            correction_prompt = self._create_combined_correction_prompt(
                all_issues, page, previous_attempt=current_content
            )

            try:
                # Track correction attempt
                total_corrections += 1

                # Get corrected content from LLM
                response = await self.llm_provider.invoke_with_image(
                    correction_prompt, page.image_path
                )
                corrected_content = clean_llm_output(response.content)

                # Validate the corrected content
                corrected_issues = []
                for validator in self.validators:
                    if validator.enabled:
                        result = await validator.validate(corrected_content, page)
                        corrected_issues.extend(result.issues)

                # Check if correction improved things
                if len(corrected_issues) < len(all_issues):
                    logger.info(
                        f"Correction improved page {page.page_number}: "
                        f"{len(all_issues)} -> {len(corrected_issues)} issues"
                    )
                    if corrected_issues:
                        # Log remaining issues after improvement
                        sample_issues = corrected_issues[:5]
                        for issue in sample_issues:
                            logger.debug(
                                f"  Remaining - {issue.rule_id}: {issue.description[:100]}..."
                                if len(issue.description) > 100
                                else f"  Remaining - {issue.rule_id}: {issue.description}"
                            )
                        if len(corrected_issues) > 5:
                            logger.debug(
                                f"  ... and {len(corrected_issues) - 5} more issues remain"
                            )
                    current_content = corrected_content
                elif len(corrected_issues) == 0:
                    logger.info(f"Correction resolved all issues for page {page.page_number}")
                    current_content = corrected_content
                    break
                else:
                    logger.warning(
                        f"Correction did not improve page {page.page_number}: "
                        f"{len(all_issues)} -> {len(corrected_issues)} issues"
                    )

                    # Log sample of actual errors (max 5)
                    sample_issues = corrected_issues[:5]
                    for issue in sample_issues:
                        logger.warning(
                            f"  - {issue.rule_id}: {issue.description[:100]}..."
                            if len(issue.description) > 100
                            else f"  - {issue.rule_id}: {issue.description}"
                        )
                    if len(corrected_issues) > 5:
                        logger.warning(f"  ... and {len(corrected_issues) - 5} more issues")

                    # Don't update content if it didn't improve
                    break

            except Exception as e:
                logger.error(f"Error during correction attempt {correction_attempt + 1}: {e}")
                break

            correction_attempt += 1

        if correction_attempt >= self.max_correction_attempts:
            logger.warning(
                f"Reached max correction attempts ({self.max_correction_attempts}) for page {page.page_number}"
            )
            if "all_issues" in locals() and all_issues:
                logger.warning(f"Final issue count: {len(all_issues)} issues remain uncorrected")
                # Log sample of final issues
                sample_issues = all_issues[:3]
                for issue in sample_issues:
                    logger.warning(
                        f"  - {issue.rule_id}: {issue.description[:100]}..."
                        if len(issue.description) > 100
                        else f"  - {issue.rule_id}: {issue.description}"
                    )
                if len(all_issues) > 3:
                    logger.warning(f"  ... and {len(all_issues) - 3} more issues")

        # Record validation statistics
        issues_resolved = (
            max(0, total_issues_found - len(all_issues))
            if "all_issues" in locals()
            else total_issues_found
        )
        stats.record_validation_stats(
            page.page_number,
            corrections=total_corrections,
            issues_found=total_issues_found,
            issues_resolved=issues_resolved,
        )

        return current_content

    def _create_combined_correction_prompt(
        self, all_issues: list, page: Page, previous_attempt: str | None = None
    ) -> str:
        """Create a combined correction prompt from all validator issues.

        Args:
            all_issues: List of all validation issues
            page: The page object
            previous_attempt: The previous markdown extraction attempt (optional)

        Returns:
            Combined correction prompt
        """
        # Group issues by validator
        issues_by_validator = {}
        for issue in all_issues:
            # Determine which validator this issue came from based on rule prefix
            for validator in self.validators:
                if issue.rule_id.startswith(validator.get_rule_prefix()):
                    if validator.name not in issues_by_validator:
                        issues_by_validator[validator.name] = []
                    issues_by_validator[validator.name].append(issue)
                    break

        # Collect correction instructions from each validator
        all_instructions = []
        for validator in self.validators:
            if validator.name in issues_by_validator:
                validator_issues = issues_by_validator[validator.name]
                if validator_issues and validator.attempt_correction:
                    instructions = validator.create_correction_instructions(validator_issues)
                    if instructions:
                        all_instructions.append(instructions)

        # Combine all instructions
        combined_instructions = "\n\n".join(all_instructions)

        # Add the previous attempt if provided to give context
        previous_attempt_section = ""
        if previous_attempt:
            # Truncate if too long (keep first 2000 chars for context)
            truncated = (
                previous_attempt[:2000] + "..."
                if len(previous_attempt) > 2000
                else previous_attempt
            )
            previous_attempt_section = f"""
## Previous Extraction Attempt

Your previous markdown extraction had validation issues. Here's the beginning of what you generated:

```markdown
{truncated}
```

Please review the issues below and generate a corrected version.
"""

        # Add overall guidance
        final_instructions = f"""
# Correction Required
{previous_attempt_section}
Please extract the content from the image again, addressing the following issues:

{combined_instructions}

## General Requirements

1. Output ONLY the extracted markdown content from the document
2. Do not include any explanations or comments
3. Ensure all issues mentioned above are resolved
4. Preserve all information from the source document
5. Maintain proper markdown formatting throughout
6. Learn from the previous attempt and avoid repeating the same mistakes
"""

        # Render the template with correction instructions
        prompt = self.prompt_template.render(
            additional_instructions=final_instructions,
        )

        return prompt

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up LLM page parser resources")

        # Cleanup the provider
        await self.llm_provider.cleanup()
