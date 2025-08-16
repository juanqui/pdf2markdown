"""Markdown validator using PyMarkdown for linting and validation."""

import logging
from dataclasses import dataclass, field
from typing import Any

from pymarkdown.api import PyMarkdownApi, PyMarkdownApiException

from pdf_to_markdown.core import Page
from pdf_to_markdown.llm_providers import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a single markdown validation issue."""

    line_number: int
    column_number: int
    rule_id: str
    rule_name: str
    description: str
    extra_info: str = ""

    def to_string(self) -> str:
        """Convert issue to a readable string format."""
        location = f"Line {self.line_number}, Column {self.column_number}"
        rule = f"[{self.rule_id}] {self.rule_name}"
        info = f" - {self.extra_info}" if self.extra_info else ""
        return f"{location}: {rule} - {self.description}{info}"


@dataclass
class ValidationResult:
    """Result of markdown validation."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    corrected_markdown: str | None = None
    error_message: str | None = None

    def get_issues_summary(self) -> str:
        """Get a formatted summary of all issues."""
        if not self.issues:
            return "No validation issues found."

        summary = f"Found {len(self.issues)} validation issue(s):\n"
        for issue in self.issues:
            summary += f"  • {issue.to_string()}\n"
        return summary


class MarkdownValidator:
    """Validates and corrects markdown content using PyMarkdown."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the markdown validator.

        Args:
            config: Configuration dictionary with optional settings:
                - disabled_rules: List of rule IDs to disable
                - enabled_rules: List of rule IDs to enable
                - strict_mode: Enable strict validation (default: False)
                - max_line_length: Maximum line length for MD013 rule
                - attempt_correction: Whether to attempt correction (default: True)
        """
        self.config = config
        self.disabled_rules = config.get("disabled_rules", [])
        self.enabled_rules = config.get("enabled_rules", [])
        self.strict_mode = config.get("strict_mode", False)
        self.max_line_length = config.get("max_line_length", 1000)
        self.attempt_correction = config.get("attempt_correction", True)

        # Initialize PyMarkdown API
        self._init_pymarkdown()

        logger.info(f"Initialized MarkdownValidator with strict_mode={self.strict_mode}")

    def _init_pymarkdown(self) -> None:
        """Initialize PyMarkdown API with configuration."""
        self.pymarkdown = PyMarkdownApi().log_error_and_above()

        # Disable some rules that might be too strict for LLM-generated content
        default_disabled_rules = [
            "MD041",  # First line should be a top-level heading (page fragments)
            "MD012",  # Multiple consecutive blank lines (formatting preference)
            "MD022",  # Headings should be surrounded by blank lines (too strict)
            "MD031",  # Fenced code blocks should be surrounded by blank lines
            "MD032",  # Lists should be surrounded by blank lines
            "MD025",  # Multiple top-level headings (technical docs often have multiple H1s)
            "MD024",  # Multiple headings with the same content (common in tech docs)
            "MD013",  # Line length (technical content often has long lines)
            "MD047",  # Files must end with single newline (not critical for generated content)
            "MD040",  # Fenced code blocks should have a language specified (often unknown in PDFs)
        ]

        # Combine with user-specified disabled rules
        all_disabled_rules = list(set(default_disabled_rules + self.disabled_rules))

        for rule_id in all_disabled_rules:
            try:
                self.pymarkdown.disable_rule_by_identifier(rule_id.lower())
            except Exception as e:
                logger.warning(f"Could not disable rule {rule_id}: {e}")

        # Enable any specifically requested rules
        for rule_id in self.enabled_rules:
            try:
                self.pymarkdown.enable_rule_by_identifier(rule_id.lower())
            except Exception as e:
                logger.warning(f"Could not enable rule {rule_id}: {e}")

        # Set configuration properties
        if self.max_line_length:
            self.pymarkdown.set_integer_property("plugins.md013.line_length", self.max_line_length)

    def validate(self, markdown_content: str) -> ValidationResult:
        """Validate markdown content.

        Args:
            markdown_content: The markdown content to validate

        Returns:
            ValidationResult with issues found
        """
        if not markdown_content:
            return ValidationResult(is_valid=False, error_message="Empty markdown content")

        try:
            # Scan the markdown content
            scan_result = self.pymarkdown.scan_string(markdown_content)

            # Convert scan failures to ValidationIssues
            issues = []
            for failure in scan_result.scan_failures:
                issue = ValidationIssue(
                    line_number=failure.line_number,
                    column_number=failure.column_number,
                    rule_id=failure.rule_id,
                    rule_name=failure.rule_name,
                    description=failure.rule_description,
                    extra_info=failure.extra_error_information,
                )
                issues.append(issue)

            # Check for pragma errors (malformed inline configuration)
            if scan_result.pragma_errors:
                logger.warning(f"Pragma errors found: {scan_result.pragma_errors}")

            is_valid = len(issues) == 0

            return ValidationResult(is_valid=is_valid, issues=issues)

        except PyMarkdownApiException as e:
            logger.error(f"PyMarkdown API error during validation: {e}")
            return ValidationResult(is_valid=False, error_message=f"Validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return ValidationResult(is_valid=False, error_message=f"Unexpected error: {str(e)}")

    async def validate_and_correct(
        self, markdown_content: str, page: Page, llm_provider: LLMProvider, prompt_template: Any
    ) -> ValidationResult:
        """Validate markdown and attempt correction using LLM if issues found.

        Args:
            markdown_content: The markdown content to validate
            page: The original page object with image
            llm_provider: LLM provider to use for correction
            prompt_template: Jinja2 template for generating prompts

        Returns:
            ValidationResult with corrected markdown if correction was attempted
        """
        # First, validate the markdown
        validation_result = self.validate(markdown_content)

        # If valid or correction not enabled, return as-is
        if validation_result.is_valid or not self.attempt_correction:
            return validation_result

        # If we have issues and correction is enabled, attempt to fix
        if validation_result.issues and llm_provider and page.image_path:
            logger.info(f"Attempting to correct {len(validation_result.issues)} markdown issues")

            try:
                # Create a correction prompt with the issues
                correction_prompt = self._create_correction_prompt(
                    validation_result.issues, prompt_template, page
                )

                # Call LLM with the correction prompt and original image
                response = await llm_provider.invoke_with_image(correction_prompt, page.image_path)

                corrected_markdown = response.content

                # Validate the corrected markdown
                corrected_validation = self.validate(corrected_markdown)

                # Log improvement
                original_issues = len(validation_result.issues)
                corrected_issues = len(corrected_validation.issues)

                if corrected_issues < original_issues:
                    logger.info(
                        f"Correction improved markdown: {original_issues} -> {corrected_issues} issues"
                    )
                elif corrected_issues == 0:
                    logger.info("Correction resolved all markdown issues")
                else:
                    logger.warning(
                        f"Correction did not improve: {original_issues} -> {corrected_issues} issues"
                    )

                # Return the corrected result
                corrected_validation.corrected_markdown = corrected_markdown
                return corrected_validation

            except Exception as e:
                logger.error(f"Error during markdown correction: {e}")
                # Return original validation result if correction fails
                validation_result.error_message = f"Correction failed: {str(e)}"
                return validation_result

        return validation_result

    def _create_correction_prompt(
        self, issues: list[ValidationIssue], prompt_template: Any, page: Page
    ) -> str:
        """Create a prompt for correcting markdown issues.

        Args:
            issues: List of validation issues found
            prompt_template: Original prompt template
            page: Page object with metadata

        Returns:
            Prompt string for correction
        """
        # Group issues by type for clearer instructions
        issues_by_rule = {}
        for issue in issues:
            if issue.rule_id not in issues_by_rule:
                issues_by_rule[issue.rule_id] = []
            issues_by_rule[issue.rule_id].append(issue)

        # Create correction instructions
        correction_instructions = """
## Markdown Correction Required

The previous markdown extraction had the following validation issues that need to be corrected:

"""

        for rule_id, rule_issues in issues_by_rule.items():
            if rule_issues:
                first_issue = rule_issues[0]
                correction_instructions += f"### {rule_id}: {first_issue.rule_name}\n"
                correction_instructions += f"**Rule**: {first_issue.description}\n"
                correction_instructions += f"**Occurrences** ({len(rule_issues)}):\n"

                # List specific occurrences (limit to first 5 per rule)
                for issue in rule_issues[:5]:
                    correction_instructions += (
                        f"  - Line {issue.line_number}, Column {issue.column_number}"
                    )
                    if issue.extra_info:
                        correction_instructions += f" - {issue.extra_info}"
                    correction_instructions += "\n"

                if len(rule_issues) > 5:
                    correction_instructions += f"  - ... and {len(rule_issues) - 5} more\n"

                correction_instructions += "\n"

        correction_instructions += """
## Instructions

Please extract the content from the image again, ensuring that:
1. All markdown syntax is valid and properly formatted
2. The issues listed above are resolved
3. The content accuracy is maintained
4. Tables use proper markdown pipe syntax
5. All formatting follows markdown best practices

Focus particularly on fixing the validation issues while preserving all information from the document.
"""

        # Render the template with correction instructions
        base_prompt = prompt_template.render(
            page_number=page.page_number,
            total_pages=page.metadata.total_pages if page.metadata else None,
            additional_instructions=correction_instructions,
        )

        return base_prompt
