"""Markdown validator using PyMarkdown for linting and validation."""

import logging
from typing import Any

from pymarkdown.api import PyMarkdownApi, PyMarkdownApiException

from pdf2markdown.core import Page
from pdf2markdown.validators.base import BaseValidator, ValidationIssue, ValidationResult

logger = logging.getLogger(__name__)


class MarkdownValidator(BaseValidator):
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
        super().__init__(config)
        self.disabled_rules = config.get("disabled_rules", [])
        self.enabled_rules = config.get("enabled_rules", [])
        self.strict_mode = config.get("strict_mode", False)
        self.max_line_length = config.get("max_line_length", 1000)

        # Initialize PyMarkdown API
        self._init_pymarkdown()

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
            "MD033",  # Inline HTML (common in technical documents and tables)
            "MD026",  # Trailing punctuation present in heading text (common in PDF headings)
            "MD042",  # No empty links (LLMs may generate placeholder links during extraction)
            "MD036",  # Emphasis possibly used instead of a heading element (common in PDF extraction)
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

    def get_rule_prefix(self) -> str:
        """Get the rule ID prefix for markdown validator."""
        return "MD"

    async def validate(self, content: str, page: Page) -> ValidationResult:
        """Validate markdown content.

        Args:
            content: The markdown content to validate
            page: The page object with metadata

        Returns:
            ValidationResult with issues found
        """
        if not content:
            return ValidationResult(
                is_valid=False,
                error_message="Empty markdown content",
                validator_name=self.name,
            )

        try:
            # Scan the markdown content
            scan_result = self.pymarkdown.scan_string(content)

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

            return ValidationResult(
                is_valid=is_valid,
                issues=issues,
                validator_name=self.name,
            )

        except PyMarkdownApiException as e:
            logger.error(f"PyMarkdown API error during validation: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}",
                validator_name=self.name,
            )
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Unexpected error: {str(e)}",
                validator_name=self.name,
            )

    def create_correction_instructions(self, issues: list[ValidationIssue]) -> str:
        """Create correction instructions for markdown syntax issues.

        Args:
            issues: List of validation issues from this validator

        Returns:
            Formatted correction instructions for the LLM
        """
        if not issues:
            return ""

        # Group issues by type for clearer instructions
        issues_by_rule = {}
        for issue in issues:
            if issue.rule_id not in issues_by_rule:
                issues_by_rule[issue.rule_id] = []
            issues_by_rule[issue.rule_id].append(issue)

        # Create correction instructions
        instructions = """
## Markdown Syntax Issues Detected

The markdown extraction has syntax issues that need to be corrected:

"""

        for rule_id, rule_issues in issues_by_rule.items():
            if rule_issues:
                first_issue = rule_issues[0]
                instructions += f"### {rule_id}: {first_issue.rule_name}\n"
                instructions += f"**Rule**: {first_issue.description}\n"
                instructions += f"**Occurrences** ({len(rule_issues)}):\n"

                # List specific occurrences (limit to first 5 per rule)
                for issue in rule_issues[:5]:
                    instructions += f"  - Line {issue.line_number}, Column {issue.column_number}"
                    if issue.extra_info:
                        instructions += f" - {issue.extra_info}"
                    instructions += "\n"

                if len(rule_issues) > 5:
                    instructions += f"  - ... and {len(rule_issues) - 5} more\n"

                instructions += "\n"

        instructions += """
## Instructions

Please extract the content from the image again, ensuring that:
1. All markdown syntax is valid and properly formatted
2. The issues listed above are resolved
3. The content accuracy is maintained
4. Tables use proper markdown pipe syntax
5. All formatting follows markdown best practices

Focus particularly on fixing the validation issues while preserving all information from the document.
"""

        return instructions
