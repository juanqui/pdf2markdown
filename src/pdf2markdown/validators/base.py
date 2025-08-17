"""Base classes for content validators."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pdf2markdown.core import Page
from pdf2markdown.llm_providers import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    line_number: int
    column_number: int
    rule_id: str
    rule_name: str
    description: str
    extra_info: str = ""
    severity: str = "error"  # error, warning, info

    def to_string(self) -> str:
        """Convert issue to a readable string format."""
        location = f"Line {self.line_number}"
        if self.column_number > 0:
            location += f", Column {self.column_number}"
        rule = f"[{self.rule_id}] {self.rule_name}"
        info = f" - {self.extra_info}" if self.extra_info else ""
        return f"{location}: {rule} - {self.description}{info}"


@dataclass
class ValidationResult:
    """Result of content validation."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    corrected_content: str | None = None
    error_message: str | None = None
    validator_name: str = ""

    def get_issues_summary(self) -> str:
        """Get a formatted summary of all issues."""
        if not self.issues:
            return f"No validation issues found by {self.validator_name}."

        summary = f"{self.validator_name} found {len(self.issues)} issue(s):\n"
        for issue in self.issues:
            summary += f"  • {issue.to_string()}\n"
        return summary


class BaseValidator(ABC):
    """Abstract base class for content validators."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the validator with configuration.

        Args:
            config: Validator-specific configuration dictionary
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.attempt_correction = config.get("attempt_correction", True)
        self.name = self.__class__.__name__

        logger.debug(f"Initialized {self.name} with enabled={self.enabled}")

    @abstractmethod
    async def validate(self, content: str, page: Page) -> ValidationResult:
        """Validate content and return validation result.

        Args:
            content: The content to validate
            page: The page object with metadata

        Returns:
            ValidationResult with any issues found
        """
        pass

    @abstractmethod
    def create_correction_instructions(self, issues: list[ValidationIssue]) -> str:
        """Create correction instructions for the issues found.

        Args:
            issues: List of validation issues from this validator

        Returns:
            Formatted correction instructions for the LLM
        """
        pass

    async def validate_and_correct(
        self,
        content: str,
        page: Page,
        llm_provider: LLMProvider,
        prompt_template: Any,
    ) -> ValidationResult:
        """Validate content and optionally attempt correction.

        Args:
            content: The content to validate
            page: The page object with metadata
            llm_provider: LLM provider for correction
            prompt_template: Template for generating prompts

        Returns:
            ValidationResult with corrected content if correction was attempted
        """
        # First validate the content
        result = await self.validate(content, page)
        result.validator_name = self.name

        # If valid or correction not enabled, return as-is
        if result.is_valid or not self.attempt_correction:
            return result

        # Correction is handled by the pipeline orchestrator
        # Individual validators only provide instructions
        return result

    def filter_issues_for_validator(self, issues: list[ValidationIssue]) -> list[ValidationIssue]:
        """Filter issues to only those from this validator.

        Args:
            issues: List of all validation issues

        Returns:
            List of issues from this validator only
        """
        # Match based on the validator name prefix in rule_id
        prefix = self.get_rule_prefix()
        return [issue for issue in issues if issue.rule_id.startswith(prefix)]

    def get_rule_prefix(self) -> str:
        """Get the rule ID prefix for this validator.

        Returns:
            Rule ID prefix (e.g., "MD" for markdown, "REP" for repetition)
        """
        # Default implementation - subclasses should override
        return self.name[:3].upper()
