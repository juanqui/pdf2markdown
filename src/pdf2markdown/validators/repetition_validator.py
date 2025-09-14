"""Validator for detecting content repetition in markdown."""

import logging
import re
from difflib import SequenceMatcher
from typing import Any

from pdf2markdown.core import Page
from pdf2markdown.validators.base import BaseValidator, ValidationIssue, ValidationResult

logger = logging.getLogger(__name__)


class RepetitionValidator(BaseValidator):
    """Detects various types of content repetition in markdown."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the repetition validator.

        Args:
            config: Configuration dictionary with settings:
                - consecutive_threshold: Number of consecutive repeats to flag (default: 3)
                - window_size: Size of sliding window for near-duplicate detection (default: 10)
                - window_threshold: Number of occurrences in window to flag (default: 3)
                - check_exact_lines: Check for exact line matches (default: True)
                - check_normalized_lines: Check ignoring whitespace/punctuation (default: True)
                - check_paragraphs: Check for duplicate paragraphs (default: True)
                - check_patterns: Detect repetitive patterns (default: True)
                - check_character_repetition: Check for excessive single-character repetition (default: True)
                - min_pattern_length: Minimum chars for pattern detection (default: 20)
                - pattern_similarity_threshold: Similarity threshold 0-1 (default: 0.9)
                - min_line_length: Minimum line length to check (default: 5)
                - max_character_repetition: Max consecutive repetitions of single char (default: 15)
                - character_repetition_chars: Characters to check for repetition (default: "0123456789-=_.")
        """
        super().__init__(config)

        # Detection thresholds
        self.consecutive_threshold = config.get("consecutive_threshold", 3)
        self.window_size = config.get("window_size", 10)
        self.window_threshold = config.get("window_threshold", 3)

        # What to check
        self.check_exact_lines = config.get("check_exact_lines", True)
        self.check_normalized_lines = config.get("check_normalized_lines", True)
        self.check_paragraphs = config.get("check_paragraphs", True)
        self.check_patterns = config.get("check_patterns", True)
        self.check_character_repetition = config.get("check_character_repetition", True)

        # Pattern detection settings
        self.min_pattern_length = config.get("min_pattern_length", 20)
        self.pattern_similarity_threshold = config.get("pattern_similarity_threshold", 0.9)
        self.min_line_length = config.get("min_line_length", 5)

        # Character repetition settings
        self.max_character_repetition = config.get("max_character_repetition", 15)
        self.character_repetition_chars = config.get("character_repetition_chars", "0123456789-=_.")

        logger.info(
            f"Initialized RepetitionValidator with consecutive_threshold={self.consecutive_threshold}, "
            f"window_size={self.window_size}"
        )

    def get_rule_prefix(self) -> str:
        """Get the rule ID prefix for repetition validator."""
        return "REP"

    async def validate(self, content: str, page: Page) -> ValidationResult:
        """Validate content for repetition issues.

        Args:
            content: The markdown content to validate
            page: The page object with metadata

        Returns:
            ValidationResult with any repetition issues found
        """
        if not content:
            return ValidationResult(
                is_valid=False,
                error_message="Empty content",
                validator_name=self.name,
            )

        issues = []
        lines = content.split("\n")

        # Check various types of repetition
        if self.check_exact_lines:
            issues.extend(self._check_consecutive_duplicates(lines))
            issues.extend(self._check_window_duplicates(lines))

        if self.check_normalized_lines:
            issues.extend(self._check_normalized_duplicates(lines))

        if self.check_paragraphs:
            issues.extend(self._check_paragraph_duplicates(content))

        if self.check_patterns:
            issues.extend(self._check_pattern_repetition(lines))

        if self.check_character_repetition:
            issues.extend(self._check_character_repetition(lines))

        # Remove duplicate issues (same line might be flagged by multiple checks)
        unique_issues = self._deduplicate_issues(issues)

        is_valid = len(unique_issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=unique_issues,
            validator_name=self.name,
        )

    def _check_consecutive_duplicates(self, lines: list[str]) -> list[ValidationIssue]:
        """Check for consecutive duplicate lines.

        Args:
            lines: List of lines to check

        Returns:
            List of validation issues for consecutive duplicates
        """
        issues = []
        consecutive_count = 1
        last_line = ""

        for i, line in enumerate(lines):
            # Skip very short lines
            if len(line.strip()) < self.min_line_length:
                consecutive_count = 1
                last_line = line
                continue

            if line == last_line and line.strip():
                consecutive_count += 1
                if consecutive_count == self.consecutive_threshold:
                    # Only report once when threshold is reached
                    issues.append(
                        ValidationIssue(
                            line_number=i + 1 - consecutive_count + 2,  # Start line of repetition
                            column_number=0,
                            rule_id="REP001",
                            rule_name="consecutive-duplicates",
                            description=f"Line repeated {consecutive_count} times consecutively",
                            extra_info=(
                                f'Content: "{line[:50]}..."'
                                if len(line) > 50
                                else f'Content: "{line}"'
                            ),
                            severity="error",
                        )
                    )
                elif consecutive_count > self.consecutive_threshold:
                    # Update the existing issue
                    for issue in issues:
                        if (
                            issue.rule_id == "REP001"
                            and issue.extra_info
                            and line[:50] in issue.extra_info
                        ):
                            issue.description = (
                                f"Line repeated {consecutive_count} times consecutively"
                            )
                            break
            else:
                consecutive_count = 1
                last_line = line

        return issues

    def _check_window_duplicates(self, lines: list[str]) -> list[ValidationIssue]:
        """Check for duplicate lines within a sliding window.

        Args:
            lines: List of lines to check

        Returns:
            List of validation issues for window duplicates
        """
        issues = []
        reported_lines = set()  # Track which lines we've already reported

        for i, line in enumerate(lines):
            # Skip very short lines and empty lines
            if len(line.strip()) < self.min_line_length:
                continue

            # Skip if we've already reported this line
            if (i, line) in reported_lines:
                continue

            # Get the window of lines before this one
            window_start = max(0, i - self.window_size)
            window = lines[window_start:i]

            # Count occurrences in the window
            count_in_window = window.count(line)

            # Also check lines after (within window distance)
            window_end = min(len(lines), i + self.window_size)
            future_window = lines[i + 1 : window_end]
            count_in_future = future_window.count(line)

            total_count = count_in_window + count_in_future + 1  # +1 for current line

            if total_count >= self.window_threshold:
                # Find all occurrences to mark them as reported
                all_positions = []
                for j in range(window_start, window_end):
                    if lines[j] == line:
                        all_positions.append(j)
                        reported_lines.add((j, line))

                issues.append(
                    ValidationIssue(
                        line_number=i + 1,
                        column_number=0,
                        rule_id="REP002",
                        rule_name="window-duplicates",
                        description=f"Line appears {total_count} times within {self.window_size * 2} lines",
                        extra_info=(
                            f'Content: "{line[:50]}..."' if len(line) > 50 else f'Content: "{line}"'
                        ),
                        severity="warning",
                    )
                )

        return issues

    def _check_normalized_duplicates(self, lines: list[str]) -> list[ValidationIssue]:
        """Check for duplicates ignoring whitespace and punctuation differences.

        Args:
            lines: List of lines to check

        Returns:
            List of validation issues for normalized duplicates
        """
        issues = []

        def normalize(text: str) -> str:
            """Normalize text by removing extra whitespace and punctuation."""
            # Remove extra whitespace
            text = " ".join(text.split())
            # Remove common punctuation variations
            text = re.sub(r"[.,;:!?'\"-]", "", text)
            return text.lower().strip()

        normalized_lines = [(i, line, normalize(line)) for i, line in enumerate(lines)]

        # Group by normalized content
        normalized_groups = {}
        for i, original, normalized in normalized_lines:
            if len(normalized) < self.min_line_length:
                continue
            if normalized not in normalized_groups:
                normalized_groups[normalized] = []
            normalized_groups[normalized].append((i, original))

        # Report groups with multiple occurrences that aren't exact matches
        for _normalized, occurrences in normalized_groups.items():
            if len(occurrences) >= self.consecutive_threshold:
                # Check if they're not exact matches
                originals = [orig for _, orig in occurrences]
                if len(set(originals)) > 1:  # Different original texts
                    issues.append(
                        ValidationIssue(
                            line_number=occurrences[0][0] + 1,
                            column_number=0,
                            rule_id="REP003",
                            rule_name="normalized-duplicates",
                            description=f"Similar content appears {len(occurrences)} times with minor variations",
                            extra_info=f"Lines: {', '.join(str(i+1) for i, _ in occurrences[:5])}",
                            severity="info",
                        )
                    )

        return issues

    def _check_paragraph_duplicates(self, content: str) -> list[ValidationIssue]:
        """Check for duplicate paragraphs.

        Args:
            content: The full content to check

        Returns:
            List of validation issues for duplicate paragraphs
        """
        issues = []

        # Split into paragraphs (separated by blank lines)
        paragraphs = re.split(r"\n\s*\n", content)

        # Track paragraph occurrences with line numbers
        paragraph_map = {}
        current_line = 1

        for para in paragraphs:
            para_stripped = para.strip()
            para_lines = para.count("\n") + 1

            if len(para_stripped) >= self.min_pattern_length:
                if para_stripped not in paragraph_map:
                    paragraph_map[para_stripped] = []
                paragraph_map[para_stripped].append(current_line)

            current_line += para_lines + 1  # +1 for the blank line separator

        # Report duplicate paragraphs
        for _para, line_numbers in paragraph_map.items():
            if len(line_numbers) > 1:
                issues.append(
                    ValidationIssue(
                        line_number=line_numbers[0],
                        column_number=0,
                        rule_id="REP004",
                        rule_name="duplicate-paragraphs",
                        description=f"Paragraph appears {len(line_numbers)} times",
                        extra_info=f"At lines: {', '.join(map(str, line_numbers))}",
                        severity="error",
                    )
                )

        return issues

    def _check_pattern_repetition(self, lines: list[str]) -> list[ValidationIssue]:
        """Check for repetitive patterns in the content.

        Args:
            lines: List of lines to check

        Returns:
            List of validation issues for pattern repetition
        """
        issues = []

        # Look for lines with similar structure (e.g., same prefix)
        prefix_groups = {}

        for i, line in enumerate(lines):
            if len(line.strip()) < self.min_pattern_length:
                continue

            # Extract potential prefix (first 20-30 chars or up to first colon/dash)
            match = re.match(r"^(.{10,30})[:\-\s]", line)
            if match:
                prefix = match.group(1)
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append((i, line))

        # Check for repetitive patterns
        for prefix, occurrences in prefix_groups.items():
            if len(occurrences) >= self.consecutive_threshold:
                # Check if the content after the prefix is too similar
                contents = [line[len(prefix) :].strip() for _, line in occurrences]
                unique_contents = set(contents)

                # If most content is the same, it's likely repetition
                if len(unique_contents) <= len(occurrences) / 3:
                    issues.append(
                        ValidationIssue(
                            line_number=occurrences[0][0] + 1,
                            column_number=0,
                            rule_id="REP005",
                            rule_name="pattern-repetition",
                            description=f'Pattern "{prefix}..." repeated {len(occurrences)} times with similar content',
                            extra_info=f"Lines: {', '.join(str(i+1) for i, _ in occurrences[:5])}",
                            severity="warning",
                        )
                    )

        # Check for structural patterns using sequence matching
        for i in range(len(lines) - 3):
            if len(lines[i].strip()) < self.min_pattern_length:
                continue

            # Look for sequences of similar lines
            similar_sequence = []
            for j in range(i + 1, min(i + 10, len(lines))):
                similarity = SequenceMatcher(None, lines[i], lines[j]).ratio()
                if similarity >= self.pattern_similarity_threshold:
                    similar_sequence.append((j, similarity))

            if len(similar_sequence) >= self.consecutive_threshold - 1:
                issues.append(
                    ValidationIssue(
                        line_number=i + 1,
                        column_number=0,
                        rule_id="REP006",
                        rule_name="sequence-repetition",
                        description=f"Highly similar content repeated {len(similar_sequence) + 1} times",
                        extra_info=f"Similarity > {self.pattern_similarity_threshold:.0%}",
                        severity="info",
                    )
                )

        return issues

    def _check_character_repetition(self, lines: list[str]) -> list[ValidationIssue]:
        """Check for excessive repetition of single characters within lines.

        Args:
            lines: List of lines to check

        Returns:
            List of validation issues for character repetition
        """
        issues = []

        for i, line in enumerate(lines):
            if len(line.strip()) < self.min_line_length:
                continue

            # Check each character in our target set
            for char in self.character_repetition_chars:
                # Find all occurrences of consecutive repeated characters
                consecutive_count = 0
                start_pos = 0

                for j, c in enumerate(line):
                    if c == char:
                        if consecutive_count == 0:
                            start_pos = j
                        consecutive_count += 1
                    else:
                        # Check if we hit the threshold before resetting
                        if consecutive_count > self.max_character_repetition:
                            # Avoid flagging legitimate markdown patterns
                            if self._is_legitimate_pattern(
                                char, consecutive_count, line, start_pos
                            ):
                                consecutive_count = 0
                                continue

                            issues.append(
                                ValidationIssue(
                                    line_number=i + 1,
                                    column_number=start_pos + 1,
                                    rule_id="REP007",
                                    rule_name="character-repetition",
                                    description=f"Character '{char}' repeated {consecutive_count} times consecutively",
                                    extra_info=f"Position {start_pos}-{start_pos + consecutive_count - 1}",
                                    severity="warning",
                                )
                            )
                        consecutive_count = 0

                # Check the final sequence if line ends with repeated character
                if consecutive_count > self.max_character_repetition:
                    if not self._is_legitimate_pattern(char, consecutive_count, line, start_pos):
                        issues.append(
                            ValidationIssue(
                                line_number=i + 1,
                                column_number=start_pos + 1,
                                rule_id="REP007",
                                rule_name="character-repetition",
                                description=f"Character '{char}' repeated {consecutive_count} times consecutively",
                                extra_info=f"Position {start_pos}-{start_pos + consecutive_count - 1}",
                                severity="warning",
                            )
                        )

        return issues

    def _is_legitimate_pattern(self, char: str, count: int, line: str, start_pos: int) -> bool:
        """Check if a character repetition is a legitimate markdown pattern.

        Args:
            char: The repeated character
            count: Number of consecutive repetitions
            line: The full line containing the pattern
            start_pos: Starting position of the repetition

        Returns:
            True if this is likely a legitimate pattern, False if it's excessive repetition
        """
        # Allow reasonable markdown patterns
        if char == "-" and count <= 20 and line.strip() == char * count:
            return True  # Horizontal rule

        if char == "=" and count <= 20 and line.strip() == char * count:
            return True  # Another type of horizontal rule

        if char == "_" and count <= 10:
            return True  # Emphasis or short dividers

        if char == "." and count <= 5:
            return True  # Ellipsis or normal punctuation

        # Check if it's part of a code block or fenced code
        stripped_line = line.strip()
        if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
            return True  # Code fence

        # Check if it's part of a table or structured data where repetition might be normal
        if "|" in line and char in "- ":
            return True  # Table formatting

        return False

    def _deduplicate_issues(self, issues: list[ValidationIssue]) -> list[ValidationIssue]:
        """Remove duplicate issues that refer to the same problem.

        Args:
            issues: List of all issues found

        Returns:
            Deduplicated list of issues
        """
        if not issues:
            return issues

        # Sort by line number and severity
        severity_order = {"error": 0, "warning": 1, "info": 2}
        issues.sort(key=lambda x: (x.line_number, severity_order.get(x.severity, 3)))

        # Keep track of lines already reported for each rule type
        reported = {}
        unique_issues = []

        for issue in issues:
            # Create a key based on rule type and approximate line
            line_range = range(max(0, issue.line_number - 2), issue.line_number + 3)
            key = issue.rule_id[:3]  # Rule prefix (REP)

            if key not in reported:
                reported[key] = set()

            # Check if we've already reported an issue near this line
            if not any(line in reported[key] for line in line_range):
                unique_issues.append(issue)
                reported[key].update(line_range)

        return unique_issues

    def create_correction_instructions(self, issues: list[ValidationIssue]) -> str:
        """Create correction instructions for repetition issues.

        Args:
            issues: List of repetition issues from this validator

        Returns:
            Formatted correction instructions for the LLM
        """
        if not issues:
            return ""

        instructions = """
## Repetition Issues Detected

Your extraction contains unnecessary repetition that needs to be corrected:

"""

        # Group issues by type
        issues_by_type = {}
        for issue in issues:
            rule_name = issue.rule_name
            if rule_name not in issues_by_type:
                issues_by_type[rule_name] = []
            issues_by_type[rule_name].append(issue)

        # Format instructions for each type
        for rule_name, rule_issues in issues_by_type.items():
            if rule_name == "consecutive-duplicates":
                instructions += "### Consecutive Duplicate Lines\n"
                instructions += "The following lines are repeated multiple times in a row:\n"
                for issue in rule_issues[:3]:  # Show first 3 examples
                    instructions += f"- Line {issue.line_number}: {issue.description}\n"
                    if issue.extra_info:
                        instructions += f"  {issue.extra_info}\n"
                instructions += "**Fix**: Include each line only once unless it genuinely appears multiple times in the source.\n\n"

            elif rule_name == "window-duplicates":
                instructions += "### Nearby Duplicate Lines\n"
                instructions += "The following lines appear multiple times within a short span:\n"
                for issue in rule_issues[:3]:
                    instructions += f"- Line {issue.line_number}: {issue.description}\n"
                    if issue.extra_info:
                        instructions += f"  {issue.extra_info}\n"
                instructions += (
                    "**Fix**: Remove duplicate occurrences, keeping only unique content.\n\n"
                )

            elif rule_name == "duplicate-paragraphs":
                instructions += "### Duplicate Paragraphs\n"
                instructions += "Entire paragraphs are repeated:\n"
                for issue in rule_issues[:2]:  # Show first 2 examples
                    instructions += f"- {issue.description} ({issue.extra_info})\n"
                instructions += "**Fix**: Include each paragraph only once.\n\n"

            elif rule_name == "pattern-repetition":
                instructions += "### Repetitive Patterns\n"
                instructions += "Content follows repetitive patterns:\n"
                for issue in rule_issues[:3]:
                    instructions += f"- Line {issue.line_number}: {issue.description}\n"
                instructions += "**Fix**: Ensure varied content, not stuck on the same pattern.\n\n"

            elif rule_name == "character-repetition":
                instructions += "### Excessive Character Repetition\n"
                instructions += "Single characters are repeated excessively:\n"
                for issue in rule_issues[:3]:
                    instructions += f"- Line {issue.line_number}: {issue.description}\n"
                    if issue.extra_info:
                        instructions += f"  {issue.extra_info}\n"
                instructions += (
                    "**Fix**: Remove excessive character repetition. Replace long sequences of repeated characters "
                    "with meaningful content from the source document.\n\n"
                )

        instructions += """
## Correction Instructions

Please extract the content again, ensuring that:
1. Each piece of information appears only once (unless legitimately repeated in the source document)
2. No lines, paragraphs, or sections are unnecessarily duplicated
3. Content progresses naturally without getting stuck in loops
4. All unique information from the source document is preserved
5. The extraction is complete and accurate

Focus on producing clean, non-repetitive markdown that accurately represents the source document.
"""

        return instructions
