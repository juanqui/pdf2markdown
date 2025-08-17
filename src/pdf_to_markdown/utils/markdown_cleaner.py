"""Utilities for cleaning and processing markdown content."""

import logging
import re

logger = logging.getLogger(__name__)


def remove_markdown_fences(content: str) -> str:
    """Remove markdown code fences that wrap the entire content.

    Some LLMs wrap their entire output in ```markdown ... ``` fences.
    This function removes those outer fences while preserving any
    code blocks that are part of the actual content.

    Args:
        content: The markdown content to clean

    Returns:
        The cleaned markdown content
    """
    if not content:
        return content

    # Remove leading/trailing whitespace for analysis
    trimmed = content.strip()

    # Pattern to match markdown fences at the start and end of content
    # This specifically looks for ```markdown or ``` at the very beginning
    # and ``` at the very end
    patterns = [
        # Match ```markdown\n...\n```
        (r"^```markdown\s*\n(.*)\n```\s*$", r"\1"),
        # Match ```md\n...\n```
        (r"^```md\s*\n(.*)\n```\s*$", r"\1"),
        # Match ```\n...\n``` (generic code fence wrapping entire content)
        (r"^```\s*\n(.*)\n```\s*$", r"\1"),
    ]

    for pattern, replacement in patterns:
        match = re.match(pattern, trimmed, re.DOTALL)
        if match:
            cleaned = re.sub(pattern, replacement, trimmed, flags=re.DOTALL)
            logger.debug(f"Removed outer markdown fence from content (pattern: {pattern[:20]}...)")
            return cleaned

    # No outer fences found, return original content
    return content


def clean_llm_output(content: str) -> str:
    """Clean LLM output by removing artifacts and unwanted formatting.

    This is the main cleaning function that applies all necessary
    transformations to LLM-generated markdown.

    Args:
        content: The raw LLM output

    Returns:
        The cleaned markdown content
    """
    if not content:
        return content

    # Remove outer markdown fences
    content = remove_markdown_fences(content)

    # Additional cleaning can be added here in the future
    # For example:
    # - Remove "Here is the content:" type preambles
    # - Fix common markdown formatting issues
    # - Normalize line endings

    return content
