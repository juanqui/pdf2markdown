"""Factory for creating LLM providers."""

import logging
from typing import Any

from .base import LLMProvider
from .openai import OpenAILLMProvider

logger = logging.getLogger(__name__)


def create_llm_provider(config: dict[str, Any]) -> LLMProvider:
    """Create an LLM provider from configuration.

    Args:
        config: Provider configuration dictionary with at least:
            - provider_type (str): Type of provider to create

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider type is not supported
    """
    provider_type = config.get("provider_type", "openai").lower()

    if provider_type == "openai":
        logger.debug("Creating OpenAI LLM provider")
        return OpenAILLMProvider(config)
    elif provider_type == "transformers":
        logger.debug("Creating Transformers LLM provider")
        # Import here to avoid requiring transformers dependency if not used
        try:
            from .transformers import TransformersLLMProvider

            return TransformersLLMProvider(config)
        except ImportError as e:
            raise ImportError(
                "transformers library is not installed. "
                "Install it with: pip install pdf2markdown[transformers]"
            ) from e
    else:
        raise ValueError(
            f"Unsupported LLM provider type: {provider_type}. "
            f"Supported types: openai, transformers"
        )


def create_llm_provider_from_schema(provider_config) -> LLMProvider:
    """Create an LLM provider from a Pydantic schema.

    Args:
        provider_config: LLMProviderConfig instance

    Returns:
        Configured LLM provider instance
    """
    config_dict = provider_config.model_dump()
    return create_llm_provider(config_dict)
