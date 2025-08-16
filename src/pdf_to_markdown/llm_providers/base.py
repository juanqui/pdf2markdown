"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the LLM provider with configuration.

        Args:
            config: Provider-specific configuration
        """
        self.config = config

    @abstractmethod
    async def invoke_with_image(self, prompt: str, image_path: Path, **kwargs: Any) -> LLMResponse:
        """Invoke the LLM with a text prompt and an image.

        Args:
            prompt: Text prompt for the LLM
            image_path: Path to the image file
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            LLMConnectionError: If there's an error connecting to the LLM
        """
        pass

    @abstractmethod
    async def invoke_with_image_base64(
        self, prompt: str, image_base64: str, **kwargs: Any
    ) -> LLMResponse:
        """Invoke the LLM with a text prompt and a base64-encoded image.

        Args:
            prompt: Text prompt for the LLM
            image_base64: Base64-encoded image string
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            LLMConnectionError: If there's an error connecting to the LLM
        """
        pass

    @abstractmethod
    async def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Invoke the LLM with a text prompt only.

        Args:
            prompt: Text prompt for the LLM
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            LLMConnectionError: If there's an error connecting to the LLM
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup any resources used by the provider."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the provider configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        pass
