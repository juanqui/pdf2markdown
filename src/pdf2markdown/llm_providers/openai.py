"""OpenAI-compatible LLM provider implementation."""

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from pdf2markdown.core import LLMConnectionError

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OpenAILLMProvider(LLMProvider):
    """LLM provider for OpenAI-compatible APIs."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the OpenAI provider with configuration.

        Args:
            config: Configuration dictionary with the following keys:
                - endpoint (str): OpenAI API endpoint URL
                - api_key (str): OpenAI API key
                - model (str): Model to use (default: "gpt-4o-mini")
                - max_tokens (int): Maximum tokens for response
                - temperature (float): Temperature for generation
                - timeout (int): Timeout in seconds for API calls
                - presence_penalty (float): Penalize new tokens based on presence in text so far
                - frequency_penalty (float): Penalize new tokens based on frequency in text so far
                - repetition_penalty (float): Alternative repetition penalty (some providers)
        """
        super().__init__(config)

        self.endpoint = config.get("endpoint", "https://api.openai.com/v1")
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise LLMConnectionError("OpenAI API key is required")

        self.model = config.get("model", "gpt-4o-mini")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.1)
        self.timeout = config.get("timeout", 60)

        # Penalty parameters to reduce repetition
        self.presence_penalty = config.get("presence_penalty", 0.0)
        self.frequency_penalty = config.get("frequency_penalty", 0.0)
        self.repetition_penalty = config.get("repetition_penalty")  # May be None

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.endpoint if self.endpoint != "https://api.openai.com/v1" else None,
            timeout=self.timeout,
        )

        # Determine which penalties will be used based on endpoint
        is_openai_official = self.endpoint.startswith("https://api.openai.com")
        penalties_info = []
        if is_openai_official:
            penalties_info.append(f"presence_penalty={self.presence_penalty}")
            penalties_info.append(f"frequency_penalty={self.frequency_penalty}")
        else:
            if self.repetition_penalty is not None:
                penalties_info.append(f"repetition_penalty={self.repetition_penalty}")
            penalties_info.append(f"presence_penalty={self.presence_penalty}")
            penalties_info.append(f"frequency_penalty={self.frequency_penalty}")

        logger.debug(
            f"Initialized OpenAILLMProvider with model={self.model}, endpoint={self.endpoint}, "
            f"{', '.join(penalties_info)}"
        )

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string

        Raises:
            FileNotFoundError: If the image file doesn't exist
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def _strip_thinking_tags(content: str) -> str:
        """Strip <think>...</think> tags from the content.

        This is used to remove reasoning/thinking content from models that
        output their internal thought process.

        Args:
            content: The content potentially containing thinking tags

        Returns:
            Content with thinking tags removed
        """
        if not content:
            return content

        # Remove <think>...</think> tags and their contents
        # Use non-greedy matching and DOTALL flag to handle multi-line content
        pattern = r"<think>.*?</think>"
        cleaned = re.sub(pattern, "", content, flags=re.DOTALL)

        # Clean up any extra whitespace that might be left
        cleaned = cleaned.strip()

        # Log if we actually removed thinking tags
        if cleaned != content:
            logger.debug(
                f"Removed thinking tags from response (original: {len(content)} chars, cleaned: {len(cleaned)} chars)"
            )

        return cleaned

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True
    )
    async def _call_api(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        """Call the OpenAI API with retry logic.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for the API call

        Returns:
            LLMResponse with the API response

        Raises:
            LLMConnectionError: If there's an error calling the API
        """
        try:
            # Merge kwargs with default parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            # Add penalty parameters based on the endpoint type
            is_openai_official = self.endpoint.startswith("https://api.openai.com")
            logger.debug(f"Endpoint: {self.endpoint}, Is OpenAI official: {is_openai_official}")

            if is_openai_official:
                # Official OpenAI API only supports frequency and presence penalties
                if self.presence_penalty != 0.0:
                    api_params["presence_penalty"] = self.presence_penalty
                if self.frequency_penalty != 0.0:
                    api_params["frequency_penalty"] = self.frequency_penalty
                # Never send repetition_penalty to official OpenAI API
            else:
                # For OpenAI-compatible APIs (local servers, vLLM, Ollama, etc.)
                # The OpenAI Python SDK doesn't recognize repetition_penalty as a valid parameter
                # So we need to pass it via extra_body for servers that support it
                if self.repetition_penalty is not None:
                    # Pass repetition_penalty via extra_body to bypass SDK validation
                    api_params["extra_body"] = {"repetition_penalty": self.repetition_penalty}

                # Always include OpenAI-style penalties if set (these are SDK-recognized)
                if self.presence_penalty != 0.0:
                    api_params["presence_penalty"] = self.presence_penalty
                if self.frequency_penalty != 0.0:
                    api_params["frequency_penalty"] = self.frequency_penalty

            # Update with any additional kwargs
            api_params.update(kwargs)

            # Debug log the parameters being sent
            params_for_log = {k: v for k, v in api_params.items() if k != "messages"}
            logger.debug(f"API parameters (excluding messages): {params_for_log}")

            # Call the API
            response = await self.client.chat.completions.create(**api_params)

            # Extract the response
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content

                # Strip thinking tags from reasoning/thinking models
                content = self._strip_thinking_tags(content)

                usage = response.usage.model_dump() if response.usage else None

                logger.debug(f"API response length: {len(content) if content else 0}")

                return LLMResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                    metadata={
                        "finish_reason": response.choices[0].finish_reason,
                        "system_fingerprint": response.system_fingerprint,
                    },
                )
            else:
                raise LLMConnectionError("Empty response from OpenAI API")

        except json.JSONDecodeError as e:
            # This typically happens when the API returns HTML (error page) instead of JSON
            logger.error(f"Invalid JSON response from API: {e}")
            logger.debug(
                "Failed to parse response - likely an HTML error page or rate limit message"
            )
            raise LLMConnectionError(
                f"API returned invalid response format (expected JSON). "
                f"This often happens with rate limits or server errors. Error: {str(e)[:100]}"
            ) from e

        except Exception as e:
            # Check if this is a JSON parsing error from the OpenAI client
            error_msg = str(e)

            # Log the full error for debugging
            logger.error(f"Error calling API: {error_msg}")

            # Check for specific error patterns
            if "Expecting value" in error_msg:
                # This is a JSON parsing error
                logger.error(
                    "API returned non-JSON response (possibly HTML error page or rate limit)"
                )
                raise LLMConnectionError(
                    "API returned invalid response (not JSON). "
                    "This usually indicates rate limiting, server errors, or invalid endpoint. "
                    "Check your API endpoint and credentials."
                ) from e
            elif "timeout" in error_msg.lower():
                raise LLMConnectionError(
                    f"API request timed out after {self.timeout} seconds"
                ) from e
            elif "connection" in error_msg.lower():
                raise LLMConnectionError(f"Connection error: {error_msg}") from e
            else:
                raise LLMConnectionError(f"Failed to call API: {error_msg}") from e

    async def invoke_with_image(self, prompt: str, image_path: Path, **kwargs: Any) -> LLMResponse:
        """Invoke the LLM with a text prompt and an image.

        Args:
            prompt: Text prompt for the LLM
            image_path: Path to the image file
            **kwargs: Additional parameters for the API call

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            LLMConnectionError: If there's an error calling the API
            FileNotFoundError: If the image file doesn't exist
        """
        logger.debug(f"Invoking LLM with image from {image_path}")

        # Encode the image
        image_base64 = self._encode_image(image_path)

        # Call the base64 method
        return await self.invoke_with_image_base64(prompt, image_base64, **kwargs)

    async def invoke_with_image_base64(
        self, prompt: str, image_base64: str, **kwargs: Any
    ) -> LLMResponse:
        """Invoke the LLM with a text prompt and a base64-encoded image.

        Args:
            prompt: Text prompt for the LLM
            image_base64: Base64-encoded image string
            **kwargs: Additional parameters for the API call

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            LLMConnectionError: If there's an error calling the API
        """
        # Prepare the message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]

        return await self._call_api(messages, **kwargs)

    async def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Invoke the LLM with a text prompt only.

        Args:
            prompt: Text prompt for the LLM
            **kwargs: Additional parameters for the API call

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            LLMConnectionError: If there's an error calling the API
        """
        # Prepare text-only message
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        return await self._call_api(messages, **kwargs)

    async def cleanup(self) -> None:
        """Cleanup the OpenAI client resources."""
        logger.info("Cleaning up OpenAI LLM provider resources")

        # Close the client if it has a close method
        if hasattr(self.client, "close"):
            await self.client.close()

    def validate_config(self) -> bool:
        """Validate the provider configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        if not self.api_key:
            logger.error("API key is missing")
            return False

        if not self.endpoint:
            logger.error("Endpoint is missing")
            return False

        if self.temperature < 0 or self.temperature > 2:
            logger.error(f"Invalid temperature: {self.temperature}")
            return False

        if self.max_tokens < 1:
            logger.error(f"Invalid max_tokens: {self.max_tokens}")
            return False

        return True
