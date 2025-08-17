"""LLM Provider implementations for PDF to Markdown conversion."""

from .base import LLMProvider, LLMResponse
from .factory import create_llm_provider, create_llm_provider_from_schema
from .openai import OpenAILLMProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "OpenAILLMProvider",
    "create_llm_provider",
    "create_llm_provider_from_schema",
]

# Optional import for TransformersLLMProvider
try:
    from .transformers import TransformersLLMProvider
    __all__.append("TransformersLLMProvider")
except ImportError:
    pass  # transformers not installed
