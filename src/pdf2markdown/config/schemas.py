"""Configuration schemas using Pydantic."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class CacheConfig(BaseModel):
    """Configuration for caching system."""

    enabled: bool = Field(default=True, description="Enable caching system")
    base_dir: Path = Field(
        default=Path("/tmp/pdf2markdown/cache"), description="Base cache directory"
    )
    max_size_gb: int = Field(default=10, ge=1, description="Maximum cache size in GB")
    cleanup_after_days: int = Field(
        default=7, ge=1, description="Clean up caches older than this many days"
    )
    resume_by_default: bool = Field(default=False, description="Resume processing by default")

    @field_validator("base_dir", mode="before")
    @classmethod
    def validate_cache_dir(cls, v):
        """Ensure base_dir is a Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class DocumentParserConfig(BaseModel):
    """Configuration for document parser."""

    type: str = "simple"
    resolution: int = Field(default=300, ge=72, le=600)
    max_dimension: int | None = Field(
        default=None,
        ge=100,
        le=10000,
        description="Maximum pixels for longest side of rendered image",
    )
    cache_dir: Path = Field(default=Path("/tmp/pdf2markdown/cache"))
    max_page_size: int = Field(default=50_000_000)  # 50MB
    timeout: int = Field(default=30)
    use_cache: bool = Field(default=True, description="Use caching for rendered images")

    @field_validator("cache_dir", mode="before")
    @classmethod
    def validate_cache_dir(cls, v):
        """Ensure cache_dir is a Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class LLMProviderConfig(BaseModel):
    """Configuration for LLM provider."""

    provider_type: str = Field(default="openai")

    # OpenAI-specific fields
    endpoint: str = Field(default="https://api.openai.com/v1")
    api_key: str | None = Field(default=None)
    model: str = Field(default="gpt-4o-mini")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.1, ge=0, le=2)
    timeout: int = Field(default=60)
    # Penalty parameters to reduce repetition
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float | None = Field(
        default=None, ge=0.0, le=2.0
    )  # Some providers use this instead

    # Transformers-specific fields
    model_name: str | None = Field(default=None)  # HuggingFace model name/path
    device: str = Field(default="auto")  # Device to run on
    torch_dtype: str = Field(default="auto")  # Data type for model
    trust_remote_code: bool = Field(default=True)  # Trust remote code
    attn_implementation: str = Field(default="sdpa")  # Attention implementation
    do_sample: bool = Field(default=False)  # Whether to use sampling
    device_map: str | None = Field(default="auto")  # Device mapping strategy
    load_in_8bit: bool = Field(default=False)  # Load model in 8-bit mode
    load_in_4bit: bool = Field(default=False)  # Load model in 4-bit mode
    cache_dir: str | None = Field(default=None)  # Directory to cache models
    model_type: str = Field(default="auto")  # Model type hint
    use_chat_method: bool = Field(default=False)  # Whether model has .chat() method
    processor_type: str = Field(default="auto")  # Type of processor to use
    # Vision model pixel limits (for memory management)
    max_pixels: int = Field(default=3145728)  # Maximum pixels (default: 2048x1536)
    min_pixels: int = Field(default=40000)  # Minimum pixels (default: 200x200)
    # Model context window limit (to prevent memory issues)
    max_length: int = Field(
        default=8192
    )  # Maximum context length (lower than model's 32k limit for memory)

    @model_validator(mode="after")
    def validate_provider_fields(self) -> "LLMProviderConfig":
        """Validate that required fields are present based on provider type."""
        if self.provider_type == "openai":
            # Only require api_key if we're actually going to use the OpenAI provider
            # Allow None or empty string during config loading
            pass
        elif self.provider_type == "transformers":
            # For transformers, if model_name is not set but model is, use model
            if not self.model_name and self.model:
                self.model_name = self.model
            if not self.model_name:
                raise ValueError("model_name or model is required for Transformers provider")
        return self


class MarkdownValidatorConfig(BaseModel):
    """Configuration for markdown validator."""

    enabled: bool = Field(default=True)
    attempt_correction: bool = Field(default=True)
    strict_mode: bool = Field(default=False)
    max_line_length: int = Field(default=1000, ge=80)
    disabled_rules: list[str] = Field(default_factory=list)
    enabled_rules: list[str] = Field(default_factory=list)


class RepetitionValidatorConfig(BaseModel):
    """Configuration for repetition validator."""

    enabled: bool = Field(default=True)
    attempt_correction: bool = Field(default=True)
    consecutive_threshold: int = Field(default=3, ge=2)
    window_size: int = Field(default=10, ge=5)
    window_threshold: int = Field(default=3, ge=2)
    check_exact_lines: bool = Field(default=True)
    check_normalized_lines: bool = Field(default=True)
    check_paragraphs: bool = Field(default=True)
    check_patterns: bool = Field(default=True)
    min_pattern_length: int = Field(default=20, ge=10)
    pattern_similarity_threshold: float = Field(default=0.9, ge=0.5, le=1.0)
    min_line_length: int = Field(default=5, ge=1)


class ValidationConfig(BaseModel):
    """Configuration for the validation pipeline."""

    validators: list[str] = Field(default_factory=lambda: ["markdown", "repetition"])
    markdown: MarkdownValidatorConfig = Field(default_factory=MarkdownValidatorConfig)
    repetition: RepetitionValidatorConfig = Field(default_factory=RepetitionValidatorConfig)
    max_correction_attempts: int = Field(default=2, ge=1, le=5)


class PageParserConfig(BaseModel):
    """Configuration for page parser."""

    type: str = "simple_llm"
    # Parser-specific fields
    prompt_template: Path | None = Field(default=None)
    additional_instructions: str | None = None

    # LLM provider configuration (can be specified directly or through llm_provider)
    llm_provider: LLMProviderConfig | None = None

    # Table format configuration
    table_format: str = Field(
        default="html",
        pattern="^(html|markdown)$",
        description="Format for table extraction: 'html' for complex layouts, 'markdown' for simple tables",
    )

    # Content validation configuration
    validate_content: bool = Field(default=True)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    # Caching configuration
    use_cache: bool = Field(default=True, description="Use caching for LLM-generated markdown")

    # Legacy validation fields (for backward compatibility)
    validate_markdown: bool | None = Field(default=None)
    markdown_validator: MarkdownValidatorConfig | None = Field(default=None)

    @field_validator("prompt_template", mode="before")
    @classmethod
    def validate_prompt_template(cls, v):
        """Ensure prompt_template is a Path object if provided."""
        if v and isinstance(v, str):
            return Path(v)
        return v

    def __init__(self, **data):
        """Initialize PageParserConfig with backward compatibility for LLM fields."""
        # Extract legacy LLM fields if present
        llm_fields = ["api_key", "endpoint", "model", "temperature", "max_tokens"]
        llm_data = {}
        for field in llm_fields:
            if field in data:
                llm_data[field] = data.pop(field)

        # If we have legacy fields and no llm_provider, create one
        if llm_data and "llm_provider" not in data:
            data["llm_provider"] = LLMProviderConfig(**llm_data)

        # Initialize the model
        super().__init__(**data)

    @model_validator(mode="after")
    def handle_legacy_fields(self) -> "PageParserConfig":
        """Handle legacy fields for backward compatibility."""
        # Handle legacy validation fields
        if self.validate_markdown is not None or self.markdown_validator is not None:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "Using deprecated validation configuration. "
                "Please update to use 'validation' field instead."
            )

            # Convert legacy config to new format
            if self.validate_markdown is False:
                self.validate_content = False
            elif self.markdown_validator is not None:
                self.validation.markdown = self.markdown_validator
                self.validation.validators = ["markdown"]

            # Clear deprecated fields
            self.validate_markdown = None
            self.markdown_validator = None

        return self

    @property
    def api_key(self) -> str | None:
        """Backward compatibility property for api_key."""
        return self.llm_provider.api_key if self.llm_provider else None

    @property
    def endpoint(self) -> str | None:
        """Backward compatibility property for endpoint."""
        return self.llm_provider.endpoint if self.llm_provider else None

    @property
    def model(self) -> str | None:
        """Backward compatibility property for model."""
        return self.llm_provider.model if self.llm_provider else None

    @property
    def temperature(self) -> float | None:
        """Backward compatibility property for temperature."""
        return self.llm_provider.temperature if self.llm_provider else None

    @property
    def max_tokens(self) -> int | None:
        """Backward compatibility property for max_tokens."""
        return self.llm_provider.max_tokens if self.llm_provider else None


class QueueConfig(BaseModel):
    """Configuration for queue sizes."""

    document_queue_size: int = Field(default=100, ge=1)
    page_queue_size: int = Field(default=1000, ge=1)
    output_queue_size: int = Field(default=500, ge=1)


class PipelineConfig(BaseModel):
    """Configuration for pipeline processing."""

    document_workers: int = Field(default=1, ge=1, le=1)  # Must be 1
    page_workers: int = Field(default=10, ge=1)
    queues: QueueConfig = Field(default_factory=QueueConfig)
    enable_progress: bool = True
    log_level: str = "INFO"

    @field_validator("document_workers")
    @classmethod
    def validate_document_workers(cls, v):
        """Ensure only 1 document worker as per requirement."""
        if v != 1:
            raise ValueError("Document workers must be exactly 1 (sequential processing required)")
        return v


class AppConfig(BaseModel):
    """Main application configuration."""

    llm_provider: LLMProviderConfig | None = None
    document_parser: DocumentParserConfig = Field(default_factory=DocumentParserConfig)
    page_parser: PageParserConfig = Field(default_factory=PageParserConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    output_dir: Path = Field(default=Path("./output"))
    temp_dir: Path = Field(default=Path("/tmp/pdf2markdown"))
    page_separator: str = Field(default="\n\n--[PAGE: {page_number}]--\n\n")

    @field_validator("output_dir", "temp_dir", mode="before")
    @classmethod
    def validate_paths(cls, v):
        """Ensure paths are Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    def model_dump_for_file(self) -> dict[str, Any]:
        """Export configuration for saving to file."""
        data = self.model_dump()

        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj

        return convert_paths(data)
