"""Configuration schemas using Pydantic."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentParserConfig(BaseModel):
    """Configuration for document parser."""

    type: str = "simple"
    resolution: int = Field(default=300, ge=72, le=600)
    cache_dir: Path = Field(default=Path("/tmp/pdf_to_markdown/cache"))
    max_page_size: int = Field(default=50_000_000)  # 50MB
    timeout: int = Field(default=30)

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
    endpoint: str = Field(default="https://api.openai.com/v1")
    api_key: str
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


class MarkdownValidatorConfig(BaseModel):
    """Configuration for markdown validator."""

    enabled: bool = Field(default=True)
    attempt_correction: bool = Field(default=True)
    strict_mode: bool = Field(default=False)
    max_line_length: int = Field(default=1000, ge=80)
    disabled_rules: list[str] = Field(default_factory=list)
    enabled_rules: list[str] = Field(default_factory=list)


class PageParserConfig(BaseModel):
    """Configuration for page parser."""

    type: str = "simple_llm"
    # Parser-specific fields
    prompt_template: Path | None = Field(default=None)
    additional_instructions: str | None = None
    # Markdown validation
    validate_markdown: bool = Field(default=True)
    markdown_validator: MarkdownValidatorConfig = Field(default_factory=MarkdownValidatorConfig)

    @field_validator("prompt_template", mode="before")
    @classmethod
    def validate_prompt_template(cls, v):
        """Ensure prompt_template is a Path object if provided."""
        if v and isinstance(v, str):
            return Path(v)
        return v


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
    output_dir: Path = Field(default=Path("./output"))
    temp_dir: Path = Field(default=Path("/tmp/pdf_to_markdown"))

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
