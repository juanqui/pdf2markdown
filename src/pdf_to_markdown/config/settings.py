"""Configuration management for PDF to Markdown converter."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .schemas import (
    AppConfig,
    DocumentParserConfig,
    LLMProviderConfig,
    PageParserConfig,
    PipelineConfig,
)

logger = logging.getLogger(__name__)


class Settings:
    """Manages application settings and configuration."""

    def __init__(self, config_path: Path | None = None, env_file: Path | None = None):
        """Initialize settings.

        Args:
            config_path: Path to configuration file
            env_file: Path to .env file
        """
        # Load environment variables
        if env_file and env_file.exists():
            load_dotenv(env_file)
        else:
            load_dotenv()  # Load from default .env

        # Load configuration
        self.config_path = config_path
        self.config = self._load_config()

        # Apply environment variable overrides
        self._apply_env_overrides()

        logger.info(f"Settings initialized from {config_path or 'defaults'}")

    def _load_config(self) -> AppConfig:
        """Load configuration from file or use defaults.

        Returns:
            AppConfig instance
        """
        if self.config_path and self.config_path.exists():
            with open(self.config_path) as f:
                data = yaml.safe_load(f)

                # Ensure API key is present in llm_provider
                if "llm_provider" not in data:
                    data["llm_provider"] = {}
                if "api_key" not in data["llm_provider"]:
                    data["llm_provider"]["api_key"] = os.getenv("OPENAI_API_KEY", "")

                return AppConfig(**data)
        else:
            # Use defaults with API key from environment
            return self._get_default_config()

    def _get_default_config(self) -> AppConfig:
        """Get default configuration.

        Returns:
            Default AppConfig instance
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        return AppConfig(
            llm_provider=LLMProviderConfig(api_key=api_key),
            document_parser=DocumentParserConfig(),
            page_parser=PageParserConfig(),
            pipeline=PipelineConfig(),
        )

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Ensure llm_provider exists
        if not self.config.llm_provider:
            self.config.llm_provider = LLMProviderConfig(api_key="")

        # Override API key if provided
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.config.llm_provider.api_key = api_key

        # Override endpoint if provided
        endpoint = os.getenv("OPENAI_API_ENDPOINT")
        if endpoint:
            self.config.llm_provider.endpoint = endpoint

        # Override model if provided
        model = os.getenv("OPENAI_MODEL")
        if model:
            self.config.llm_provider.model = model

        # Override cache directory if provided
        cache_dir = os.getenv("PDF_TO_MARKDOWN_CACHE_DIR")
        if cache_dir:
            self.config.document_parser.cache_dir = Path(cache_dir)

        # Override output directory if provided
        output_dir = os.getenv("PDF_TO_MARKDOWN_OUTPUT_DIR")
        if output_dir:
            self.config.output_dir = Path(output_dir)

        # Override log level if provided
        log_level = os.getenv("PDF_TO_MARKDOWN_LOG_LEVEL")
        if log_level:
            self.config.pipeline.log_level = log_level

    def save(self, path: Path | None = None) -> None:
        """Save configuration to file.

        Args:
            path: Path to save configuration to
        """
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No path provided for saving configuration")

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and save
        data = self.config.model_dump_for_file()

        # Remove sensitive data
        if "llm_provider" in data and "api_key" in data["llm_provider"]:
            data["llm_provider"]["api_key"] = "YOUR_API_KEY_HERE"

        with open(save_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Configuration saved to {save_path}")

    def get_document_parser_config(self) -> dict[str, Any]:
        """Get document parser configuration.

        Returns:
            Configuration dictionary for document parser
        """
        return self.config.document_parser.model_dump()

    def get_page_parser_config(self) -> dict[str, Any]:
        """Get page parser configuration.

        Returns:
            Configuration dictionary for page parser
        """
        return self.config.page_parser.model_dump()

    def get_pipeline_config(self) -> dict[str, Any]:
        """Get pipeline configuration.

        Returns:
            Configuration dictionary for pipeline
        """
        config = self.config.pipeline.model_dump()
        config["document_parser"] = self.get_document_parser_config()
        config["page_parser"] = self.get_page_parser_config()
        config["llm_provider"] = (
            self.config.llm_provider.model_dump() if self.config.llm_provider else None
        )
        config["output_dir"] = str(self.config.output_dir)
        return config

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if configuration is valid
        """
        # Check API key from top-level llm_provider
        if not self.config.llm_provider or not self.config.llm_provider.api_key:
            logger.error("API key is not configured in llm_provider")
            return False

        # Check directories exist or can be created
        try:
            self.config.document_parser.cache_dir.mkdir(parents=True, exist_ok=True)
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False

        return True


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from file or environment.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Settings instance
    """
    # Look for config file in standard locations if not provided
    if not config_path:
        standard_paths = [
            Path("config.yaml"),
            Path("config/default.yaml"),
            Path.home() / ".pdf_to_markdown" / "config.yaml",
        ]
        for path in standard_paths:
            if path.exists():
                config_path = path
                break

    return Settings(config_path)
