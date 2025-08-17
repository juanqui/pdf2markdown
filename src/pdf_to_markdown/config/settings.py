"""Configuration management for PDF to Markdown converter."""

import logging
import os
import re
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

    def _expand_env_vars(self, data: Any) -> Any:
        """Recursively expand environment variables in configuration data.
        
        Supports ${VAR_NAME} syntax for environment variable substitution.
        
        Args:
            data: Configuration data (dict, list, or scalar)
            
        Returns:
            Data with environment variables expanded
        """
        if isinstance(data, dict):
            return {k: self._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Replace ${VAR_NAME} with environment variable value
            def replacer(match):
                var_name = match.group(1)
                value = os.getenv(var_name)
                if value is None:
                    logger.warning(f"Environment variable ${{{var_name}}} not found")
                    return match.group(0)  # Keep original if not found
                return value
            
            return re.sub(r'\$\{([^}]+)\}', replacer, data)
        else:
            return data

    def _load_config(self) -> AppConfig:
        """Load configuration from file or use defaults.

        Returns:
            AppConfig instance
        """
        if self.config_path and self.config_path.exists():
            with open(self.config_path) as f:
                data = yaml.safe_load(f)
                
                # Expand environment variables in the configuration
                data = self._expand_env_vars(data)

                # Ensure llm_provider exists
                if "llm_provider" not in data:
                    data["llm_provider"] = {}
                
                # Only set API key from env if provider type is openai (or not specified)
                # and if api_key is not already set (from YAML with env var expansion)
                provider_type = data["llm_provider"].get("provider_type", "openai")
                if provider_type == "openai" and "api_key" not in data["llm_provider"]:
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        data["llm_provider"]["api_key"] = api_key

                return AppConfig(**data)
        else:
            # Use defaults with API key from environment
            return self._get_default_config()

    def _get_default_config(self) -> AppConfig:
        """Get default configuration.

        Returns:
            Default AppConfig instance
        """
        # For default config, we'll use OpenAI provider
        # But we won't require the API key to be set immediately
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Create config without api_key if not set
        llm_config_args = {}
        if api_key:
            llm_config_args["api_key"] = api_key
        
        return AppConfig(
            llm_provider=LLMProviderConfig(**llm_config_args),
            document_parser=DocumentParserConfig(),
            page_parser=PageParserConfig(),
            pipeline=PipelineConfig(),
        )

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Ensure llm_provider exists
        if not self.config.llm_provider:
            self.config.llm_provider = LLMProviderConfig()

        # Override API key if provided and using OpenAI provider
        if self.config.llm_provider.provider_type == "openai":
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
        # Check provider-specific requirements
        if self.config.llm_provider:
            if self.config.llm_provider.provider_type == "openai":
                if not self.config.llm_provider.api_key:
                    logger.error("API key is not configured for OpenAI provider")
                    return False
            elif self.config.llm_provider.provider_type == "transformers":
                # Transformers provider doesn't need an API key
                # Check for model or model_name
                if not (self.config.llm_provider.model or self.config.llm_provider.model_name):
                    logger.error("Model name is not configured for Transformers provider")
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
