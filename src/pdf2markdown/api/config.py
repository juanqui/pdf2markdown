"""Configuration builder and management for the library API."""

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError
from .types import ConfigDict


class Config:
    """Configuration container for pdf2markdown."""

    def __init__(self, config_dict: ConfigDict):
        """Initialize configuration from dictionary."""
        self._config = self._resolve_environment_variables(config_dict)
        self._validate()

    @classmethod
    def builder(cls) -> "ConfigBuilder":
        """Create a new configuration builder."""
        return ConfigBuilder()

    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> "Config":
        """Create configuration from a dictionary."""
        return cls(config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise ConfigurationError(f"Configuration file not found: {yaml_path}")

        try:
            with open(yaml_path) as f:
                config_dict = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {yaml_path}: {e}") from e

        return cls(config_dict)

    @classmethod
    def default(cls) -> "Config":
        """Create configuration with default values."""
        return ConfigBuilder().build()

    def _resolve_environment_variables(self, config: ConfigDict) -> ConfigDict:
        """Recursively resolve environment variables in configuration."""
        if isinstance(config, dict):
            return {k: self._resolve_environment_variables(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_environment_variables(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            value = os.environ.get(env_var)
            if value is None:
                raise ConfigurationError(f"Environment variable {env_var} is not set")
            return value
        else:
            return config

    def _validate(self):
        """Validate the configuration."""
        # Check for required fields when llm_provider is configured
        if "llm_provider" in self._config:
            provider = self._config["llm_provider"]
            if "provider_type" not in provider:
                raise ConfigurationError("llm_provider.provider_type is required")

            if provider["provider_type"] == "openai":
                if "api_key" not in provider:
                    raise ConfigurationError("llm_provider.api_key is required for OpenAI provider")
            elif provider["provider_type"] == "transformers":
                if "model_name" not in provider:
                    raise ConfigurationError(
                        "llm_provider.model_name is required for Transformers provider"
                    )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (supports dot notation)."""
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def to_dict(self) -> ConfigDict:
        """Get the configuration as a dictionary."""
        return copy.deepcopy(self._config)

    @property
    def llm_provider(self) -> ConfigDict | None:
        """Get LLM provider configuration."""
        return self._config.get("llm_provider")

    @property
    def document_parser(self) -> ConfigDict | None:
        """Get document parser configuration."""
        return self._config.get("document_parser")

    @property
    def page_parser(self) -> ConfigDict | None:
        """Get page parser configuration."""
        return self._config.get("page_parser")

    @property
    def pipeline(self) -> ConfigDict | None:
        """Get pipeline configuration."""
        return self._config.get("pipeline")


class ConfigBuilder:
    """Builder pattern for creating configurations programmatically."""

    def __init__(self):
        """Initialize the configuration builder with defaults."""
        self._config: ConfigDict = {
            "document_parser": {
                "type": "simple",
                "resolution": 300,
                "cache_dir": "/tmp/pdf2markdown/cache",
                "max_page_size": 50_000_000,
                "timeout": 30,
            },
            "page_parser": {
                "type": "simple_llm",
                "validate_content": True,
                "validation": {
                    "validators": ["markdown", "repetition"],
                    "max_correction_attempts": 2,
                },
            },
            "pipeline": {
                "document_workers": 1,
                "page_workers": 10,
                "queues": {
                    "document_queue_size": 100,
                    "page_queue_size": 1000,
                    "output_queue_size": 500,
                },
                "enable_progress": False,  # Disabled by default for library usage
                "log_level": "WARNING",  # Less verbose for library usage
            },
            "output_dir": "./output",
            "temp_dir": "/tmp/pdf2markdown",
        }

    def with_llm_provider(self, provider_type: str, **kwargs) -> "ConfigBuilder":
        """Configure LLM provider."""
        if "llm_provider" not in self._config:
            self._config["llm_provider"] = {}

        self._config["llm_provider"]["provider_type"] = provider_type
        self._config["llm_provider"].update(kwargs)
        return self

    def with_openai(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        endpoint: str = "https://api.openai.com/v1",
        **kwargs,
    ) -> "ConfigBuilder":
        """Convenience method for OpenAI configuration."""
        return self.with_llm_provider(
            "openai",
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.1),
            timeout=kwargs.get("timeout", 60),
            **{
                k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature", "timeout"]
            },
        )

    def with_transformers(self, model_name: str, device: str = "auto", **kwargs) -> "ConfigBuilder":
        """Convenience method for Transformers configuration."""
        return self.with_llm_provider(
            "transformers",
            model_name=model_name,
            device=device,
            torch_dtype=kwargs.get("torch_dtype", "auto"),
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.1),
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["torch_dtype", "max_tokens", "temperature"]
            },
        )

    def with_resolution(self, dpi: int) -> "ConfigBuilder":
        """Set PDF rendering resolution."""
        self._config["document_parser"]["resolution"] = dpi
        return self

    def with_page_workers(self, workers: int) -> "ConfigBuilder":
        """Set number of parallel page workers."""
        self._config["pipeline"]["page_workers"] = workers
        return self

    def with_validators(self, validators: list[str]) -> "ConfigBuilder":
        """Configure validation pipeline."""
        if "page_parser" not in self._config:
            self._config["page_parser"] = {}
        if "validation" not in self._config["page_parser"]:
            self._config["page_parser"]["validation"] = {}

        self._config["page_parser"]["validation"]["validators"] = validators
        return self

    def with_cache_dir(self, cache_dir: str | Path) -> "ConfigBuilder":
        """Set cache directory for rendered images."""
        self._config["document_parser"]["cache_dir"] = str(cache_dir)
        return self

    def with_output_dir(self, output_dir: str | Path) -> "ConfigBuilder":
        """Set default output directory."""
        self._config["output_dir"] = str(output_dir)
        return self

    def with_progress(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable or disable progress tracking."""
        self._config["pipeline"]["enable_progress"] = enabled
        return self

    def with_log_level(self, level: str) -> "ConfigBuilder":
        """Set logging level."""
        self._config["pipeline"]["log_level"] = level
        return self

    def with_page_separator(self, separator: str) -> "ConfigBuilder":
        """Set page separator for markdown output."""
        self._config["page_separator"] = separator
        return self

    def with_validation_correction(
        self, enabled: bool = True, max_attempts: int = 2
    ) -> "ConfigBuilder":
        """Configure validation correction settings."""
        if "page_parser" not in self._config:
            self._config["page_parser"] = {}
        if "validation" not in self._config["page_parser"]:
            self._config["page_parser"]["validation"] = {}

        validation = self._config["page_parser"]["validation"]
        validation["max_correction_attempts"] = max_attempts if enabled else 0

        # Configure individual validators
        for validator in ["markdown", "repetition"]:
            if validator not in validation:
                validation[validator] = {}
            validation[validator]["attempt_correction"] = enabled

        return self

    def merge(self, config_dict: ConfigDict) -> "ConfigBuilder":
        """Merge additional configuration."""
        self._deep_merge(self._config, config_dict)
        return self

    def _deep_merge(self, base: dict, update: dict) -> dict:
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def build(self) -> Config:
        """Build and validate configuration."""
        return Config(self._config)
