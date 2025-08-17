"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from pdf2markdown.config import (
    AppConfig,
    DocumentParserConfig,
    PageParserConfig,
    PipelineConfig,
    Settings,
)


class TestDocumentParserConfig:
    """Test DocumentParserConfig."""

    def test_defaults(self):
        """Test default values."""
        config = DocumentParserConfig()

        assert config.type == "simple"
        assert config.resolution == 300
        assert config.max_page_size == 50_000_000
        assert config.timeout == 30

    def test_validation(self):
        """Test configuration validation."""
        # Valid resolution
        config = DocumentParserConfig(resolution=150)
        assert config.resolution == 150

        # Invalid resolution (too low)
        with pytest.raises(ValueError):
            DocumentParserConfig(resolution=50)

        # Invalid resolution (too high)
        with pytest.raises(ValueError):
            DocumentParserConfig(resolution=700)


class TestPageParserConfig:
    """Test PageParserConfig."""

    def test_creation(self):
        """Test creating PageParserConfig."""
        # Test with backward compatibility (flat structure)
        config = PageParserConfig(api_key="test-key")
        assert config.api_key == "test-key"
        # These fields are now optional in the PageParserConfig itself
        # The defaults are created when the LLMProviderConfig is instantiated
        assert config.llm_provider is not None
        assert config.llm_provider.api_key == "test-key"
        assert config.llm_provider.model == "gpt-4o-mini"
        assert config.llm_provider.temperature == 0.1
        assert config.llm_provider.max_tokens == 4096

        # Test with explicit llm_provider
        from pdf2markdown.config.schemas import LLMProviderConfig

        provider_config = LLMProviderConfig(api_key="test-key-2")
        config2 = PageParserConfig(llm_provider=provider_config)
        assert config2.llm_provider.api_key == "test-key-2"
        assert config2.llm_provider.model == "gpt-4o-mini"

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperature - now goes through llm_provider
        config = PageParserConfig(api_key="test", temperature=0.5)
        assert config.llm_provider.temperature == 0.5

        # Temperature up to 2.0 is now valid for providers like OpenAI
        config2 = PageParserConfig(api_key="test", temperature=1.5)
        assert config2.llm_provider.temperature == 1.5

        # Invalid temperature (> 2.0)
        with pytest.raises(ValueError):
            PageParserConfig(api_key="test", temperature=2.5)


class TestPipelineConfig:
    """Test PipelineConfig."""

    def test_defaults(self):
        """Test default values."""
        config = PipelineConfig()

        assert config.document_workers == 1
        assert config.page_workers == 10
        assert config.enable_progress is True

    def test_document_workers_validation(self):
        """Test document workers must be 1."""
        # Valid
        config = PipelineConfig(document_workers=1)
        assert config.document_workers == 1

        # Invalid
        with pytest.raises(ValueError):
            PipelineConfig(document_workers=2)


class TestAppConfig:
    """Test AppConfig."""

    def test_creation(self):
        """Test creating AppConfig."""
        config = AppConfig(page_parser=PageParserConfig(api_key="test-key"))

        assert config.page_parser.api_key == "test-key"
        assert isinstance(config.document_parser, DocumentParserConfig)
        assert isinstance(config.pipeline, PipelineConfig)

    def test_model_dump_for_file(self):
        """Test exporting configuration for file."""
        config = AppConfig(page_parser=PageParserConfig(api_key="test-key"))

        data = config.model_dump_for_file()

        # Check paths are converted to strings
        assert isinstance(data["output_dir"], str)
        assert isinstance(data["temp_dir"], str)
        assert isinstance(data["document_parser"]["cache_dir"], str)


class TestSettings:
    """Test Settings class."""

    def test_load_from_env(self):
        """Test loading settings from environment."""
        os.environ["OPENAI_API_KEY"] = "env-test-key"

        settings = Settings()

        assert settings.config.page_parser.api_key == "env-test-key"

        # Cleanup
        del os.environ["OPENAI_API_KEY"]

    def test_load_from_file(self):
        """Test loading settings from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {"page_parser": {"api_key": "file-test-key", "model": "gpt-4o"}}
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            settings = Settings(config_path)

            assert settings.config.page_parser.api_key == "file-test-key"
            assert settings.config.page_parser.model == "gpt-4o"
        finally:
            config_path.unlink()

    def test_env_override(self):
        """Test environment variable overrides."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {"page_parser": {"api_key": "file-key", "model": "gpt-4o"}}
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        os.environ["OPENAI_API_KEY"] = "env-override-key"
        os.environ["OPENAI_MODEL"] = "gpt-4o-mini"

        try:
            settings = Settings(config_path)

            # Environment should override file
            assert settings.config.page_parser.api_key == "env-override-key"
            assert settings.config.page_parser.model == "gpt-4o-mini"
        finally:
            config_path.unlink()
            del os.environ["OPENAI_API_KEY"]
            del os.environ["OPENAI_MODEL"]

    def test_save_config(self):
        """Test saving configuration."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        settings = Settings()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            save_path = Path(f.name)

        try:
            settings.save(save_path)

            # Load saved config
            with open(save_path) as f:
                saved_data = yaml.safe_load(f)

            # Check API key is masked
            assert saved_data["page_parser"]["api_key"] == "YOUR_API_KEY_HERE"

            # Check other values are preserved
            assert saved_data["document_parser"]["resolution"] == 300
        finally:
            save_path.unlink()
            del os.environ["OPENAI_API_KEY"]
