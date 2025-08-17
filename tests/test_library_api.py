"""Tests for the library API."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pdf2markdown import (
    Config,
    ConfigBuilder,
    ConfigurationError,
    ConversionStatus,
    DocumentResult,
    LLMError,
    PageResult,
    ParsingError,
    PDFConverter,
)


class TestConfigBuilder:
    """Test the ConfigBuilder class."""

    def test_default_config(self):
        """Test creating default configuration."""
        config = ConfigBuilder().build()
        assert isinstance(config, Config)
        assert config.document_parser is not None
        assert config.page_parser is not None
        assert config.pipeline is not None

    def test_with_openai(self):
        """Test configuring OpenAI provider."""
        config = ConfigBuilder().with_openai(api_key="test-key", model="gpt-4o").build()

        assert config.llm_provider is not None
        assert config.llm_provider["provider_type"] == "openai"
        assert config.llm_provider["api_key"] == "test-key"
        assert config.llm_provider["model"] == "gpt-4o"

    def test_with_transformers(self):
        """Test configuring Transformers provider."""
        config = ConfigBuilder().with_transformers(model_name="test-model", device="cpu").build()

        assert config.llm_provider is not None
        assert config.llm_provider["provider_type"] == "transformers"
        assert config.llm_provider["model_name"] == "test-model"
        assert config.llm_provider["device"] == "cpu"

    def test_with_resolution(self):
        """Test setting resolution."""
        config = ConfigBuilder().with_resolution(400).build()

        assert config.document_parser["resolution"] == 400

    def test_with_page_workers(self):
        """Test setting page workers."""
        config = ConfigBuilder().with_page_workers(20).build()

        assert config.pipeline["page_workers"] == 20

    def test_with_validators(self):
        """Test configuring validators."""
        config = ConfigBuilder().with_validators(["markdown"]).build()

        assert config.page_parser["validation"]["validators"] == ["markdown"]

    def test_chaining(self):
        """Test method chaining."""
        config = (
            ConfigBuilder()
            .with_openai(api_key="test-key")
            .with_resolution(400)
            .with_page_workers(15)
            .with_cache_dir("/tmp/test")
            .with_output_dir("./test_output")
            .with_progress(False)
            .with_log_level("DEBUG")
            .build()
        )

        assert config.llm_provider["api_key"] == "test-key"
        assert config.document_parser["resolution"] == 400
        assert config.pipeline["page_workers"] == 15
        assert config.document_parser["cache_dir"] == "/tmp/test"
        assert config.get("output_dir") == "./test_output"
        assert config.pipeline["enable_progress"] is False
        assert config.pipeline["log_level"] == "DEBUG"


class TestConfig:
    """Test the Config class."""

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {"llm_provider": {"provider_type": "openai", "api_key": "test-key"}}
        config = Config.from_dict(config_dict)
        assert config.llm_provider["api_key"] == "test-key"

    def test_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        import yaml

        config_file = tmp_path / "test_config.yaml"
        config_dict = {
            "llm_provider": {
                "provider_type": "openai",
                "api_key": "test-key",
                "model": "gpt-4o-mini",
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = Config.from_yaml(config_file)
        assert config.llm_provider["model"] == "gpt-4o-mini"

    def test_environment_variable_resolution(self, monkeypatch):
        """Test resolving environment variables in config."""
        monkeypatch.setenv("TEST_API_KEY", "secret-key")

        config_dict = {"llm_provider": {"provider_type": "openai", "api_key": "${TEST_API_KEY}"}}

        config = Config.from_dict(config_dict)
        assert config.llm_provider["api_key"] == "secret-key"

    def test_get_with_dot_notation(self):
        """Test getting config values with dot notation."""
        config_dict = {
            "llm_provider": {
                "provider_type": "openai",  # Add required field
                "api_key": "test-key",  # Add required API key
                "model": "test-model",
                "settings": {"temperature": 0.7},
            }
        }

        config = Config.from_dict(config_dict)
        assert config.get("llm_provider.model") == "test-model"
        assert config.get("llm_provider.settings.temperature") == 0.7
        assert config.get("non.existent.key", "default") == "default"


class TestPDFConverter:
    """Test the PDFConverter class."""

    @pytest.fixture
    def mock_converter(self):
        """Create a mock converter with patched dependencies."""
        with (
            patch("pdf2markdown.api.converter.create_llm_provider") as mock_create_llm,
            patch("pdf2markdown.api.converter.SimpleDocumentParser") as mock_doc_parser,
            patch("pdf2markdown.api.converter.SimpleLLMPageParser") as mock_page_parser,
            patch("pdf2markdown.api.converter.PipelineCoordinator") as mock_pipeline,
        ):

            # Configure mocks
            mock_create_llm.return_value = Mock()
            mock_doc_parser.return_value = Mock()
            mock_page_parser.return_value = Mock()
            mock_pipeline_instance = AsyncMock()
            mock_pipeline.return_value = mock_pipeline_instance

            # Create converter
            config = ConfigBuilder().with_openai(api_key="test-key").build()
            converter = PDFConverter(config=config)

            yield converter, mock_pipeline_instance

    def test_init_with_dict(self):
        """Test initializing converter with dictionary config."""
        config_dict = {"llm_provider": {"provider_type": "openai", "api_key": "test-key"}}
        converter = PDFConverter(config=config_dict)
        assert converter.config.llm_provider["api_key"] == "test-key"

    def test_init_with_config_object(self):
        """Test initializing converter with Config object."""
        config = ConfigBuilder().with_openai(api_key="test-key").build()
        converter = PDFConverter(config=config)
        assert converter.config.llm_provider["api_key"] == "test-key"

    def test_init_with_defaults(self):
        """Test initializing converter with default config."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            converter = PDFConverter()
            assert converter.config is not None

    @pytest.mark.asyncio
    async def test_convert_async(self, tmp_path):
        """Test async conversion."""
        # Create a test PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")

        # Create mock converter
        with patch("pdf2markdown.api.converter.PipelineCoordinator") as mock_pipeline:

            # Configure pipeline mock
            mock_pipeline_instance = AsyncMock()
            mock_pipeline.return_value = mock_pipeline_instance

            # Create mock document
            from datetime import datetime

            from pdf2markdown.core.models import Document, Page, ProcessingStatus

            mock_doc = Document(
                id="test-doc",
                source_path=pdf_path,
                status=ProcessingStatus.COMPLETED,
                created_at=datetime.now(),
            )
            mock_page = Page(
                id="test-page",
                document_id="test-doc",
                page_number=1,
                markdown_content="# Test Content",
                status=ProcessingStatus.COMPLETED,
            )
            mock_doc.pages = [mock_page]

            mock_pipeline_instance.process.return_value = mock_doc

            # Create converter and convert
            config = ConfigBuilder().with_openai(api_key="test-key").build()
            converter = PDFConverter(config=config)

            result = await converter.convert(pdf_path)

            assert result == "# Test Content"
            mock_pipeline_instance.process.assert_called_once()

    def test_convert_sync(self, tmp_path):
        """Test synchronous conversion."""
        # Create a test PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")

        with patch.object(PDFConverter, "convert") as mock_convert:
            # Create a coroutine that returns the content
            async def async_convert(*args, **kwargs):
                return "# Test Content"

            mock_convert.return_value = async_convert()

            converter = PDFConverter()
            result = converter.convert_sync(pdf_path)

            # The test should complete without errors
            assert result is not None

    def test_convert_file_not_found(self):
        """Test conversion with non-existent file."""
        converter = PDFConverter()

        with pytest.raises(ParsingError, match="PDF file not found"):
            asyncio.run(converter.convert("non_existent.pdf"))

    @pytest.mark.asyncio
    async def test_stream_pages(self, tmp_path):
        """Test streaming pages."""
        # Create a test PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")

        converter = PDFConverter()

        with (
            patch.object(converter, "_ensure_initialized"),
            patch.object(converter, "_pipeline") as mock_pipeline,
        ):

            # Mock pipeline methods
            mock_pipeline.start_processing = AsyncMock(return_value="doc-id")
            mock_pipeline.get_status = AsyncMock(
                side_effect=[
                    {"status": "processing", "total_pages": 2, "pages": {1: "completed"}},
                    {
                        "status": "completed",
                        "total_pages": 2,
                        "pages": {1: "completed", 2: "completed"},
                    },
                ]
            )

            from pdf2markdown.core.models import Page, ProcessingStatus

            mock_page1 = Page(
                id="page1",
                document_id="doc-id",
                page_number=1,
                markdown_content="Page 1",
                status=ProcessingStatus.COMPLETED,
            )
            mock_page2 = Page(
                id="page2",
                document_id="doc-id",
                page_number=2,
                markdown_content="Page 2",
                status=ProcessingStatus.COMPLETED,
            )
            mock_pipeline.get_page_result = AsyncMock(side_effect=[mock_page1, mock_page2])

            pages = []
            async for page in converter.stream_pages(pdf_path):
                pages.append(page)

            assert len(pages) == 2
            assert pages[0].page_number == 1
            assert pages[0].content == "Page 1"

    @pytest.mark.asyncio
    async def test_process_batch(self, tmp_path):
        """Test batch processing."""
        # Create test PDF files
        pdf1 = tmp_path / "test1.pdf"
        pdf2 = tmp_path / "test2.pdf"
        pdf1.write_bytes(b"fake pdf 1")
        pdf2.write_bytes(b"fake pdf 2")

        converter = PDFConverter()

        # Create a mock that properly tracks calls
        call_count = 0

        async def mock_convert(pdf_path, output_path=None, progress_callback=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "# Content 1"
            else:
                return "# Content 2"

        # Use patch on the instance method
        with patch.object(converter, "convert", side_effect=mock_convert):
            results = await converter.process_batch([pdf1, pdf2])

            assert len(results) == 2
            assert results[0].status == ConversionStatus.COMPLETED
            assert results[1].status == ConversionStatus.COMPLETED
            # Check that both contents are present
            assert results[0].markdown_content == "# Content 1"
            assert results[1].markdown_content == "# Content 2"


class TestDocumentResult:
    """Test the DocumentResult class."""

    def test_to_markdown_default_separator(self):
        """Test converting to markdown with default separator."""
        pages = [
            PageResult(page_number=1, content="Page 1", status=ConversionStatus.COMPLETED),
            PageResult(page_number=2, content="Page 2", status=ConversionStatus.COMPLETED),
        ]

        result = DocumentResult(
            source_path=Path("test.pdf"),
            pages=pages,
            total_pages=2,
            status=ConversionStatus.COMPLETED,
        )

        markdown = result.to_markdown()
        assert "Page 1" in markdown
        assert "--[PAGE: 2]--" in markdown
        assert "Page 2" in markdown

    def test_to_markdown_custom_separator(self):
        """Test converting to markdown with custom separator."""
        pages = [
            PageResult(page_number=1, content="Page 1", status=ConversionStatus.COMPLETED),
            PageResult(page_number=2, content="Page 2", status=ConversionStatus.COMPLETED),
        ]

        result = DocumentResult(
            source_path=Path("test.pdf"),
            pages=pages,
            total_pages=2,
            status=ConversionStatus.COMPLETED,
        )

        markdown = result.to_markdown(page_separator="\n---\n")
        assert "Page 1\n---\nPage 2" in markdown

    def test_save(self, tmp_path):
        """Test saving markdown to file."""
        pages = [
            PageResult(page_number=1, content="# Test", status=ConversionStatus.COMPLETED),
        ]

        result = DocumentResult(
            source_path=Path("test.pdf"),
            pages=pages,
            total_pages=1,
            status=ConversionStatus.COMPLETED,
        )

        output_path = tmp_path / "output.md"
        result.save(output_path)

        assert output_path.exists()
        assert output_path.read_text() == "# Test"


class TestExceptions:
    """Test custom exceptions."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config", details={"key": "value"})
        assert str(error) == "Invalid config"
        assert error.details == {"key": "value"}

    def test_parsing_error(self):
        """Test ParsingError."""
        error = ParsingError("Parse failed", page_number=5)
        assert str(error) == "Parse failed"
        assert error.page_number == 5

    def test_llm_error(self):
        """Test LLMError."""
        error = LLMError("API failed", provider="openai")
        assert str(error) == "API failed"
        assert error.provider == "openai"
