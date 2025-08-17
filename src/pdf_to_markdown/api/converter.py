"""Main converter class for the pdf-to-markdown library API."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, Union, AsyncIterator, Dict, Any, List
from contextlib import asynccontextmanager

from ..pipeline.coordinator import PipelineCoordinator
from ..core.models import Document as InternalDocument, Page as InternalPage
from ..config.settings import Settings
from ..config.schemas import AppConfig

from .config import Config, ConfigBuilder
from .types import (
    DocumentResult, PageResult, ConversionStatus, 
    ProgressCallback, AsyncProgressCallback, ConfigDict
)
from .exceptions import PDFConversionError, ConfigurationError, ParsingError


logger = logging.getLogger(__name__)


class PDFConverter:
    """High-level API for converting PDF documents to Markdown."""
    
    def __init__(self, config: Optional[Union[Config, ConfigDict]] = None):
        """
        Initialize the PDF converter.
        
        Args:
            config: Configuration object, dictionary, or None for defaults
        """
        # Handle different config types
        if config is None:
            self.config = Config.default()
        elif isinstance(config, dict):
            self.config = Config.from_dict(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise ConfigurationError(f"Invalid configuration type: {type(config)}")
        
        # Set up logging
        log_level = self.config.get('pipeline.log_level', 'WARNING')
        logging.getLogger('pdf_to_markdown').setLevel(getattr(logging, log_level.upper()))
        
        # Initialize components lazily
        self._pipeline: Optional[PipelineCoordinator] = None
        
    async def _ensure_initialized(self):
        """Ensure all components are initialized."""
        if self._pipeline is None:
            await self._initialize_components()
    
    async def _initialize_components(self):
        """Initialize the converter components."""
        try:
            # Create internal settings from our config
            settings = self._create_settings()
            
            # Create pipeline configuration
            pipeline_config = settings.pipeline.model_dump() if settings.pipeline else {}
            
            # Add llm_provider to pipeline config
            if settings.llm_provider:
                pipeline_config['llm_provider'] = settings.llm_provider.model_dump()
            else:
                # Try to get from environment or use defaults
                pipeline_config['llm_provider'] = {
                    'provider_type': 'openai',
                    'api_key': os.environ.get('OPENAI_API_KEY'),
                    'model': 'gpt-4o-mini'
                }
            
            # Add document_parser config
            if settings.document_parser:
                pipeline_config['document_parser'] = settings.document_parser.model_dump()
            
            # Add page_parser config
            if settings.page_parser:
                pipeline_config['page_parser'] = settings.page_parser.model_dump()
            
            # Initialize pipeline with complete config
            self._pipeline = PipelineCoordinator(pipeline_config)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize converter: {e}")
    
    def _create_settings(self) -> Settings:
        """Create internal Settings object from our Config."""
        config_dict = self.config.to_dict()
        
        # Map our config structure to internal AppConfig structure
        app_config_dict = {}
        
        # Copy over the configurations
        for key in ['llm_provider', 'document_parser', 'page_parser', 'pipeline', 
                    'output_dir', 'temp_dir', 'page_separator']:
            if key in config_dict:
                app_config_dict[key] = config_dict[key]
        
        # Create AppConfig and Settings
        try:
            app_config = AppConfig(**app_config_dict)
            settings = Settings(config=app_config)
            return settings
        except Exception as e:
            # If validation fails, create minimal settings
            from ..config.schemas import (
                DocumentParserConfig, PageParserConfig, 
                PipelineConfig, LLMProviderConfig
            )
            
            # Build configs piece by piece
            settings = Settings()
            
            if 'llm_provider' in config_dict:
                settings.llm_provider = LLMProviderConfig(**config_dict['llm_provider'])
            
            if 'document_parser' in config_dict:
                settings.document_parser = DocumentParserConfig(**config_dict['document_parser'])
            
            if 'page_parser' in config_dict:
                settings.page_parser = PageParserConfig(**config_dict['page_parser'])
            
            if 'pipeline' in config_dict:
                settings.pipeline = PipelineConfig(**config_dict['pipeline'])
            
            return settings
    
    async def convert(
        self, 
        pdf_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[AsyncProgressCallback] = None
    ) -> str:
        """
        Convert a PDF to Markdown asynchronously.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Optional path to save the output
            progress_callback: Optional callback for progress updates
            
        Returns:
            The converted Markdown content
            
        Raises:
            PDFConversionError: If conversion fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise ParsingError(f"PDF file not found: {pdf_path}")
        
        await self._ensure_initialized()
        
        try:
            # Set up progress tracking
            if progress_callback:
                self._setup_progress_callback(progress_callback)
            
            # Process the document
            start_time = time.time()
            internal_doc = await self._pipeline.process(pdf_path)
            processing_time = time.time() - start_time
            
            # Convert to our API types
            result = self._convert_to_result(internal_doc, pdf_path, processing_time)
            
            # Get markdown content
            page_separator = self.config.get('page_separator', '\n\n--[PAGE: {page_number}]--\n\n')
            markdown = result.to_markdown(page_separator)
            
            # Save if output path provided
            if output_path:
                result.save(output_path)
            
            return markdown
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise PDFConversionError(f"Failed to convert {pdf_path}: {e}")
        finally:
            # Clean up progress callback
            if progress_callback:
                self._teardown_progress_callback()
    
    def convert_sync(
        self,
        pdf_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[ProgressCallback] = None
    ) -> str:
        """
        Convert a PDF to Markdown synchronously.
        
        This is a convenience wrapper around the async convert method.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Optional path to save the output
            progress_callback: Optional callback for progress updates
            
        Returns:
            The converted Markdown content
        """
        # Create async wrapper for sync callback if provided
        async_callback = None
        if progress_callback:
            async def async_callback(current: int, total: int, message: str):
                progress_callback(current, total, message)
        
        # Run async method in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running (e.g., in Jupyter), create a task
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self.convert(pdf_path, output_path, async_callback)
                )
            else:
                # Normal case: run in the loop
                return loop.run_until_complete(
                    self.convert(pdf_path, output_path, async_callback)
                )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(
                self.convert(pdf_path, output_path, async_callback)
            )
    
    async def stream_pages(
        self,
        pdf_path: Union[str, Path],
        progress_callback: Optional[AsyncProgressCallback] = None
    ) -> AsyncIterator[PageResult]:
        """
        Stream pages as they're processed.
        
        Args:
            pdf_path: Path to the PDF file
            progress_callback: Optional callback for progress updates
            
        Yields:
            PageResult objects as pages are processed
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise ParsingError(f"PDF file not found: {pdf_path}")
        
        await self._ensure_initialized()
        
        try:
            # Set up progress tracking
            if progress_callback:
                self._setup_progress_callback(progress_callback)
            
            # Start processing
            document_id = await self._pipeline.start_processing(pdf_path)
            
            # Stream pages as they complete
            processed_pages = set()
            total_pages = None
            
            while True:
                status = await self._pipeline.get_status(document_id)
                
                if status['total_pages'] and total_pages is None:
                    total_pages = status['total_pages']
                    if progress_callback:
                        await progress_callback(0, total_pages, "Processing pages...")
                
                # Yield newly completed pages
                for page_num, page_status in status.get('pages', {}).items():
                    if page_num not in processed_pages and page_status == 'completed':
                        processed_pages.add(page_num)
                        
                        # Get the page content
                        page = await self._pipeline.get_page_result(document_id, page_num)
                        if page:
                            yield self._convert_page_to_result(page)
                            
                            if progress_callback and total_pages:
                                await progress_callback(
                                    len(processed_pages), 
                                    total_pages,
                                    f"Processed page {page_num}"
                                )
                
                # Check if processing is complete
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    break
                
                await asyncio.sleep(0.1)  # Small delay to avoid busy waiting
                
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise PDFConversionError(f"Failed to stream {pdf_path}: {e}")
        finally:
            if progress_callback:
                self._teardown_progress_callback()
    
    async def process_batch(
        self,
        pdf_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[AsyncProgressCallback] = None
    ) -> List[DocumentResult]:
        """
        Process multiple PDFs in batch.
        
        Args:
            pdf_paths: List of paths to PDF files
            output_dir: Optional directory to save outputs
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of DocumentResult objects
        """
        results = []
        total = len(pdf_paths)
        
        for idx, pdf_path in enumerate(pdf_paths):
            if progress_callback:
                await progress_callback(idx, total, f"Processing {Path(pdf_path).name}")
            
            try:
                # Determine output path
                output_path = None
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{Path(pdf_path).stem}.md"
                
                # Convert the document
                markdown = await self.convert(pdf_path, output_path)
                
                # Create result
                result = DocumentResult(
                    source_path=Path(pdf_path),
                    pages=[],  # Will be populated if needed
                    total_pages=0,
                    status=ConversionStatus.COMPLETED,
                    markdown_content=markdown
                )
                results.append(result)
                
            except Exception as e:
                # Create failed result
                result = DocumentResult(
                    source_path=Path(pdf_path),
                    pages=[],
                    total_pages=0,
                    status=ConversionStatus.FAILED,
                    error_message=str(e)
                )
                results.append(result)
        
        if progress_callback:
            await progress_callback(total, total, "Batch processing complete")
        
        return results
    
    def _convert_to_result(
        self, 
        internal_doc: InternalDocument, 
        pdf_path: Path,
        processing_time: float
    ) -> DocumentResult:
        """Convert internal document to API result."""
        from ..core.models import ProcessingStatus
        
        # Map internal status to API status
        status_map = {
            ProcessingStatus.PENDING: ConversionStatus.PENDING,
            ProcessingStatus.PROCESSING: ConversionStatus.PROCESSING,
            ProcessingStatus.COMPLETED: ConversionStatus.COMPLETED,
            ProcessingStatus.FAILED: ConversionStatus.FAILED
        }
        
        # Convert pages
        pages = []
        for internal_page in internal_doc.pages:
            pages.append(self._convert_page_to_result(internal_page))
        
        # Create result
        return DocumentResult(
            source_path=pdf_path,
            pages=pages,
            total_pages=len(pages),
            status=status_map.get(internal_doc.status, ConversionStatus.COMPLETED),
            processing_time=processing_time,
            metadata=internal_doc.metadata,
            created_at=internal_doc.created_at,
            completed_at=internal_doc.completed_at
        )
    
    def _convert_page_to_result(self, internal_page: InternalPage) -> PageResult:
        """Convert internal page to API result."""
        from ..core.models import ProcessingStatus
        
        # Map internal status to API status
        status_map = {
            ProcessingStatus.PENDING: ConversionStatus.PENDING,
            ProcessingStatus.PROCESSING: ConversionStatus.PROCESSING,
            ProcessingStatus.COMPLETED: ConversionStatus.COMPLETED,
            ProcessingStatus.FAILED: ConversionStatus.FAILED
        }
        
        return PageResult(
            page_number=internal_page.page_number,
            content=internal_page.markdown_content or "",
            status=status_map.get(internal_page.status, ConversionStatus.COMPLETED),
            error_message=internal_page.error_message,
            metadata=internal_page.metadata.__dict__ if internal_page.metadata else {}
        )
    
    def _setup_progress_callback(self, callback: AsyncProgressCallback):
        """Set up progress tracking with callback."""
        # This would integrate with the pipeline's progress tracking
        # For now, we'll use the callback directly
        pass
    
    def _teardown_progress_callback(self):
        """Clean up progress tracking."""
        pass
    
    async def cleanup(self):
        """Clean up resources."""
        if self._pipeline:
            # Check if pipeline has cleanup method
            if hasattr(self._pipeline, 'cleanup'):
                await self._pipeline.cleanup()
            # Otherwise just set to None to release reference
            self._pipeline = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_initialized()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# Import os for environment variables
import os