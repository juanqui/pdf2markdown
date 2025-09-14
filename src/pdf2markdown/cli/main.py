"""Command-line interface for PDF to Markdown converter using the library API."""

import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from .. import __version__
from ..api import Config, ConfigBuilder, PDFConverter
from ..core import ProcessingStatus
from ..pipeline.global_manager import GlobalPipelineManager
from ..utils import setup_logging
from ..utils.cache_manager import CacheManager
from ..utils.statistics import get_statistics_tracker, reset_statistics

console = Console()
logger = logging.getLogger(__name__)


@click.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file/directory path (optional, defaults to same path with .md extension)",
)
@click.option(
    "-c", "--config", type=click.Path(exists=True, path_type=Path), help="Configuration file path"
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (can also be set via OPENAI_API_KEY env var)",
)
@click.option("--model", default=None, help="LLM model to use (overrides config file)")
@click.option(
    "--resolution",
    type=int,
    default=None,
    help="DPI resolution for rendering PDF pages (overrides config file)",
)
@click.option(
    "--max-dimension",
    type=int,
    default=None,
    help="Maximum pixels for longest side of rendered images (overrides config file)",
)
@click.option(
    "--page-workers",
    type=int,
    default=None,
    help="Number of parallel page processing workers (overrides config file)",
)
@click.option("--no-progress", is_flag=True, help="Disable progress logging")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level",
)
@click.option(
    "--cache-dir", type=click.Path(path_type=Path), help="Directory for caching rendered images"
)
@click.option(
    "--page-limit", type=int, help="Limit the number of pages to convert (useful for debugging)"
)
@click.option(
    "--save-config", type=click.Path(path_type=Path), help="Save current configuration to file"
)
@click.option("--resume", is_flag=True, help="Resume processing using cached images and markdown")
@click.option("--clear-cache", is_flag=True, help="Clear all cached data before processing")
@click.option("--cache-stats", is_flag=True, help="Show cache statistics and exit")
@click.version_option(version=__version__)
def main(
    inputs: tuple[Path, ...],
    output: Path | None,
    config: Path | None,
    api_key: str | None,
    model: str | None,
    resolution: int | None,
    max_dimension: int | None,
    page_workers: int | None,
    no_progress: bool,
    log_level: str,
    cache_dir: Path | None,
    page_limit: int | None,
    save_config: Path | None,
    resume: bool,
    clear_cache: bool,
    cache_stats: bool,
):
    """Convert PDF documents to Markdown using LLMs.

    This tool processes PDF files by rendering each page as an image
    and using an LLM to extract and format the content as Markdown.

    Supports multiple inputs:
    - Single file: pdf2markdown input.pdf
    - Multiple files: pdf2markdown file1.pdf file2.pdf file3.pdf
    - Directory: pdf2markdown /path/to/pdfs/ (processes all PDFs in directory)
    - Mixed: pdf2markdown file1.pdf /path/to/pdfs/ file2.pdf

    Caching and Resume:
    - Automatically caches rendered images and LLM outputs
    - Use --resume to skip processing of already-converted content
    - Use --clear-cache to force fresh processing
    - Use --cache-stats to view cache usage

    Examples:
        pdf2markdown input.pdf                    # Output: input.md
        pdf2markdown input.pdf -o output.md       # Output: output.md
        pdf2markdown *.pdf                        # Process all PDFs in current dir
        pdf2markdown /docs/ -o /output/            # Process dir, output to dir
        pdf2markdown input.pdf --resume           # Resume interrupted processing
        pdf2markdown input.pdf --clear-cache      # Force fresh processing
    """
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Setup logging
    setup_logging(level=log_level)

    # Add rich handler for better console output
    logging.getLogger().handlers = [RichHandler(console=console, rich_tracebacks=True)]

    try:
        # Build configuration using the library API
        if config:
            # Load from YAML file
            cfg = Config.from_yaml(config)
            builder = ConfigBuilder().merge(cfg.to_dict())
        else:
            # Try to load default config if it exists
            default_config = Path("config/default.yaml")
            if default_config.exists():
                cfg = Config.from_yaml(default_config)
                builder = ConfigBuilder().merge(cfg.to_dict())
            else:
                # Start with defaults
                builder = ConfigBuilder()

        # Apply command-line overrides
        if api_key:
            # Update existing llm_provider or create new one
            current_config = builder._config
            if "llm_provider" in current_config:
                current_config["llm_provider"]["api_key"] = api_key
            else:
                builder.with_openai(api_key=api_key)

        if model is not None:
            current_config = builder._config
            if "llm_provider" in current_config:
                current_config["llm_provider"]["model"] = model
            else:
                builder.with_openai(api_key=api_key or "${OPENAI_API_KEY}", model=model)

        if resolution is not None:
            builder.with_resolution(resolution)

        if max_dimension is not None:
            current_config = builder._config
            if "document_parser" not in current_config:
                current_config["document_parser"] = {}
            current_config["document_parser"]["max_dimension"] = max_dimension

        if page_workers is not None:
            builder.with_page_workers(page_workers)

        if cache_dir:
            builder.with_cache_dir(cache_dir)

        if no_progress:
            builder.with_progress(False)
        else:
            builder.with_progress(True)

        # Set log level for library
        builder.with_log_level(log_level)

        # Build final configuration
        final_config = builder.build()
        final_config_dict = final_config.to_dict()

        # Handle cache-related operations
        cache_config = final_config_dict.get("cache", {}) or final_config_dict.get(
            "document_parser", {}
        )
        cache_base_dir = Path(
            cache_config.get("base_dir", cache_config.get("cache_dir", "/tmp/pdf2markdown/cache"))
        )
        cache_manager = CacheManager(cache_base_dir)

        # Show cache statistics if requested
        if cache_stats:
            stats = cache_manager.get_cache_stats()
            console.print("[bold blue]Cache Statistics[/bold blue]")
            console.print(f"Cache directory: {cache_base_dir}")
            console.print(f"Total size: {stats['total_size'] / (1024*1024*1024):.2f} GB")
            console.print(f"Document count: {stats['document_count']}")
            if stats["directories"]:
                console.print("\n[bold]Recent documents:[/bold]")
                for _i, doc in enumerate(stats["directories"][:10]):
                    size_mb = doc["size"] / (1024 * 1024)
                    import datetime

                    mod_time = datetime.datetime.fromtimestamp(doc["modified"]).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    console.print(f"  {doc['hash'][:12]}... - {size_mb:.1f} MB - {mod_time}")
                if len(stats["directories"]) > 10:
                    console.print(f"  ... and {len(stats['directories']) - 10} more")
            return

        # Clear cache if requested
        if clear_cache:
            cache_manager.clear_cache()
            console.print("[green]Cache cleared successfully[/green]")

        # Set cache usage flags
        if cache_dir:
            builder.with_cache_dir(cache_dir)

        # Handle resume mode
        if resume:
            # Enable caching for resume mode
            current_config = builder._config
            if "document_parser" not in current_config:
                current_config["document_parser"] = {}
            if "page_parser" not in current_config:
                current_config["page_parser"] = {}
            current_config["document_parser"]["use_cache"] = True
            current_config["page_parser"]["use_cache"] = True
            console.print("[cyan]Resume mode enabled - using cached data where available[/cyan]")

        # Save configuration if requested
        if save_config:
            import yaml

            config_dict = final_config_dict
            with open(save_config, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            console.print(f"[green]Configuration saved to {save_config}[/green]")
            return

        # Collect all PDF files to process
        pdf_files = []
        for input_path in inputs:
            if input_path.is_file():
                if input_path.suffix.lower() == ".pdf":
                    pdf_files.append(input_path)
                else:
                    console.print(f"[yellow]Warning: Skipping non-PDF file: {input_path}[/yellow]")
            elif input_path.is_dir():
                # Find all PDF files in directory
                dir_pdfs = list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF"))
                if dir_pdfs:
                    pdf_files.extend(sorted(dir_pdfs))
                    console.print(f"[cyan]Found {len(dir_pdfs)} PDF files in {input_path}[/cyan]")
                else:
                    console.print(
                        f"[yellow]Warning: No PDF files found in directory: {input_path}[/yellow]"
                    )
            else:
                console.print(f"[red]Error: Path does not exist: {input_path}[/red]")
                sys.exit(1)

        if not pdf_files:
            console.print("[red]Error: No PDF files found to process[/red]")
            sys.exit(1)

        # Determine output strategy
        multiple_inputs = len(pdf_files) > 1
        output_is_dir = output and output.is_dir()

        if multiple_inputs and output and not output_is_dir and not output.parent.exists():
            # Check if output looks like a directory (ends with /)
            if str(output).endswith("/") or str(output).endswith("\\"):
                output_is_dir = True
                output.mkdir(parents=True, exist_ok=True)

        # Display configuration
        console.print(f"[bold blue]PDF to Markdown Converter v{__version__}[/bold blue]")
        if multiple_inputs:
            console.print(f"Inputs: {len(pdf_files)} PDF files")
            if output_is_dir or not output:
                console.print(
                    f"Output: Individual .md files {'in ' + str(output) if output else 'next to source files'}"
                )
            else:
                console.print(
                    "[yellow]Warning: Multiple inputs with single output file - will concatenate results[/yellow]"
                )
                console.print(f"Output: {output}")
        else:
            console.print(f"Input: {pdf_files[0]}")
            if output:
                console.print(f"Output: {output}")
            else:
                console.print(f"Output: {pdf_files[0].with_suffix('.md')}")

        # Get model from configuration
        llm_config = final_config.llm_provider
        display_model = (
            llm_config.get("model", "Not configured") if llm_config else "Not configured"
        )
        console.print(f"Model: {display_model}")

        doc_config = final_config.document_parser
        resolution_val = doc_config.get("resolution", 300) if doc_config else 300
        console.print(f"Resolution: {resolution_val} DPI")

        if doc_config and doc_config.get("max_dimension"):
            console.print(f"Max Dimension: {doc_config.get('max_dimension')} pixels")

        pipeline_config = final_config.pipeline
        workers = pipeline_config.get("page_workers", 10) if pipeline_config else 10
        console.print(f"Page Workers: {workers}")

        if page_limit:
            console.print(f"Page Limit: {page_limit}")
        console.print()

        # Reset statistics for this run
        reset_statistics()

        # Create progress callback if enabled
        progress_callback = None
        if not no_progress:

            def progress_callback(current: int, total: int, message: str):
                console.print(f"[cyan]Progress: {current}/{total} - {message}[/cyan]")

        # Handle page limit by modifying the configuration
        if page_limit:
            final_config_dict["page_limit"] = page_limit

        # Process files
        successful_conversions = []
        failed_conversions = []

        # Choose processing strategy based on number of files
        if multiple_inputs:
            # Use GlobalPipelineManager for multiple files to ensure proper worker limits
            console.print(f"[cyan]Using shared worker pool for {len(pdf_files)} documents[/cyan]")

            # Prepare document list with output paths
            documents = []
            for pdf_file in pdf_files:
                # Determine output path for this file
                if output_is_dir:
                    # Output to specified directory
                    output_path = output / pdf_file.with_suffix(".md").name
                elif output and not output_is_dir:
                    # Single output file (concatenate mode) - handle specially
                    output_path = output
                else:
                    # Default: next to source file
                    output_path = pdf_file.with_suffix(".md")

                documents.append((pdf_file, output_path))

            # Handle concatenation mode specially
            if output and not output_is_dir and len(pdf_files) > 1:
                console.print(
                    f"[yellow]Concatenation mode: combining {len(pdf_files)} files into {output}[/yellow]"
                )

                # Process each file and concatenate results
                try:
                    # Use the async batch processor
                    import asyncio

                    def sync_progress_callback(current: int, total: int, message: str):
                        if progress_callback:
                            progress_callback(current, total, message)

                    async def process_with_concatenation():
                        async with GlobalPipelineManager(final_config_dict) as manager:
                            combined_content = []

                            for i, (pdf_file, _) in enumerate(documents, 1):
                                sync_progress_callback(
                                    i, len(documents), f"Processing {pdf_file.name}"
                                )

                                try:
                                    # Process without output path (get content only)
                                    result = await manager.process_document(pdf_file, None)

                                    if i > 1:
                                        combined_content.append(
                                            f"\n\n<!-- Source: {pdf_file.name} -->\n\n"
                                        )

                                    # Combine page content
                                    page_separator = final_config_dict.get(
                                        "page_separator", "\n\n--[PAGE: {page_number}]--\n\n"
                                    )
                                    page_contents = []
                                    for page in sorted(result.pages, key=lambda p: p.page_number):
                                        if page.markdown_content:
                                            if page.page_number > 1:
                                                separator = page_separator.format(
                                                    page_number=page.page_number
                                                )
                                                page_contents.append(separator)
                                            page_contents.append(page.markdown_content)

                                    combined_content.append("".join(page_contents))
                                    successful_conversions.append((pdf_file, output))

                                except Exception as e:
                                    failed_conversions.append((pdf_file, str(e)))
                                    console.print(
                                        f"[red]✗ Failed to convert {pdf_file.name}: {e}[/red]"
                                    )

                            # Write combined content
                            if combined_content:
                                output.parent.mkdir(parents=True, exist_ok=True)
                                with open(output, "w", encoding="utf-8") as f:
                                    f.write("".join(combined_content))

                    asyncio.run(process_with_concatenation())

                except Exception as e:
                    console.print(f"[red]✗ Batch processing failed: {e}[/red]")
                    failed_conversions.extend(
                        [
                            (pdf, str(e))
                            for pdf, _ in documents
                            if (pdf, output) not in successful_conversions
                        ]
                    )
            else:
                # Normal multi-file processing (separate outputs)
                try:
                    import asyncio

                    def sync_progress_callback(current: int, total: int, message: str):
                        if progress_callback:
                            progress_callback(current, total, message)

                    async def process_batch():
                        async with GlobalPipelineManager(final_config_dict) as manager:
                            results = await manager.process_documents_batch(
                                documents, sync_progress_callback
                            )

                            for (pdf_file, output_path), result in zip(
                                documents, results, strict=False
                            ):
                                if result.status == ProcessingStatus.COMPLETED:
                                    successful_conversions.append((pdf_file, output_path))
                                else:
                                    error_msg = result.error_message or "Unknown error"
                                    failed_conversions.append((pdf_file, error_msg))

                    asyncio.run(process_batch())

                except Exception as e:
                    console.print(f"[red]✗ Batch processing failed: {e}[/red]")
                    failed_conversions.extend([(pdf, str(e)) for pdf, _ in documents])
        else:
            # Single file - use existing PDFConverter for compatibility
            pdf_file = pdf_files[0]
            if output:
                output_path = output
            else:
                output_path = pdf_file.with_suffix(".md")

            try:
                converter = PDFConverter(config=final_config)
                converter.convert_sync(pdf_file, output_path, progress_callback=progress_callback)
                successful_conversions.append((pdf_file, output_path))
            except Exception as e:
                failed_conversions.append((pdf_file, str(e)))
                raise

        # Report results
        if successful_conversions:
            console.print(
                f"\n[green]✓ Successfully converted {len(successful_conversions)} file(s)![/green]"
            )
            if not multiple_inputs or (multiple_inputs and not output_is_dir and output):
                # Show single output or concatenated output
                console.print(f"Output saved to: {successful_conversions[0][1]}")
            else:
                # Show directory of outputs
                for pdf_file, output_path in successful_conversions:
                    console.print(f"  {pdf_file.name} → {output_path}")

        if failed_conversions:
            console.print(f"\n[red]✗ Failed to convert {len(failed_conversions)} file(s):[/red]")
            for pdf_file, error in failed_conversions:
                console.print(f"  {pdf_file.name}: {error}")

        # Display statistics report if available
        stats = get_statistics_tracker()
        if stats:
            stats.print_report(console)

        # Exit with error code if any conversions failed
        if failed_conversions and not successful_conversions:
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Conversion cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
