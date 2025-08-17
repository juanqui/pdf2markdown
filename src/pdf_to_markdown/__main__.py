"""Main entry point for PDF to Markdown converter."""

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from pdf_to_markdown import __version__
from pdf_to_markdown.config import load_settings
from pdf_to_markdown.pipeline import PipelineCoordinator
from pdf_to_markdown.utils import setup_logging
from pdf_to_markdown.utils.statistics import get_statistics_tracker, reset_statistics

console = Console()
logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Output markdown file path")
@click.option(
    "-c", "--config", type=click.Path(exists=True, path_type=Path), help="Configuration file path"
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (can also be set via OPENAI_API_KEY env var)",
)
@click.option("--model", default=None, help="LLM model to use (overrides config file)")
@click.option("--resolution", type=int, default=None, help="DPI resolution for rendering PDF pages (overrides config file)")
@click.option(
    "--page-workers", type=int, default=None, help="Number of parallel page processing workers (overrides config file)"
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
@click.version_option(version=__version__)
def main(
    input_file: Path,
    output: Path | None,
    config: Path | None,
    api_key: str | None,
    model: str | None,
    resolution: int | None,
    page_workers: int | None,
    no_progress: bool,
    log_level: str,
    cache_dir: Path | None,
    page_limit: int | None,
    save_config: Path | None,
):
    """Convert PDF documents to Markdown using LLMs.

    This tool processes PDF files by rendering each page as an image
    and using an LLM to extract and format the content as Markdown.

    Example:
        pdf-to-markdown input.pdf -o output.md
    """
    # Setup logging
    setup_logging(level=log_level)

    # Add rich handler for better console output
    logging.getLogger().handlers = [RichHandler(console=console, rich_tracebacks=True)]

    try:
        # Load settings
        settings = load_settings(config)

        # Apply command-line overrides to top-level llm_provider
        if not settings.config.llm_provider:
            from pdf_to_markdown.config.schemas import LLMProviderConfig

            # Only set api_key if provided
            llm_config_args = {}
            if api_key:
                llm_config_args["api_key"] = api_key
            settings.config.llm_provider = LLMProviderConfig(**llm_config_args)

        if api_key:
            settings.config.llm_provider.api_key = api_key
        if model is not None:
            settings.config.llm_provider.model = model
        if resolution is not None:
            settings.config.document_parser.resolution = resolution
        if page_workers is not None:
            settings.config.pipeline.page_workers = page_workers
        if cache_dir:
            settings.config.document_parser.cache_dir = cache_dir
        if no_progress:
            settings.config.pipeline.enable_progress = False

        # Set output path
        if output:
            output_path = output
        else:
            output_path = input_file.with_suffix(".md")

        # Save configuration if requested
        if save_config:
            settings.save(save_config)
            console.print(f"[green]Configuration saved to {save_config}[/green]")
            return

        # Validate settings
        if not settings.validate():
            console.print("[red]Invalid configuration. Please check your settings.[/red]")
            sys.exit(1)

        # Display configuration
        console.print(f"[bold blue]PDF to Markdown Converter v{__version__}[/bold blue]")
        console.print(f"Input: {input_file}")
        console.print(f"Output: {output_path}")
        # Get model from top-level llm_provider
        display_model = (
            settings.config.llm_provider.model if settings.config.llm_provider else "Not configured"
        )
        console.print(f"Model: {display_model}")
        console.print(f"Resolution: {settings.config.document_parser.resolution} DPI")
        console.print(f"Page Workers: {settings.config.pipeline.page_workers}")
        if page_limit:
            console.print(f"Page Limit: {page_limit}")
        console.print()
        
        # Reset statistics for this run
        reset_statistics()

        # Run the conversion
        asyncio.run(convert_pdf(input_file, output_path, settings, page_limit))

        console.print("\n[green]✓ Conversion complete![/green]")
        console.print(f"Output saved to: {output_path}")
        
        # Display statistics report
        stats = get_statistics_tracker()
        if stats:
            stats.print_report(console)

    except KeyboardInterrupt:
        console.print("\n[yellow]Conversion cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Conversion failed")
        sys.exit(1)


async def convert_pdf(
    input_file: Path, output_path: Path, settings, page_limit: int | None = None
) -> None:
    """Convert a PDF file to Markdown.

    Args:
        input_file: Path to input PDF file
        output_path: Path to output Markdown file
        settings: Application settings
        page_limit: Optional limit on number of pages to convert
    """
    # Get pipeline configuration
    pipeline_config = settings.get_pipeline_config()
    pipeline_config["output_dir"] = output_path.parent
    if page_limit:
        pipeline_config["page_limit"] = page_limit

    # Create pipeline coordinator
    pipeline = PipelineCoordinator(pipeline_config)

    # Process the document
    document = await pipeline.process(input_file)

    # Save the output
    document.output_path = output_path

    # Write markdown content
    markdown_content = []
    logger.debug(f"Document has {len(document.pages)} pages")

    # Get the page separator template from settings
    page_separator_template = settings.config.page_separator

    sorted_pages = sorted(document.pages, key=lambda p: p.page_number)
    for i, page in enumerate(sorted_pages):
        logger.debug(
            f"Page {page.page_number}: content={bool(page.markdown_content)}, length={len(page.markdown_content) if page.markdown_content else 0}"
        )
        if page.markdown_content:
            # Add the page content
            markdown_content.append(page.markdown_content)

            # Add separator between pages (but not after the last page)
            if i < len(sorted_pages) - 1:
                # Format the separator with the next page number
                separator = page_separator_template.format(
                    page_number=sorted_pages[i + 1].page_number
                )
                markdown_content.append(separator)

    logger.debug(f"Writing {len(markdown_content)} sections to output file")

    # Join all content and write to file
    final_content = "".join(markdown_content)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_content)


if __name__ == "__main__":
    main()
