"""Command-line interface for PDF to Markdown converter using the library API."""

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from .. import __version__
from ..api import Config, ConfigBuilder, PDFConverter
from ..utils import setup_logging
from ..utils.statistics import get_statistics_tracker, reset_statistics

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
@click.option(
    "--resolution",
    type=int,
    default=None,
    help="DPI resolution for rendering PDF pages (overrides config file)",
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
        pdf2markdown input.pdf -o output.md
    """
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

        # Set output path
        if output:
            output_path = output
        else:
            output_path = input_file.with_suffix(".md")

        # Save configuration if requested
        if save_config:
            import yaml

            config_dict = final_config.to_dict()
            with open(save_config, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            console.print(f"[green]Configuration saved to {save_config}[/green]")
            return

        # Display configuration
        console.print(f"[bold blue]PDF to Markdown Converter v{__version__}[/bold blue]")
        console.print(f"Input: {input_file}")
        console.print(f"Output: {output_path}")

        # Get model from configuration
        llm_config = final_config.llm_provider
        display_model = (
            llm_config.get("model", "Not configured") if llm_config else "Not configured"
        )
        console.print(f"Model: {display_model}")

        doc_config = final_config.document_parser
        resolution_val = doc_config.get("resolution", 300) if doc_config else 300
        console.print(f"Resolution: {resolution_val} DPI")

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

        # Run the conversion using the library API
        converter = PDFConverter(config=final_config)

        # Handle page limit by modifying the configuration
        if page_limit:
            # This would need to be implemented in the converter
            # For now, we'll just process normally
            console.print(
                "[yellow]Note: Page limit is not yet implemented in library mode[/yellow]"
            )

        # Run synchronously for CLI
        converter.convert_sync(input_file, output_path, progress_callback=progress_callback)

        console.print("\n[green]✓ Conversion complete![/green]")
        console.print(f"Output saved to: {output_path}")

        # Display statistics report if available
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


if __name__ == "__main__":
    main()
