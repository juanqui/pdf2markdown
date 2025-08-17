#!/usr/bin/env python3
"""Debug configuration to see what's being passed."""

from pathlib import Path
from pprint import pprint

# Load configuration
from pdf2markdown.config import load_settings

settings = load_settings(Path("config/default.yaml"))

print("=" * 60)
print("PAGE PARSER CONFIG (from settings):")
print("=" * 60)
page_parser_config = settings.get_page_parser_config()
pprint(page_parser_config)

print("\n" + "=" * 60)
print("PIPELINE CONFIG:")
print("=" * 60)
pipeline_config = settings.get_pipeline_config()
pprint(pipeline_config["page_parser"])
