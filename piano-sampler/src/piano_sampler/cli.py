"""piano-sampler CLI."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .pipeline import build as run_build


@click.group()
@click.version_option()
def main() -> None:
    """Slice a continuous piano-sampling session into a playable instrument."""


@main.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--instrument-name", required=True, help='Display name, e.g. "Bubby\'s Piano".')
def build(input_dir: Path, output_dir: Path, instrument_name: str) -> None:
    """End-to-end build: discover input WAVs -> slice -> label -> fades -> SFZ."""
    summary = run_build(input_dir=input_dir, output_dir=output_dir, instrument_name=instrument_name)
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
