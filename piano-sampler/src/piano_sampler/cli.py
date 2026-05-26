"""piano-sampler CLI — scaffold only, stages wired in subsequent commits."""

from __future__ import annotations

import click


@click.group()
@click.version_option()
def main() -> None:
    """Slice a continuous piano-sampling session into a playable instrument."""


@main.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output", "output_dir", required=True, type=click.Path(file_okay=False))
@click.option("--instrument-name", required=True, help='Display name, e.g. "Bubby\'s Piano".')
def build(input_dir: str, output_dir: str, instrument_name: str) -> None:
    """End-to-end build: slice → label → fades → SFZ."""
    click.echo(f"[stub] build {input_dir!r} -> {output_dir!r} as {instrument_name!r}")


@main.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output", "output_dir", required=True, type=click.Path(file_okay=False))
def slice_(input_dir: str, output_dir: str) -> None:
    """Stage 1: slice each input WAV into per-note WAVs (unlabeled)."""
    click.echo(f"[stub] slice {input_dir!r} -> {output_dir!r}")


@main.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True, file_okay=False))
def label(input_dir: str) -> None:
    """Stage 2: pitch-detect + chromatic-walk relabel sliced WAVs."""
    click.echo(f"[stub] label {input_dir!r}")


if __name__ == "__main__":
    main()
