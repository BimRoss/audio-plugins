# audio-plugins

BimRoss audio plugin projects and shared sampler-pipeline tooling.

## Projects

- [`piano-sampler/`](./piano-sampler) — Python pipeline that slices a continuous piano-sampling session into per-note WAVs, labels them via pitch detection with a chromatic-order prior, and emits an SFZ + Kontakt-ready instrument folder. First instrument built with it: *Bubby's Piano* (see issue #1).

## Issues

- [#1 — Bubby's Piano (Matt Wood)](https://github.com/BimRoss/audio-plugins/issues/1) — first instrument, drives the pipeline design.
- [#3 — Audio/ML workbench node pool](https://github.com/BimRoss/audio-plugins/issues/3) — infra ask for bigger jobs.
