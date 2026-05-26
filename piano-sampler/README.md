# piano-sampler

Pipeline that turns a continuous piano-sampling session into a playable sampled instrument (SFZ + Kontakt .nki).

## What it does

Input: a folder of long WAV recordings, each one a chromatic A0 → C8 pass at a single articulation / dynamic / round-robin.

Output: per-note labeled WAVs + an SFZ instrument + a "Kontakt fixups" checklist.

Stages:

1. **Slice** — onset detection per file, with per-file adaptive noise floor. Each slice runs from a detected onset to just before the next onset (or EOF), with trailing silence trimmed.
2. **Label** — pitch detection (YIN above the bottom octave, CREPE on A0-A1) plus a chromatic-order prior. Resolves retakes by keeping the *last* slice per detected note.
3. **Edge treatment** — short head fade-in (~2 ms) and tail fade-out (~80 ms) only, never altering the body of the note.
4. **SFZ generation** — emits a Sustain instrument and a Staccato instrument with velocity crossfade between quiet/loud layers and CC64 sustain-pedal behavior.

Full spec: [issue #1](https://github.com/BimRoss/audio-plugins/issues/1).

## Usage

```bash
# install
pip install -e .

# end-to-end
piano-sampler build \
  --input ./audio-in \
  --output ./audio-out \
  --instrument-name "Bubby's Piano"
```

See `piano-sampler --help` for stage-by-stage commands.

## Project layout

```
piano-sampler/
├── src/piano_sampler/
│   ├── slice.py           # onset detection, slicing
│   ├── pitch.py           # pitch detection + chromatic walk
│   ├── edge_treatment.py  # head/tail fades
│   ├── sfz.py             # SFZ generator
│   └── cli.py
├── scripts/
│   └── synth_take.py      # builds a synthetic chromatic take for testing
├── tests/
└── kontakt-fixups.md      # manual steps in Kontakt 6 after SFZ import
```
