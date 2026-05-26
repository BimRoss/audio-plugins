"""Generate a synthetic chromatic piano take for end-to-end pipeline testing.

Emits one WAV file containing 88 chromatic notes (A0 -> C8), each with a piano-ish
ADSR envelope, optional tiny global detune (so the synth doesn't trivially equal A440),
a couple of intentional retakes to exercise the chromatic-walk dedup, and a low-level
hiss + occasional "chair creak" thump in the silence between notes.

Usage:
    python scripts/synth_take.py --out /tmp/fake_take.wav [--articulation long|short]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from piano_sampler.notes import midi_to_freq  # noqa: E402
from piano_sampler.wav_io import write_wav  # noqa: E402


def synth_note(
    freq: float,
    sample_rate: int,
    duration: float,
    *,
    attack: float = 0.005,
    decay: float = 0.4,
    sustain_level: float = 0.5,
    release: float = 1.5,
    sustain_time: float = 0.0,
    n_partials: int = 6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthesize a stereo note with a piano-ish exponential decay envelope."""
    rng = rng or np.random.default_rng(0)
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate

    # Sum of partials with rolloff -> piano-ish timbre, not a pure sine.
    sig = np.zeros(n, dtype=np.float32)
    for k in range(1, n_partials + 1):
        amp = 1.0 / (k**1.6)
        phase = rng.uniform(0, 2 * np.pi)
        sig += amp * np.sin(2 * np.pi * k * freq * t + phase).astype(np.float32)
    sig /= np.max(np.abs(sig)) + 1e-9

    # ADSR-ish envelope: short attack, fast decay to sustain, then exponential release.
    env = np.zeros(n, dtype=np.float32)
    a = int(attack * sample_rate)
    d = int(decay * sample_rate)
    s = int(sustain_time * sample_rate)
    if a > 0:
        env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    if d > 0:
        env[a : a + d] = np.linspace(1, sustain_level, d, dtype=np.float32)
    env[a + d : a + d + s] = sustain_level
    rel_start = a + d + s
    rel_len = n - rel_start
    if rel_len > 0:
        env[rel_start:] = sustain_level * np.exp(
            -np.arange(rel_len, dtype=np.float32) / (release * sample_rate)
        )

    mono = (sig * env).astype(np.float32)
    # Stereo with very mild width: identical L/R for test simplicity.
    stereo = np.stack([mono, mono], axis=1)
    return stereo


def build_take(
    *,
    sample_rate: int = 48000,
    articulation: str = "long",
    retake_indices: tuple[int, ...] = (12, 47),
    creak_indices: tuple[int, ...] = (30,),
    global_detune_cents: float = -4.0,  # Bubby-ish: a few cents flat globally
    gap_seconds: float = 0.6,
    noise_rms: float = 0.0008,
    seed: int = 42,
) -> np.ndarray:
    """Build a stereo synthetic chromatic take A0..C8 with retakes + room noise.

    Returns a (N, 2) float32 array in [-1, 1].
    """
    rng = np.random.default_rng(seed)

    if articulation == "long":
        note_dur = 1.8
        release = 1.2
        sustain_time = 0.2
    elif articulation == "short":
        note_dur = 0.5
        release = 0.15
        sustain_time = 0.0
    else:
        raise ValueError(f"unknown articulation: {articulation}")

    detune = 2 ** (global_detune_cents / 1200.0)

    notes: list[np.ndarray] = []
    plan: list[int] = list(range(21, 109))  # A0..C8 MIDI
    # Insert retakes: duplicate slice at the given chromatic index (0-based within plan).
    expanded: list[int] = []
    for i, midi in enumerate(plan):
        expanded.append(midi)
        if i in retake_indices:
            expanded.append(midi)  # retake = same note again

    gap_samples = int(gap_seconds * sample_rate)
    gap = np.zeros((gap_samples, 2), dtype=np.float32)

    for slice_idx, midi in enumerate(expanded):
        freq = midi_to_freq(midi) * detune
        note = synth_note(
            freq,
            sample_rate,
            note_dur,
            release=release,
            sustain_time=sustain_time,
            rng=rng,
        )
        notes.append(note)
        notes.append(gap)
        # Inject a "chair creak" in the gap after a specific slice.
        if slice_idx in creak_indices:
            creak_len = int(0.08 * sample_rate)
            creak = (rng.standard_normal((creak_len, 2)) * 0.04).astype(np.float32)
            # Low-pass-ish: cumulative-sum -> drift, then normalize.
            creak = np.cumsum(creak, axis=0)
            creak /= np.max(np.abs(creak)) + 1e-9
            creak *= 0.03
            # Place it mid-gap.
            offset = gap_samples // 2
            gap_with_creak = gap.copy()
            gap_with_creak[offset : offset + creak_len] += creak
            notes[-1] = gap_with_creak

    take = np.concatenate(notes, axis=0)

    # Add baseline room noise (white + tiny pink-ish tilt).
    noise = (rng.standard_normal(take.shape) * noise_rms).astype(np.float32)
    take = take + noise

    # Headroom: keep peaks well below clipping.
    take *= 0.5
    return take


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--articulation", choices=["long", "short"], default="long")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--bit-depth", type=int, default=24, choices=[16, 24, 32])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    take = build_take(
        sample_rate=args.sample_rate,
        articulation=args.articulation,
        seed=args.seed,
    )
    write_wav(args.out, take, args.sample_rate, bit_depth=args.bit_depth)
    duration = take.shape[0] / args.sample_rate
    print(f"wrote {args.out} — {duration:.1f}s, {args.sample_rate} Hz, {args.bit_depth}-bit stereo")


if __name__ == "__main__":
    main()
