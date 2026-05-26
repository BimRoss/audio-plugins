"""MIDI note number <-> frequency / name helpers. A4 = MIDI 69 = 440 Hz."""

from __future__ import annotations

_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]

PIANO_LOWEST = 21  # A0
PIANO_HIGHEST = 108  # C8


def midi_to_freq(midi: float, a4: float = 440.0) -> float:
    return a4 * (2 ** ((midi - 69) / 12))


def freq_to_midi(freq: float, a4: float = 440.0) -> float:
    import math

    return 69.0 + 12.0 * math.log2(freq / a4)


def midi_to_name(midi: int) -> str:
    pc = midi % 12
    octave = (midi // 12) - 1
    return f"{_NAMES[pc]}{octave}"
