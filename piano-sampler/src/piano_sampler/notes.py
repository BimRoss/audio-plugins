"""MIDI note number <-> frequency / name helpers. A4 = MIDI 69 = 440 Hz."""

from __future__ import annotations

_PITCH_CLASSES = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
_IS_SHARP = [False, True, False, True, False, False, True, False, True, False, True, False]

PIANO_LOWEST = 21  # A0
PIANO_HIGHEST = 108  # C8


def midi_to_freq(midi: float, a4: float = 440.0) -> float:
    return a4 * (2 ** ((midi - 69) / 12))


def freq_to_midi(freq: float, a4: float = 440.0) -> float:
    import math

    return 69.0 + 12.0 * math.log2(freq / a4)


def midi_to_name(midi: int, sharp: str = "#") -> str:
    """Render a MIDI note as e.g. 'A#0' (sharp='#') or 'As0' (sharp='s')."""
    pc = midi % 12
    octave = (midi // 12) - 1
    name = _PITCH_CLASSES[pc]
    if _IS_SHARP[pc]:
        name += sharp
    return f"{name}{octave}"
