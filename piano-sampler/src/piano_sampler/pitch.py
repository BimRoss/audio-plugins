"""Pitch detection + chromatic-walk relabeler.

Pitch detection: simple autocorrelation-based fundamental estimator, run on a
short window taken after the attack settles. Tolerant of ±50 cents (so Bubby
being a few cents flat globally is invisible).

Chromatic walk: given a sequence of detected pitches, assign each slice a
canonical MIDI note assuming the operator played A0 -> C8 chromatically.
Handles retakes (same pitch repeated -> keep last) and rejects non-piano
transients (e.g. chair creaks) that don't match the chromatic progression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .notes import PIANO_HIGHEST, PIANO_LOWEST, freq_to_midi, midi_to_freq


@dataclass
class PitchConfig:
    analysis_offset_seconds: float = 0.02  # skip past hammer transient
    analysis_window_seconds: float = 0.40  # length of pitch-analysis window
    min_freq_hz: float = 25.0  # A0 = 27.5 Hz, give a hair of margin below
    max_freq_hz: float = 4500.0  # C8 = 4186 Hz
    tolerance_cents: float = 50.0  # how far a detected pitch may sit from a MIDI note
    subharmonic_check_ratio: float = 0.7  # if peak at 2x lag has at least 0.7 * peak height, prefer 2x lag (octave-down fundamental)


def estimate_pitch_hz(
    mono_slice: np.ndarray,
    sample_rate: int,
    cfg: PitchConfig | None = None,
) -> float | None:
    """Return the estimated fundamental frequency in Hz, or None if no clear pitch."""
    cfg = cfg or PitchConfig()
    offset = int(cfg.analysis_offset_seconds * sample_rate)
    window = int(cfg.analysis_window_seconds * sample_rate)
    # Find the highest-RMS sub-window of length `window` within the slice
    # starting at `offset`. This places the analysis window on the strongest
    # part of the note's body (essential for staccato samples where the body
    # is short and pitch dies before the slice ends).
    seg_end = mono_slice.size
    if seg_end <= offset + 256:
        return None
    if seg_end - offset <= window:
        seg = mono_slice[offset:seg_end]
    else:
        # Scan in window/4 steps for the highest-RMS sub-window.
        step = max(1, window // 4)
        best_start = offset
        best_rms = -1.0
        s = offset
        while s + window <= seg_end:
            chunk = mono_slice[s : s + window]
            r = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2) + 1e-20))
            if r > best_rms:
                best_rms = r
                best_start = s
            s += step
        seg = mono_slice[best_start : best_start + window]
    if seg.size < 256:
        return None

    seg = seg.astype(np.float32)
    # Remove DC and apply a Hann window so autocorrelation focuses on the
    # central body rather than the boundaries.
    seg = seg - seg.mean()
    hann = np.hanning(seg.size).astype(np.float32)
    seg = seg * hann
    energy = float((seg**2).sum())
    if energy < 1e-8:
        return None

    # Autocorrelation via FFT.
    n = 1 << (int(np.ceil(np.log2(2 * seg.size))))
    spec = np.fft.rfft(seg, n=n)
    ac = np.fft.irfft(spec * np.conj(spec), n=n).real[: seg.size]
    if ac[0] <= 0:
        return None
    ac /= ac[0]

    min_lag = max(2, int(sample_rate / cfg.max_freq_hz))
    max_lag = min(ac.size - 2, int(sample_rate / cfg.min_freq_hz))
    if max_lag <= min_lag + 1:
        return None

    # Find ALL local maxima in [min_lag, max_lag] with normalized correlation
    # above a confidence floor. For piano, the fundamental is not always the
    # tallest peak — upper partials often dominate, especially mid-decay in
    # the bass. We pick the LARGEST-lag (lowest-frequency) candidate whose
    # height is at least subharmonic_check_ratio * tallest_peak. This biases
    # toward the true fundamental and avoids octave-error.
    floor = 0.2
    peaks: list[tuple[int, float]] = []
    for i in range(min_lag, max_lag):
        if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > floor:
            peaks.append((i, float(ac[i])))
    if not peaks:
        return None
    tallest_height = max(h for _, h in peaks)
    candidate_floor = cfg.subharmonic_check_ratio * tallest_height
    # Pick the largest lag whose height clears the candidate floor.
    qualifying = [(lag, h) for lag, h in peaks if h >= candidate_floor]
    peak = max(lag for lag, _ in qualifying)

    # Parabolic interpolation around the peak for sub-sample lag.
    if 1 <= peak < ac.size - 1:
        a, b, c = ac[peak - 1], ac[peak], ac[peak + 1]
        denom = (a - 2 * b + c)
        if abs(denom) > 1e-12:
            shift = 0.5 * (a - c) / denom
            peak_f = peak + shift
        else:
            peak_f = float(peak)
    else:
        peak_f = float(peak)

    if peak_f <= 0:
        return None
    return sample_rate / peak_f


def nearest_midi(freq_hz: float, a4: float = 440.0) -> tuple[int, float]:
    """Return (nearest_midi_note, cents_offset). cents_offset signed (+sharp, -flat)."""
    midi_f = freq_to_midi(freq_hz, a4=a4)
    midi_i = int(round(midi_f))
    cents = (midi_f - midi_i) * 100.0
    return midi_i, cents


@dataclass
class LabeledSlice:
    slice_index: int  # original index in the slicer's output
    midi: int | None  # assigned MIDI note (None = rejected as non-pitched)
    detected_freq_hz: float | None
    detected_midi: int | None
    cents_off: float | None
    reason: str  # "accepted", "retake_replaced", "non_pitched", "out_of_walk"


def chromatic_walk(
    detected_freqs: list[float | None],
    *,
    low_midi: int = PIANO_LOWEST,
    high_midi: int = PIANO_HIGHEST,
    cfg: PitchConfig | None = None,
) -> list[LabeledSlice]:
    """Assign MIDI notes to slices assuming the operator played low->high chromatically.

    Rules:
    - Expected pitch at step `e` is `low_midi + e`. Walk advances on a good match.
    - If the detected pitch equals the *current* expected note: accept, advance.
    - If the detected pitch equals the *previously accepted* note (within tolerance):
      treat as retake -> replace the previous LabeledSlice with this one, do not advance.
    - If the detected pitch equals expected+1 but we've never produced expected:
      accept expected+1, advance past expected (e.g. operator skipped a note).
      We do NOT inject phantoms — gaps stay as-is for the SFZ to interpolate.
    - Otherwise (non-pitched, out-of-walk, e.g. a chair creak): reject (`midi=None`).
    """
    cfg = cfg or PitchConfig()
    tol_cents = cfg.tolerance_cents

    labels: list[LabeledSlice] = []
    expected = low_midi
    last_accepted_midi: int | None = None
    last_accepted_label_index: int | None = None

    for i, freq in enumerate(detected_freqs):
        if freq is None or freq < cfg.min_freq_hz or freq > cfg.max_freq_hz:
            labels.append(
                LabeledSlice(i, midi=None, detected_freq_hz=freq, detected_midi=None, cents_off=None, reason="non_pitched")
            )
            continue
        det_midi, cents = nearest_midi(freq)
        # Distance from this slice's detected pitch to candidate notes, in cents.
        if last_accepted_midi is not None:
            cents_to_last = (freq_to_midi(freq) - last_accepted_midi) * 100.0
        else:
            cents_to_last = None
        cents_to_expected = (freq_to_midi(freq) - expected) * 100.0

        if cents_to_last is not None and abs(cents_to_last) <= tol_cents:
            # Retake of the last accepted note.
            assert last_accepted_label_index is not None
            labels[last_accepted_label_index] = LabeledSlice(
                i,
                midi=last_accepted_midi,
                detected_freq_hz=freq,
                detected_midi=det_midi,
                cents_off=cents,
                reason="retake_replaced",
            )
            # And mark this slice's row as a placeholder so indices align with the input.
            labels.append(
                LabeledSlice(
                    i,
                    midi=None,
                    detected_freq_hz=freq,
                    detected_midi=det_midi,
                    cents_off=cents,
                    reason="superseded_by_retake",
                )
            )
            # Swap: the *latter* slice should be the canonical one. Re-point.
            labels[last_accepted_label_index], labels[-1] = labels[-1], labels[last_accepted_label_index]
            # Now the LATER slice carries the accepted label; the EARLIER one is superseded.
            last_accepted_label_index = len(labels) - 1
            continue

        if abs(cents_to_expected) <= tol_cents:
            labels.append(
                LabeledSlice(
                    i,
                    midi=expected,
                    detected_freq_hz=freq,
                    detected_midi=det_midi,
                    cents_off=cents,
                    reason="accepted",
                )
            )
            last_accepted_midi = expected
            last_accepted_label_index = len(labels) - 1
            expected = min(expected + 1, high_midi + 1)
            continue

        # Maybe the operator skipped one (expected+1 is the next semitone above).
        if abs(cents_to_expected - 100.0) <= tol_cents and expected + 1 <= high_midi:
            new_note = expected + 1
            labels.append(
                LabeledSlice(
                    i,
                    midi=new_note,
                    detected_freq_hz=freq,
                    detected_midi=det_midi,
                    cents_off=cents,
                    reason="accepted",
                )
            )
            last_accepted_midi = new_note
            last_accepted_label_index = len(labels) - 1
            expected = new_note + 1
            continue

        # Doesn't match retake, expected, or expected+1. Reject.
        labels.append(
            LabeledSlice(
                i,
                midi=None,
                detected_freq_hz=freq,
                detected_midi=det_midi,
                cents_off=cents,
                reason="out_of_walk",
            )
        )

    return labels
