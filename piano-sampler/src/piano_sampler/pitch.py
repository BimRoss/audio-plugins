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
    tolerance_cents: float = 80.0  # how far a detected pitch may sit from a MIDI note (piano stretch tuning at extremes + detector noise)
    yin_threshold: float = 0.15  # CMNDF absolute threshold for accepting a τ (0.10-0.20 typical)


def slice_spectrum(
    mono_slice: np.ndarray,
    sample_rate: int,
    cfg: PitchConfig | None = None,
) -> tuple[np.ndarray, float] | None:
    """Return (magnitude spectrum, bin_hz) of the highest-RMS sub-window of the slice."""
    cfg = cfg or PitchConfig()
    offset = int(cfg.analysis_offset_seconds * sample_rate)
    window = int(cfg.analysis_window_seconds * sample_rate)
    seg_end = mono_slice.size
    if seg_end <= offset + 256:
        return None
    if seg_end - offset <= window:
        seg = mono_slice[offset:seg_end]
    else:
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
    seg = seg - seg.mean()
    if float((seg**2).sum()) < 1e-8:
        return None
    n_fft = 1 << int(np.ceil(np.log2(seg.size * 2)))
    seg_w = seg * np.hanning(seg.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(seg_w, n=n_fft))
    bin_hz = sample_rate / n_fft
    return spec, bin_hz


def harmonic_score(spec: np.ndarray, bin_hz: float, freq_hz: float, n_harmonics: int = 6) -> float:
    """Sum log-magnitudes at harmonics k*f0 for k=1..n. Higher = better fit for f0."""
    max_bin = len(spec) - 1
    score = 0.0
    for k in range(1, n_harmonics + 1):
        b = int(round(freq_hz * k / bin_hz))
        if b > max_bin:
            break
        # Pick the peak within ±1 bin for robustness to bin quantization.
        lo = max(0, b - 1)
        hi = min(max_bin, b + 1)
        score += float(np.log1p(spec[lo : hi + 1].max()))
    return score


def score_midi_candidates(
    mono_slice: np.ndarray,
    sample_rate: int,
    *,
    midi_range: tuple[int, int] = (21, 108),
    cfg: PitchConfig | None = None,
) -> dict[int, float] | None:
    """Return {midi_note: harmonic_score} for every MIDI in the requested range.

    This is the input to the DP labeler — the labeler picks the slice→note
    assignment that maximizes total score subject to the monotonic A0..C8
    constraint. Returning per-MIDI scores (instead of a single best f0) lets
    the chromatic-order prior break octave ambiguity at assignment time.
    """
    res = slice_spectrum(mono_slice, sample_rate, cfg)
    if res is None:
        return None
    spec, bin_hz = res
    out: dict[int, float] = {}
    for midi in range(midi_range[0], midi_range[1] + 1):
        f0 = 440.0 * (2 ** ((midi - 69) / 12))
        out[midi] = harmonic_score(spec, bin_hz, f0)
    return out


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
    seg = seg - seg.mean()
    if float((seg**2).sum()) < 1e-8:
        return None

    # ---- Harmonic-summation pitch detection ----
    # For each candidate f0 in [min_freq, max_freq] (logarithmic 1-cent steps),
    # score = Σ_{k=1..N} log(1 + spec[k*f0]). The candidate whose harmonic stack
    # accumulates the most spectral energy wins. This explicitly handles the
    # "missing fundamental" case on real piano bass strings, where YIN and
    # plain autocorrelation pick a strong upper partial instead of f0.
    n_fft = 1 << int(np.ceil(np.log2(seg.size * 2)))
    # Apply a Hann window so we get cleaner peaks.
    seg_w = seg * np.hanning(seg.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(seg_w, n=n_fft))
    bin_hz = sample_rate / n_fft
    max_bin = len(spec) - 1

    # Candidate f0 grid: 1 cent spacing across [min_freq, max_freq].
    f_lo = max(cfg.min_freq_hz, 20.0)
    f_hi = min(cfg.max_freq_hz, sample_rate / 2.5)
    n_steps = int(np.ceil(np.log2(f_hi / f_lo) * 1200))  # one step per cent
    cents_grid = np.arange(n_steps) / 1200.0
    candidates = f_lo * (2.0 ** cents_grid)

    n_harmonics = 8
    # Build score by summing log(1 + spec[bin]) at each harmonic.
    scores = np.zeros_like(candidates)
    for k in range(1, n_harmonics + 1):
        bins = np.round(candidates * k / bin_hz).astype(np.int64)
        bins = np.clip(bins, 0, max_bin)
        scores += np.log1p(spec[bins])

    # Penalize very-high candidates whose first harmonic exceeds Nyquist coverage.
    # And require min two strong harmonics in spec to call it a pitch.
    best_idx = int(np.argmax(scores))
    best_f0 = float(candidates[best_idx])
    best_score = float(scores[best_idx])
    if best_score < 0.5:  # essentially no harmonic content
        return None

    return best_f0


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

        # Matt played all 88 chromatically with no skips — so if the detected
        # pitch doesn't match expected or last (retake), we reject. Even if it
        # looks like expected+1 it's safer to wait for the true expected match;
        # accepting expected+1 here would orphan the real expected slice later.
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
