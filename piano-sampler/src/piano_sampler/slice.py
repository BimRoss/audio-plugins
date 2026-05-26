"""Onset-based slicing with per-file adaptive noise floor.

We do NOT try to be clever about silence; we identify *onsets* (transients) and
treat each onset as the start of a slice. Each slice runs from its onset to the
next onset (or EOF), then has trailing silence trimmed using a noise-floor +
margin threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Slice:
    start: int  # inclusive sample index in the source file (mono mixdown reference)
    end: int  # exclusive sample index after trailing-silence trim


@dataclass
class SliceConfig:
    frame_size: int = 1024  # samples per analysis frame
    hop: int = 256  # frame hop (analysis stride)
    onset_jump_db: float = 18.0  # current frame RMS must exceed past-window RMS by this much
    onset_abs_margin_db: float = 12.0  # absolute floor: rms must also exceed noise_floor + this
    onset_lookback_seconds: float = 0.20  # how far back the comparison window starts
    onset_window_seconds: float = 0.10  # length of the past-comparison window
    onset_min_gap_seconds: float = 2.0  # minimum spacing between accepted onsets
    onset_snap_back_seconds: float = 0.02  # capture pre-attack air on the slice start
    trim_margin_db: float = 6.0  # trailing-silence threshold above noise floor
    trim_min_silence_seconds: float = 0.3  # silence duration to confirm "end of note"
    noise_floor_seconds: float = 1.0  # legacy; current measure_noise_floor uses percentile


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Reduce stereo (or N-channel) to mono by averaging."""
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=1).astype(np.float32)


def frame_rms(mono: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    """Per-frame RMS over a 1-D mono signal."""
    if mono.size < frame_size:
        return np.array([], dtype=np.float32)
    n_frames = 1 + (mono.size - frame_size) // hop
    out = np.empty(n_frames, dtype=np.float32)
    sq = mono.astype(np.float32) ** 2
    for i in range(n_frames):
        s = i * hop
        out[i] = float(np.sqrt(sq[s : s + frame_size].mean() + 1e-20))
    return out


def measure_noise_floor(
    mono: np.ndarray,
    sample_rate: int,
    seconds: float,
    cfg: SliceConfig | None = None,
) -> float:
    """Robustly estimate the noise floor as the 10th-percentile frame RMS.

    Using a percentile of frame RMS makes us robust to operators who hit record
    and immediately played, leaving no clean pre-roll. The bottom 10% of frames
    are guaranteed to be in the gaps between notes (or before the first note).
    """
    cfg = cfg or SliceConfig()
    rms = frame_rms(mono, cfg.frame_size, cfg.hop)
    if rms.size == 0:
        return 1e-6
    return float(np.percentile(rms, 10.0))


def db(x: float) -> float:
    return 20.0 * float(np.log10(max(x, 1e-12)))


def find_onsets(
    mono: np.ndarray,
    sample_rate: int,
    noise_floor: float,
    cfg: SliceConfig,
) -> list[int]:
    """Detect note onsets as transient rises in frame RMS.

    Each frame's RMS is compared to the mean RMS of a short past-window placed
    `onset_lookback_seconds` ago. A frame is an onset when:

      - rms[i] > k_jump * past_rms[i]   (sudden rise; piano attack)
      - rms[i] > noise_floor * k_abs    (above pure-noise threshold)
      - at least `onset_min_gap_seconds` since the previous accepted onset

    This rejects sustained-but-not-attacking frames (note bodies, mechanical
    noise mid-decay) and avoids splitting a single note into many onsets.
    """
    rms = frame_rms(mono, cfg.frame_size, cfg.hop)
    if rms.size == 0:
        return []

    lookback_frames = max(1, int(cfg.onset_lookback_seconds * sample_rate / cfg.hop))
    window_frames = max(1, int(cfg.onset_window_seconds * sample_rate / cfg.hop))
    gap_frames = max(1, int(cfg.onset_min_gap_seconds * sample_rate / cfg.hop))
    k_jump = 10 ** (cfg.onset_jump_db / 20.0)
    k_abs = 10 ** (cfg.onset_abs_margin_db / 20.0)
    abs_threshold = noise_floor * k_abs

    # Past-window mean (frames [i - lookback - window, i - lookback)).
    past = np.empty_like(rms)
    for i in range(len(rms)):
        lo = max(0, i - lookback_frames - window_frames)
        hi = max(0, i - lookback_frames)
        past[i] = float(rms[lo:hi].mean()) if hi - lo >= 2 else noise_floor

    onsets: list[int] = []
    last = -gap_frames - 1
    for i in range(len(rms)):
        if rms[i] < abs_threshold:
            continue
        if rms[i] > k_jump * past[i] and (i - last) >= gap_frames:
            onsets.append(i)
            last = i

    snap = int(cfg.onset_snap_back_seconds * sample_rate)
    return [max(0, idx * cfg.hop - snap) for idx in onsets]


def trim_trailing_silence(
    mono: np.ndarray,
    start: int,
    end: int,
    sample_rate: int,
    noise_floor: float,
    cfg: SliceConfig,
) -> int:
    """Walk backward from `end` and stop where the signal last exceeds noise_floor + margin."""
    threshold = noise_floor * (10 ** (cfg.trim_margin_db / 20.0))
    if end <= start:
        return end
    # Coarse scan in hop-sized windows for speed, then refine.
    win = cfg.frame_size
    hop = cfg.hop
    # Find the last hop-window whose RMS is above threshold.
    last_loud_end = start
    for s in range(start, end - win, hop):
        seg = mono[s : s + win]
        if float(np.sqrt(np.mean(seg.astype(np.float32) ** 2) + 1e-20)) > threshold:
            last_loud_end = s + win
    # Keep at least trim_min_silence_seconds of silence after the last loud window
    # (i.e. the audible tail can taper into noise without being cut sharp).
    min_silence = int(cfg.trim_min_silence_seconds * sample_rate)
    return min(end, last_loud_end + min_silence)


def slice_file(
    samples: np.ndarray,
    sample_rate: int,
    cfg: SliceConfig | None = None,
) -> tuple[list[Slice], float]:
    """Onset-based slicing. Returns (slices, measured_noise_floor)."""
    cfg = cfg or SliceConfig()
    mono = to_mono(samples)
    noise_floor = measure_noise_floor(mono, sample_rate, cfg.noise_floor_seconds, cfg)
    onsets = find_onsets(mono, sample_rate, noise_floor, cfg)
    slices: list[Slice] = []
    for i, start in enumerate(onsets):
        end = onsets[i + 1] if i + 1 < len(onsets) else mono.size
        end = trim_trailing_silence(mono, start, end, sample_rate, noise_floor, cfg)
        if end > start:
            slices.append(Slice(start=start, end=end))
    return slices, noise_floor
