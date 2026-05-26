"""WAV read/write using stdlib `wave` + numpy. Supports 16/24/32-bit PCM, stereo or mono."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    """Read a PCM WAV. Returns (samples_float32 in [-1, 1], sample_rate, bit_depth).

    Shape: (num_samples,) for mono, (num_samples, num_channels) for multi-channel.
    """
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    bit_depth = sampwidth * 8

    if sampwidth == 2:
        ints = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit little-endian -> int32 -> float32
        b = np.frombuffer(raw, dtype=np.uint8)
        # Pad each 3-byte sample to 4 bytes (sign-extend the top byte).
        n_samples = len(b) // 3
        b4 = np.zeros(n_samples * 4, dtype=np.uint8)
        b3 = b.reshape(n_samples, 3)
        b4 = b4.reshape(n_samples, 4)
        b4[:, :3] = b3
        # Sign-extend: if high bit of top byte is set, fill byte 3 with 0xFF.
        b4[:, 3] = np.where(b3[:, 2] & 0x80, 0xFF, 0x00)
        ints = b4.view(np.int32).astype(np.float32).reshape(-1) / (2**23)
    elif sampwidth == 4:
        ints = np.frombuffer(raw, dtype="<i4").astype(np.float32) / (2**31)
    else:
        raise ValueError(f"unsupported sample width: {sampwidth} bytes")

    if n_channels > 1:
        ints = ints.reshape(-1, n_channels)
    return ints, sample_rate, bit_depth


def write_wav(path: Path, samples: np.ndarray, sample_rate: int, bit_depth: int = 24) -> None:
    """Write float samples in [-1, 1] to a PCM WAV.

    Accepts mono (1-D) or stereo (N, 2) float arrays.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if samples.ndim == 1:
        n_channels = 1
        flat = samples
    else:
        n_channels = samples.shape[1]
        flat = samples.reshape(-1)

    clipped = np.clip(flat, -1.0, 1.0)

    if bit_depth == 16:
        ints = (clipped * 32767.0).astype("<i2")
        raw = ints.tobytes()
        sampwidth = 2
    elif bit_depth == 24:
        scaled = (clipped * (2**23 - 1)).astype(np.int32)
        # int32 little-endian -> drop the high byte for 24-bit.
        b4 = scaled.view(np.uint8).reshape(-1, 4)
        raw = b4[:, :3].tobytes()
        sampwidth = 3
    elif bit_depth == 32:
        ints = (clipped * (2**31 - 1)).astype("<i4")
        raw = ints.tobytes()
        sampwidth = 4
    else:
        raise ValueError(f"unsupported bit depth: {bit_depth}")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)


def read_wav_header(path: Path) -> tuple[int, int, int, int]:
    """Return (n_channels, bit_depth, sample_rate, n_frames) without loading samples."""
    with wave.open(str(path), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth() * 8, wf.getframerate(), wf.getnframes()
