"""Head fade-in + tail fade-out at slice boundaries.

The body of every slice is otherwise untouched (no EQ, no compression, no
normalization, no denoise). The head fade is purely to kill boundary pops; the
tail fade is purely to ensure zero-crossing termination at file-end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FadeConfig:
    head_seconds: float = 0.002  # 2 ms linear ramp in
    tail_seconds: float = 0.080  # 80 ms linear ramp out at file end


def apply_fades(samples: np.ndarray, sample_rate: int, cfg: FadeConfig | None = None) -> np.ndarray:
    """Return a copy of `samples` with head + tail fades applied."""
    cfg = cfg or FadeConfig()
    out = samples.astype(np.float32, copy=True)
    n = out.shape[0]

    head = int(cfg.head_seconds * sample_rate)
    tail = int(cfg.tail_seconds * sample_rate)
    if head > 0 and head < n:
        ramp = np.linspace(0.0, 1.0, head, dtype=np.float32)
        if out.ndim == 1:
            out[:head] *= ramp
        else:
            out[:head] *= ramp[:, None]
    if tail > 0 and tail < n:
        ramp = np.linspace(1.0, 0.0, tail, dtype=np.float32)
        if out.ndim == 1:
            out[-tail:] *= ramp
        else:
            out[-tail:] *= ramp[:, None]
    return out
