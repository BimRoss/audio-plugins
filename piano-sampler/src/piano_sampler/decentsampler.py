"""DecentSampler .dspreset generator.

DecentSampler (https://www.decentsamples.com/) is a free VST3/AU/standalone
sampler. Its preset format is plain XML, so we can generate it directly — no
import step, no Kontakt-version games (Kontakt 6 dropped SFZ import entirely).

Round-robin: each <sample> carries seqMode="round_robin" + seqPosition. Samples
overlapping in key+velocity with different seqPosition cycle.

Velocity: clean split — quiet layer loVel 0..velocity_split, loud layer
velocity_split+1..127. Velocity drives amplitude within each layer natively
(DecentSampler scales sample gain by velocity), which gives Matt the
"quiet RRs reduce in volume proportionally" behavior he asked for.

Sustain pedal: DecentSampler responds to CC64 (damper) natively — no config
needed. The release envelope governs how notes taper on note-off / pedal-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

from .sfz import Region


@dataclass
class DSInstrumentSpec:
    name: str
    regions: list[Region] = field(default_factory=list)
    attack: float = 0.001
    decay: float = 0.5
    sustain: float = 1.0  # level 0..1
    release: float = 1.2  # seconds
    velocity_split: int = 110  # quiet 0..split, loud split+1..127


def _sample_el(r: Region, lovel: int, hivel: int, seq_len: int) -> str:
    attrs = {
        "path": r.sample_relpath,
        "rootNote": str(r.midi),
        "loNote": str(r.midi),
        "hiNote": str(r.midi),
        "loVel": str(lovel),
        "hiVel": str(hivel),
        "seqMode": "round_robin",
        "seqPosition": str(r.rr),
        "seqLength": str(seq_len),
    }
    parts = " ".join(f"{k}={quoteattr(v)}" for k, v in attrs.items())
    return f"      <sample {parts} />"


def render_dspreset(spec: DSInstrumentSpec) -> str:
    quiet = [r for r in spec.regions if r.velocity == "quiet"]
    loud = [r for r in spec.regions if r.velocity == "loud"]
    quiet_seq_len = max((r.rr for r in quiet), default=1)
    loud_seq_len = max((r.rr for r in loud), default=1)

    env = f'attack="{spec.attack}" decay="{spec.decay}" sustain="{spec.sustain}" release="{spec.release}"'

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(f"<DecentSampler minVersion=\"1.0.0\">")
    lines.append("  <groups>")

    lines.append(f'    <group name="Quiet" {env}>')
    for r in sorted(quiet, key=lambda x: (x.midi, x.rr)):
        lines.append(_sample_el(r, 0, spec.velocity_split, quiet_seq_len))
    lines.append("    </group>")

    lines.append(f'    <group name="Loud" {env}>')
    for r in sorted(loud, key=lambda x: (x.midi, x.rr)):
        lines.append(_sample_el(r, spec.velocity_split + 1, 127, loud_seq_len))
    lines.append("    </group>")

    lines.append("  </groups>")
    lines.append("</DecentSampler>")
    return "\n".join(lines) + "\n"


def write_dspreset(path: Path, spec: DSInstrumentSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dspreset(spec))
