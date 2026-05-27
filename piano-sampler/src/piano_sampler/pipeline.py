"""Top-level pipeline orchestrator: input WAVs -> labeled samples + SFZ files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .decentsampler import DSInstrumentSpec, write_dspreset
from .edge_treatment import FadeConfig, apply_fades
from .labeler import label_slices
from .notes import midi_to_name
from .pitch import PitchConfig, score_midi_candidates
from .sfz import InstrumentSpec, Region, write_sfz
from .slice import SliceConfig, slice_file, to_mono
from .wav_io import read_wav, write_wav

# Matches Matt's naming convention:
#   "Bubby's Piano v1.0_Long_Quiet_RR 1.wav"
#   "Bubby's Piano v1.0_Short_Loud_RR 1.wav"
# Also accepts simpler "Long_Loud_RR1.wav" for test fixtures.
INPUT_RE = re.compile(
    r"(?P<artic>Long|Short)[_ ](?P<vel>Quiet|Loud)[_ ]RR[_ ]?(?P<rr>\d+)",
    re.IGNORECASE,
)


@dataclass
class InputFile:
    path: Path
    articulation: str  # "long" or "short"
    velocity: str  # "quiet" or "loud"
    rr: int


def discover_inputs(input_dir: Path) -> list[InputFile]:
    found: list[InputFile] = []
    for p in sorted(input_dir.glob("**/*.wav")):
        m = INPUT_RE.search(p.name)
        if not m:
            continue
        found.append(
            InputFile(
                path=p,
                articulation=m.group("artic").lower(),
                velocity=m.group("vel").lower(),
                rr=int(m.group("rr")),
            )
        )
    return found


@dataclass
class ProcessReport:
    input_path: Path
    articulation: str
    velocity: str
    rr: int
    n_slices: int
    n_accepted: int
    n_retakes: int
    n_rejected: int
    notes_covered: list[int]
    notes_missing: list[int]
    noise_floor_dbfs: float
    written_samples: list[str]  # relative paths of written WAVs


def _process_one(
    inp: InputFile,
    sample_out_dir: Path,
    samples_relpath_prefix: str,
    slice_cfg: SliceConfig,
    pitch_cfg: PitchConfig,
    fade_cfg: FadeConfig,
) -> tuple[ProcessReport, list[Region]]:
    samples, sr, _bd = read_wav(inp.path)
    mono = to_mono(samples)
    slices, noise_floor = slice_file(samples, sr, slice_cfg)
    slice_scores = [score_midi_candidates(mono[s.start : s.end], sr, cfg=pitch_cfg) for s in slices]
    labels = label_slices(slice_scores)

    written: list[str] = []
    regions: list[Region] = []
    notes_covered: list[int] = []
    for i, lbl in enumerate(labels):
        if lbl.midi is None:
            continue
        sl = slices[i]
        body = samples[sl.start : sl.end]
        body = apply_fades(body, sr, fade_cfg)
        name = midi_to_name(lbl.midi)
        # Use MIDI number as primary id, note name as readability suffix.
        fname = f"{lbl.midi:03d}_{name}_{inp.articulation}_{inp.velocity}_rr{inp.rr}.wav"
        out_path = sample_out_dir / fname
        write_wav(out_path, body, sr, bit_depth=24)
        rel = f"{samples_relpath_prefix}{fname}"
        written.append(rel)
        notes_covered.append(lbl.midi)
        regions.append(
            Region(
                sample_relpath=rel,
                midi=lbl.midi,
                velocity=inp.velocity,
                rr=inp.rr,
            )
        )

    expected_notes = set(range(21, 109))
    missing = sorted(expected_notes - set(notes_covered))

    nf_db = 20.0 * float(np.log10(max(noise_floor, 1e-12)))
    report = ProcessReport(
        input_path=inp.path,
        articulation=inp.articulation,
        velocity=inp.velocity,
        rr=inp.rr,
        n_slices=len(slices),
        n_accepted=sum(1 for l in labels if l.reason == "accepted"),
        n_retakes=sum(1 for l in labels if l.reason == "retake_replaced"),
        n_rejected=sum(1 for l in labels if l.midi is None),
        notes_covered=sorted(notes_covered),
        notes_missing=missing,
        noise_floor_dbfs=nf_db,
        written_samples=written,
    )
    return report, regions


def build(
    input_dir: Path,
    output_dir: Path,
    instrument_name: str,
    *,
    slice_cfg: SliceConfig | None = None,
    pitch_cfg: PitchConfig | None = None,
    fade_cfg: FadeConfig | None = None,
) -> dict:
    slice_cfg = slice_cfg or SliceConfig()
    pitch_cfg = pitch_cfg or PitchConfig()
    fade_cfg = fade_cfg or FadeConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "Samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(input_dir)
    if not inputs:
        raise SystemExit(f"no input WAVs found in {input_dir}")

    reports: list[ProcessReport] = []
    regions_by_artic: dict[str, list[Region]] = {"long": [], "short": []}
    for inp in inputs:
        rep, regs = _process_one(
            inp,
            sample_out_dir=samples_dir,
            samples_relpath_prefix="Samples/",
            slice_cfg=slice_cfg,
            pitch_cfg=pitch_cfg,
            fade_cfg=fade_cfg,
        )
        reports.append(rep)
        regions_by_artic[inp.articulation].extend(regs)

    # Sanitize for filenames: drop apostrophes/spaces.
    safe = instrument_name.replace("'", "").replace(" ", "")
    sustain_spec = InstrumentSpec(
        name=f"{safe}_Sustain",
        is_sustain=True,
        regions=regions_by_artic["long"],
        ampeg_release_seconds=1.2,
    )
    staccato_spec = InstrumentSpec(
        name=f"{safe}_Staccato",
        is_sustain=False,
        regions=regions_by_artic["short"],
        ampeg_release_seconds=0.2,
    )
    write_sfz(output_dir / f"{safe}_Sustain.sfz", sustain_spec)
    write_sfz(output_dir / f"{safe}_Staccato.sfz", staccato_spec)

    # DecentSampler presets (free VST3/AU, no import step, no Kontakt needed).
    write_dspreset(
        output_dir / f"{safe}_Sustain.dspreset",
        DSInstrumentSpec(name=f"{safe}_Sustain", regions=regions_by_artic["long"], release=1.2),
    )
    write_dspreset(
        output_dir / f"{safe}_Staccato.dspreset",
        DSInstrumentSpec(name=f"{safe}_Staccato", regions=regions_by_artic["short"], release=0.2),
    )

    summary = _write_summary(output_dir, instrument_name, reports)
    return summary


def _write_summary(output_dir: Path, instrument_name: str, reports: list[ProcessReport]) -> dict:
    summary = {
        "instrument_name": instrument_name,
        "files": [
            {
                "input": str(r.input_path.name),
                "articulation": r.articulation,
                "velocity": r.velocity,
                "rr": r.rr,
                "n_slices": r.n_slices,
                "n_accepted": r.n_accepted,
                "n_retakes": r.n_retakes,
                "n_rejected": r.n_rejected,
                "notes_covered": r.notes_covered,
                "notes_missing": [midi_to_name(n) for n in r.notes_missing],
                "noise_floor_dbfs": round(r.noise_floor_dbfs, 1),
            }
            for r in reports
        ],
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
