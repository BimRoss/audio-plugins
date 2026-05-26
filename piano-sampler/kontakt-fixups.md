# Kontakt 6 fixups (post-SFZ-import checklist)

Kontakt 6's SFZ importer is partial. After `File → Import → SFZ`, walk this checklist before saving the .nki. Each instrument (Sustain, Staccato) gets the same treatment.

> If the SFZ sounds correct in [sfizz](https://sfz.tools/sfizz/) and wrong in Kontakt, the gap is one of the opcodes below.

## 1. Round robins

Kontakt's import often lands each `seq_position` region as its own overlapping group → all RRs trigger at once.

**Fix:** in the Group Editor, set `Group Start` rules:
- Round Robin → cycle through the per-RR groups.
- `seq_length=2` groups (quiet layers) → 2 RR slots.
- `seq_length=1` groups (loud layers) → single, no cycling.

## 2. Velocity crossfade (xfin / xfout)

Kontakt's import drops `xfin_*` / `xfout_*` opcodes.

**Fix:** in the Group Editor, on the Quiet and Loud groups for both Sustain and Staccato:
- Quiet group: `Vel Range 0–120`, `Crossfade Out` from 100 → 120.
- Loud group: `Vel Range 100–127`, `Crossfade In` from 100 → 120.

## 3. Sustain pedal (CC64)

`sustain_sw=on` may not be honored. Pedal behavior must be wired manually for the Sustain instrument.

**Fix:** in Instrument Options → Engine, enable "Accept MIDI CC". Then in the Group Editor for the Sustain group:
- Add `Group Start` condition: `Released Key` → Off.
- Add CC64 handling so note-off is deferred until CC64 < 64. (KSP one-liner if needed; standard "Damper Pedal" preset works on most factory templates.)

Staccato instrument: no pedal behavior needed.

## 4. Envelopes

Confirm `ampeg_release` imported correctly:
- Sustain group: ~1.2 s.
- Staccato group: ~0.2 s.

If Kontakt shows AHDSR defaults (instant attack, no release), set them manually on the Amp envelope in the Group Editor.

## 5. Output stage — no processing

Belt-and-braces sanity check:
- Insert FX on the instrument → empty.
- Group FX → empty.
- Output gain → 0 dB.
- Tuning → 0 cents (we are not detuning to fake RRs).

## 6. Save

`File → Save as...` → `BubbysPiano_Sustain.nki` / `BubbysPiano_Staccato.nki`, **alongside the `Samples/` folder** so relative paths resolve.

Test: load the saved .nki into a fresh Kontakt instance, confirm samples resolve and behavior matches the SFZ in sfizz.
