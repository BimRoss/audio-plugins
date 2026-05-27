"""DP-based labeler: assign N candidate slices to the chromatic sequence A0..C8.

Input: per-slice score_midi_candidates output — a dict {midi_note: score} per
slice, where `score` is harmonic-summation goodness-of-fit for `slice` matching
`midi_note`.

Algorithm: find the assignment of slices → MIDI notes (A0..C8) that maximizes
total score, subject to:

1. Each MIDI note in [low_midi, high_midi] is assigned exactly one slice.
2. Time order is preserved (slice index for note m+1 > slice index for note m).
3. Retake rule ("keep last") is applied by marking earlier same-note matches
   as `superseded_by_retake` in the output.

DP recurrence:
    dp[n_off][i] = max total score for notes [0..n_off] using slice i as the
                   canonical slice for note (low_midi + n_off).
    dp[n_off][i] = max over i' < i of (dp[n_off - 1][i'] + score[i][n_off])
    Base: dp[0][i] = score[i][0].

O(notes × slices × slices), trivial for our sizes (~88 × 126 × 126 = 1.4M ops).
"""

from __future__ import annotations

from dataclasses import dataclass

from .notes import PIANO_HIGHEST, PIANO_LOWEST


@dataclass
class Assignment:
    slice_index: int
    midi: int | None
    score: float | None
    reason: str  # "accepted", "retake_replaced", "superseded_by_retake", "non_pitched"


def label_slices(
    slice_scores: list[dict[int, float] | None],
    *,
    low_midi: int = PIANO_LOWEST,
    high_midi: int = PIANO_HIGHEST,
    min_score: float = 1.0,
) -> list[Assignment]:
    """DP-assign K slices to the chromatic A0..C8 sequence using per-MIDI scores."""
    n_slices = len(slice_scores)
    n_notes = high_midi - low_midi + 1

    inf_neg = float("-inf")

    # Build a (n_slices x n_notes) matrix of scores. Slices with no pitch info
    # (None) get -inf for every note (cannot be canonical for any note).
    score: list[list[float]] = [[inf_neg] * n_notes for _ in range(n_slices)]
    for i, sc in enumerate(slice_scores):
        if sc is None:
            continue
        for n_off in range(n_notes):
            midi = low_midi + n_off
            score[i][n_off] = sc.get(midi, inf_neg)

    dp: list[list[float]] = [[inf_neg] * n_slices for _ in range(n_notes)]
    parent: list[list[int]] = [[-1] * n_slices for _ in range(n_notes)]

    # Base: note 0 = low_midi.
    for i in range(n_slices):
        if score[i][0] > inf_neg:
            dp[0][i] = score[i][0]

    # Forward: prefix max over dp[n_off-1] for fast transition.
    for n_off in range(1, n_notes):
        best_so_far = inf_neg
        best_idx_so_far = -1
        prefix_max: list[tuple[float, int]] = [(inf_neg, -1)] * n_slices
        for i in range(n_slices):
            if i > 0:
                v = dp[n_off - 1][i - 1]
                if v > best_so_far:
                    best_so_far = v
                    best_idx_so_far = i - 1
            prefix_max[i] = (best_so_far, best_idx_so_far)

        for i in range(n_slices):
            if score[i][n_off] == inf_neg:
                continue
            prev, prev_idx = prefix_max[i]
            if prev == inf_neg:
                continue
            total = prev + score[i][n_off]
            if total > dp[n_off][i]:
                dp[n_off][i] = total
                parent[n_off][i] = prev_idx

    # Find best terminal.
    best_end_i = -1
    best_end_score = inf_neg
    for i in range(n_slices):
        if dp[n_notes - 1][i] > best_end_score:
            best_end_score = dp[n_notes - 1][i]
            best_end_i = i

    chosen_slice_for_note: list[int] = [-1] * n_notes
    if best_end_i >= 0:
        cur = best_end_i
        for n_off in range(n_notes - 1, -1, -1):
            chosen_slice_for_note[n_off] = cur
            if n_off > 0:
                cur = parent[n_off][cur]

    canonical_slice_to_midi: dict[int, int] = {}
    for n_off, sl in enumerate(chosen_slice_for_note):
        if sl >= 0:
            canonical_slice_to_midi[sl] = low_midi + n_off

    # Retake marking: if an earlier (non-canonical) slice has a high score for
    # the same MIDI as a later canonical slice, mark it superseded.
    superseded: set[int] = set()
    for i, sc in enumerate(slice_scores):
        if sc is None or i in canonical_slice_to_midi:
            continue
        for canon_slice, canon_midi in canonical_slice_to_midi.items():
            if canon_slice <= i:
                continue
            if sc.get(canon_midi, inf_neg) >= min_score:
                superseded.add(i)
                break

    out: list[Assignment] = []
    for i, sc in enumerate(slice_scores):
        if i in canonical_slice_to_midi:
            m = canonical_slice_to_midi[i]
            sval = sc.get(m, None) if sc is not None else None
            # Tag as retake_replaced only when an earlier slice was a STRONG
            # candidate for this same MIDI — specifically, scored at least
            # 80% as well as the canonical pick. This avoids tagging every
            # bass note as a retake just because the harmonic-summation
            # detector assigns nonzero scores to many neighboring slices.
            had_earlier = False
            canonical_score = sval if sval is not None else inf_neg
            for j in range(i):
                if j in canonical_slice_to_midi:
                    continue
                js = slice_scores[j]
                if js is None:
                    continue
                if js.get(m, inf_neg) >= max(min_score, 0.8 * canonical_score):
                    had_earlier = True
                    break
            out.append(
                Assignment(
                    slice_index=i,
                    midi=m,
                    score=sval,
                    reason="retake_replaced" if had_earlier else "accepted",
                )
            )
        elif i in superseded:
            out.append(Assignment(slice_index=i, midi=None, score=None, reason="superseded_by_retake"))
        else:
            out.append(
                Assignment(
                    slice_index=i,
                    midi=None,
                    score=None,
                    reason="non_pitched" if sc is None else "out_of_walk",
                )
            )

    return out
