"""Turn a raw generated MIDI into a coherent arrangement.

Why this stage exists
---------------------
The model writes tokens, not a score. Even when the notes are valid, the
*arrangement* is often not something a person would play:

* 10-18 instruments for an intimate ballad, several with two or three notes;
* a bass line sitting above the melody, or a lead spread over four octaves;
* notes outside the detected key (the model samples pitch tokens freely);
* every instrument at the same velocity, so nothing sits in the background;
* duplicate notes on the same pitch and start time (phasey doubling);
* monophonic lines that overlap themselves (stuck/retriggered notes).

None of that is fixed by more training alone, and all of it makes a render
sound like "MIDI" rather than music. This module applies the fixes a human
arranger would make before handing the score to a renderer:

1. **Detect the key** (Krumhansl-Schmuckler pitch-class correlation).
2. **Prune to roles** - keep the lead, bass, pad, drums and a small number
   of supporting parts; drop one-note tracks and duplicates.
3. **Snap out-of-key notes** to the nearest scale tone.
4. **Fold ranges** - bass below the melody, lead in a singable register.
5. **Balance velocities** by role so the lead and drums sit forward.
6. **De-duplicate** notes and stop a monophonic lead from overlapping.

Everything is deterministic and offline; the report says exactly what was
changed so the transformation is auditable rather than magic.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("promptmusic.arrange")

__all__ = [
    "ArrangementReport",
    "estimate_key",
    "arrange",
    "arrange_midi",
    "MAJOR_PROFILE",
    "MINOR_PROFILE",
]

#: Krumhansl-Kessler key profiles (perceived stability of each scale degree).
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

#: Scale degrees allowed per mode. Minor includes the raised 7th so dominant
#: chords (harmonic minor) are not "corrected" into something weaker.
_SCALES = {
    "major": {0, 2, 4, 5, 7, 9, 11},
    "minor": {0, 2, 3, 5, 7, 8, 10, 11},
}

_PAD_PROGRAMS = {48, 49, 50, 51, 52, 53, 88, 89, 90, 91, 94, 95}
_BASS_PROGRAMS = {32, 33, 34, 35, 36, 37, 38, 39}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class ArrangementReport:
    """What :func:`arrange` changed."""

    key: str = "C major"
    key_confidence: float = 0.0
    instruments_before: int = 0
    instruments_after: int = 0
    notes_before: int = 0
    notes_after: int = 0
    kept: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    snapped: int = 0
    folded: int = 0
    deduped: int = 0
    sustained: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "key_confidence": round(self.key_confidence, 3),
            "instruments_before": self.instruments_before,
            "instruments_after": self.instruments_after,
            "notes_before": self.notes_before,
            "notes_after": self.notes_after,
            "kept": self.kept,
            "removed": self.removed,
            "snapped": self.snapped,
            "folded": self.folded,
            "deduped": self.deduped,
            "sustained": self.sustained,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _label(index: int, inst: Any) -> str:
    name = inst.name if getattr(inst, "name", None) else f"program {inst.program}"
    return f"{index}:{name}({inst.program})"


def _is_drum(inst: Any) -> bool:
    return bool(getattr(inst, "is_drum", False))


def _median(values: list[float], default: float = 0.0) -> float:
    return float(np.median(values)) if values else default


def _stats(inst: Any) -> dict[str, float]:
    notes = list(inst.notes)
    if not notes:
        return {"count": 0, "pitch": 60.0, "duration": 0.0, "polyphony": 0.0, "span": 0.0}
    start = min(n.start for n in notes)
    end = max(n.end for n in notes)
    span = max(end - start, 1e-6)
    return {
        "count": float(len(notes)),
        "pitch": _median([n.pitch for n in notes], 60.0),
        "duration": _median([n.end - n.start for n in notes], 0.0),
        "polyphony": sum(max(0.0, n.end - n.start) for n in notes) / span,
        "span": span,
    }


def _fold_into(pitch: int, low: int, high: int) -> tuple[int, bool]:
    """Move a pitch by whole octaves into ``[low, high]``."""
    original = pitch
    for _ in range(10):
        if pitch < low:
            pitch += 12
        elif pitch > high:
            pitch -= 12
        else:
            break
    pitch = int(np.clip(pitch, 0, 127))
    return pitch, pitch != original


# --------------------------------------------------------------------------- #
# Key detection
# --------------------------------------------------------------------------- #
def estimate_key(pm: Any) -> tuple[int, str, float]:
    """Return ``(tonic_pitch_class, "major"|"minor", confidence)``.

    Pitch classes are weighted by note duration, so a sustained tonic/dominant
    counts more than a fast passing note. Confidence is the Pearson
    correlation of the histogram with the winning profile.
    """
    hist = np.zeros(12, dtype=np.float64)
    for inst in pm.instruments:
        if _is_drum(inst):
            continue
        for n in inst.notes:
            hist[n.pitch % 12] += max(0.05, float(n.end - n.start))
    if hist.sum() <= 0:
        return 0, "major", 0.0
    best_score, best_tonic, best_mode = -2.0, 0, "major"
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            rotated = np.roll(profile, tonic)
            if hist.std() < 1e-9:
                score = 0.0
            else:
                score = float(np.corrcoef(hist, rotated)[0, 1])
            if score > best_score:
                best_score, best_tonic, best_mode = score, tonic, mode
    return best_tonic, best_mode, best_score


# --------------------------------------------------------------------------- #
# Role selection
# --------------------------------------------------------------------------- #
def _choose_roles(
    instruments: list[tuple[int, Any]], max_instruments: int
) -> dict[int, str]:
    """Pick lead / bass / pad / supporting indices for non-drum tracks."""
    if not instruments:
        return {}
    stats = {idx: _stats(inst) for idx, inst in instruments}

    def lead_score(idx: int) -> float:
        s = stats[idx]
        if s["count"] < 3:
            return -1e9
        mono = 1.0 if s["polyphony"] <= 1.4 else max(0.35, 1.4 / max(s["polyphony"], 0.1))
        return s["count"] * mono - 0.6 * abs(s["pitch"] - 67.0)

    lead = max((idx for idx, _ in instruments), key=lead_score)
    roles: dict[int, str] = {lead: "lead"}

    # Bass: a real bass program when one has enough notes; otherwise the
    # lowest-register monophonic part, but only if it really is low (so a
    # piano is not folded into a bass line).
    low = [
        (idx, inst)
        for idx, inst in instruments
        if idx != lead and stats[idx]["count"] >= 3
    ]
    bass_idx: int | None = None
    bass_programs = [(i, inst) for i, inst in low if inst.program in _BASS_PROGRAMS]
    if bass_programs:
        bass_idx = max(bass_programs, key=lambda p: stats[p[0]]["count"])[0]
    else:
        playable = [
            (i, inst)
            for i, inst in low
            if stats[i]["pitch"] <= 57.0 and stats[i]["polyphony"] <= 2.0
        ]
        if playable:
            bass_idx = min(playable, key=lambda p: stats[p[0]]["pitch"])[0]
    if bass_idx is not None:
        roles[bass_idx] = "bass"

    # Pad: prefer strings/choir/pad programs even when sparse, then the part
    # with the longest sustained notes.
    remaining = [(i, inst) for i, inst in instruments if i not in roles and stats[i]["count"] >= 2]
    pad_idx: int | None = None
    pad_programs = [(i, inst) for i, inst in remaining if inst.program in _PAD_PROGRAMS]
    if pad_programs:
        pad_idx = max(pad_programs, key=lambda p: stats[p[0]]["count"])[0]
    elif remaining:
        sustained = [(i, inst) for i, inst in remaining if stats[i]["duration"] >= 0.35]
        if sustained:
            pad_idx = max(sustained, key=lambda p: stats[p[0]]["duration"] * stats[p[0]]["count"])[0]
    if pad_idx is not None:
        roles[pad_idx] = "pad"

    # Fill the rest by note count.
    remaining = sorted(
        [(i, inst) for i, inst in instruments if i not in roles],
        key=lambda p: stats[p[0]]["count"],
        reverse=True,
    )
    for idx, _ in remaining:
        if len(roles) >= max_instruments:
            break
        if stats[idx]["count"] < 3:
            continue
        roles[idx] = "other"

    # Cap at the requested size, dropping "other" first, then pad.
    if len(roles) > max_instruments:
        order = {"lead": 0, "bass": 1, "pad": 2, "other": 3}
        keep = sorted(roles.items(), key=lambda kv: order.get(kv[1], 9))[:max_instruments]
        roles = dict(keep)
    return roles


# --------------------------------------------------------------------------- #
# Note lengths
# --------------------------------------------------------------------------- #
#: ``(max_hold_seconds, release_gap_seconds)`` per role. The model almost
#: always emits the shortest duration token (a 16th note, ~0.12 s), which
#: makes every instrument sound like a plucked blip. Real players hold notes
#: until the next one (legato) or for the length of a chord.
_SUSTAIN = {
    "lead": (2.0, 0.02),
    "bass": (1.5, 0.03),
    "pad": (6.0, 0.05),
    "other": (1.5, 0.04),
}


def _sustain_notes(inst: Any, role: str, end_of_piece: float) -> int:
    """Lengthen notes towards the next onset. Returns how many changed."""
    limit, gap = _SUSTAIN.get(role, (1.5, 0.04))
    notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
    if not notes:
        return 0
    onsets = sorted({round(n.start, 4) for n in notes})
    changed = 0
    for note in notes:
        # The next onset *after* this note's start (chords share an onset).
        idx = bisect.bisect_right(onsets, round(note.start, 4) + 1e-4)
        target = onsets[idx] - gap if idx < len(onsets) else end_of_piece
        candidate = min(note.start + limit, max(note.end, target))
        if candidate > note.end + 1e-4:
            note.end = candidate
            changed += 1
    return changed


# --------------------------------------------------------------------------- #
# Main transform
# --------------------------------------------------------------------------- #
def arrange(
    pm: Any,
    *,
    max_instruments: int = 6,
    snap_key: bool = True,
    balance: bool = True,
    fold_ranges: bool = True,
    dedupe: bool = True,
    sustain: bool = True,
) -> tuple[Any, ArrangementReport]:
    """Clean a ``pretty_midi.PrettyMIDI`` and return ``(new_pm, report)``.

    The input is not mutated. Tempo and time signatures are preserved, so the
    output renders at the same length as the original.
    """
    import pretty_midi

    report = ArrangementReport()
    report.instruments_before = len(pm.instruments)
    report.notes_before = sum(len(i.notes) for i in pm.instruments)
    tonic, mode, confidence = estimate_key(pm)
    report.key = f"{_NOTE_NAMES[tonic]} {mode}"
    report.key_confidence = confidence
    if confidence < 0.25:
        report.warnings.append(
            f"weak key detection (r={confidence:.2f}); key snapping may be unreliable"
        )
    scale = _SCALES[mode]

    # --- choose which instruments survive --------------------------------- #
    non_drum = [(i, inst) for i, inst in enumerate(pm.instruments) if not _is_drum(inst)]
    drums = [(i, inst) for i, inst in enumerate(pm.instruments) if _is_drum(inst)]
    roles = _choose_roles(non_drum, max(1, max_instruments))

    kept_roles: dict[int, str] = dict(roles)
    # Keep at most two drum tracks that actually contain notes.
    for k, (idx, inst) in enumerate(drums):
        if k < 2 and inst.notes:
            kept_roles[idx] = "drums"

    kept_indices = set(kept_roles)

    # --- build the cleaned copy ------------------------------------------- #
    _, tempos = pm.get_tempo_changes()
    valid = [float(t) for t in np.atleast_1d(tempos) if 20.0 <= float(t) <= 400.0]
    tempo = valid[0] if valid else 120.0
    out = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    out.time_signature_changes = list(getattr(pm, "time_signature_changes", []))

    lead_idx = next((i for i, r in kept_roles.items() if r == "lead"), None)
    new_instruments: dict[int, Any] = {}
    for idx in sorted(kept_indices):
        src = pm.instruments[idx]
        role = kept_roles[idx]
        inst = pretty_midi.Instrument(program=int(src.program), is_drum=_is_drum(src), name=src.name)
        seen: set[tuple[int, float]] = set()
        kept_notes = []
        for n in sorted(src.notes, key=lambda n: (n.start, n.pitch)):
            if n.end - n.start < 0.02:
                continue
            if dedupe:
                key = (int(n.pitch), round(float(n.start), 3))
                if key in seen:
                    report.deduped += 1
                    continue
                seen.add(key)
            note = pretty_midi.Note(
                velocity=int(np.clip(n.velocity, 1, 127)),
                pitch=int(n.pitch),
                start=float(n.start),
                end=float(n.end),
            )
            # --- key snap ---
            if snap_key and not _is_drum(src):
                pc = (note.pitch - tonic) % 12
                if pc not in scale:
                    down, up = note.pitch - 1, note.pitch + 1
                    if (down - tonic) % 12 in scale:
                        note.pitch = down
                    elif (up - tonic) % 12 in scale:
                        note.pitch = up
                    report.snapped += 1
            # --- range fold ---
            if fold_ranges and not _is_drum(src):
                if role == "bass":
                    new_pitch, moved = _fold_into(note.pitch, 28, 55)
                elif role == "lead":
                    new_pitch, moved = _fold_into(note.pitch, 55, 84)
                elif role == "pad":
                    new_pitch, moved = _fold_into(note.pitch, 48, 84)
                else:
                    new_pitch, moved = _fold_into(note.pitch, 36, 91)
                if moved:
                    note.pitch = new_pitch
                    report.folded += 1
            kept_notes.append(note)
        inst.notes = kept_notes
        if kept_notes:
            new_instruments[idx] = inst

    # Remove instruments that lost every note.
    for idx in list(new_instruments):
        if not new_instruments[idx].notes:
            del new_instruments[idx]

    # Record what survived (an instrument can also be dropped if every note
    # was filtered out).
    for idx, inst in enumerate(pm.instruments):
        label = _label(idx, inst)
        if idx in new_instruments:
            report.kept.append(f"{label} role={kept_roles[idx]}")
        else:
            report.removed.append(label)

    # --- note lengths: stop the "16th-note blip" texture ------------------ #
    if sustain and new_instruments:
        end_of_piece = max(n.end for inst in new_instruments.values() for n in inst.notes)
        for idx, inst in new_instruments.items():
            if _is_drum(inst):
                continue
            report.sustained += _sustain_notes(inst, kept_roles.get(idx, "other"), end_of_piece)

    # --- monophonic lead: no overlapping notes ---------------------------- #
    if lead_idx is not None and lead_idx in new_instruments:
        lead = new_instruments[lead_idx]
        lead.notes.sort(key=lambda n: n.start)
        for a, b in zip(lead.notes, lead.notes[1:]):
            # Only trim a *melodic* overlap (a later onset). Notes that share
            # an onset are a chord - shortening them to 0.02 s would turn
            # every chord into a click.
            if b.start > a.start + 1e-4 and a.end > b.start:
                a.end = max(a.start + 0.02, b.start - 0.005)

    # --- velocity balance by role ----------------------------------------- #
    if balance:
        targets = {"lead": 96.0, "bass": 88.0, "drums": 100.0, "pad": 70.0, "other": 78.0}
        for idx, inst in new_instruments.items():
            if not inst.notes:
                continue
            target = targets.get(kept_roles.get(idx, "other"), 78.0)
            med = _median([n.velocity for n in inst.notes], target)
            gain = target / med if med > 1 else 1.0
            for n in inst.notes:
                n.velocity = int(np.clip(round(n.velocity * gain), 20, 120))

    for idx in sorted(new_instruments):
        out.instruments.append(new_instruments[idx])

    report.instruments_after = len(out.instruments)
    report.notes_after = sum(len(i.notes) for i in out.instruments)
    if report.instruments_after == 0:
        report.warnings.append("no notes survived arrangement cleanup")
    elif report.instruments_after < 3:
        report.warnings.append("very sparse arrangement (fewer than 3 instruments)")
    if report.notes_after < 20:
        report.warnings.append(f"only {report.notes_after} notes after cleanup")
    LOG.info(
        "arranged: key %s (r=%.2f), %d -> %d instruments, %d -> %d notes, "
        "%d snapped, %d folded",
        report.key,
        report.key_confidence,
        report.instruments_before,
        report.instruments_after,
        report.notes_before,
        report.notes_after,
        report.snapped,
        report.folded,
    )
    return out, report


def arrange_midi(
    midi_path: str | Path,
    out_path: str | Path | None = None,
    **kwargs: Any,
) -> tuple[Path, ArrangementReport]:
    """Load, clean and write a MIDI file. Returns ``(output_path, report)``."""
    import pretty_midi

    src = Path(midi_path)
    if not src.is_file():
        raise FileNotFoundError(f"MIDI not found: {src}")
    if out_path is None:
        out_path = src.with_name(f"{src.stem}_arranged.mid")
    out_path = Path(out_path)
    pm = pretty_midi.PrettyMIDI(str(src))
    cleaned, report = arrange(pm, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.write(str(out_path))
    return out_path, report
