"""Tests for arrangement cleanup (arrange.py).

The model's raw MIDI is often "valid but unmusical": too many instruments,
notes outside the key, a bass above the melody, flat velocities. These tests
pin the corrections down with tiny hand-built files so they stay fast and
offline.
"""

from __future__ import annotations

import pretty_midi
import pytest

from promptmusic.arrange import (
    _SCALES,
    arrange,
    arrange_midi,
    estimate_key,
)


def _pm(parts, tempo=100.0) -> pretty_midi.PrettyMIDI:
    """Build a PrettyMIDI from ``[(program, notes, is_drum), ...]``.

    ``notes`` is a list of ``(pitch, start, end, velocity)`` tuples.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for program, notes, is_drum in parts:
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
        for pitch, start, end, velocity in notes:
            inst.notes.append(
                pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
            )
        pm.instruments.append(inst)
    return pm


def _scale_run(tonic: int, mode: str, octave: int = 4, repeats: int = 2):
    degrees = sorted(_SCALES[mode])
    notes = []
    t = 0.0
    for _ in range(repeats):
        for degree in degrees:
            pitch = 12 * octave + tonic + degree
            notes.append((pitch, t, t + 0.5, 100))
            t += 0.5
    return notes


def test_estimate_key_major() -> None:
    # C major triad plus a leading tone, with the tonic on top.
    notes = [
        (60, 0.0, 1.0, 100),
        (64, 1.0, 2.0, 100),
        (67, 2.0, 3.0, 100),
        (71, 3.0, 3.5, 100),
        (72, 3.5, 5.0, 100),
    ]
    tonic, mode, confidence = estimate_key(_pm([(0, notes, False)]))
    assert tonic == 0
    assert mode == "major"
    assert confidence > 0.5


def test_estimate_key_minor() -> None:
    # A minor with a raised 7th (G#) makes the mode unambiguous.
    notes = [
        (57, 0.0, 2.0, 100),
        (60, 2.0, 3.0, 100),
        (64, 3.0, 4.0, 100),
        (68, 4.0, 4.5, 100),
        (69, 4.5, 6.0, 100),
        (57, 6.0, 8.0, 100),
    ]
    tonic, mode, confidence = estimate_key(_pm([(0, notes, False)]))
    assert tonic == 9
    assert mode == "minor"
    assert confidence > 0.5


def test_snaps_out_of_key_notes() -> None:
    tonic = 0
    scale = _SCALES["major"]
    # Strong C-major context (C-E-G repeated), plus six C# notes that do not
    # belong to the key.
    context = [(60, 0.0, 0.5, 100), (64, 0.5, 1.0, 100), (67, 1.0, 1.5, 100)]
    context = [(p, s + i * 1.5, e + i * 1.5, v)
               for i in range(4) for p, s, e, v in context]
    out_of_key = [(61, 6.0 + i * 0.5, 6.4 + i * 0.5, 100) for i in range(6)]
    pm = _pm([(0, context + out_of_key, False)])
    out, report = arrange(pm, max_instruments=1, fold_ranges=False)
    assert report.snapped == 6
    assert report.key == "C major"
    for note in out.instruments[0].notes:
        assert (note.pitch - tonic) % 12 in scale


def test_prunes_instruments_and_keeps_lead_and_drums() -> None:
    lead = [(60 + (i % 12), i * 0.25, i * 0.25 + 0.2, 100) for i in range(24)]
    small = [(55 + i, i * 0.5, i * 0.5 + 0.4, 100) for i in range(4)]
    parts = [(0, lead, False), (32, small, False), (48, small, False),
             (25, small, False), (30, small, False), (71, small, False),
             (0, [(36, i * 0.5, i * 0.5 + 0.2, 90) for i in range(8)], True)]
    out, report = arrange(_pm(parts), max_instruments=4)
    assert report.instruments_before == 7
    assert report.instruments_after <= 5  # 4 non-drum + drums
    programs = {inst.program for inst in out.instruments}
    assert 0 in programs  # lead piano
    assert any(inst.is_drum for inst in out.instruments)
    assert len(out.instruments) == len(report.kept)
    assert report.instruments_before == len(report.kept) + len(report.removed)


def test_bass_is_folded_below_the_lead() -> None:
    lead = [(72, i * 0.25, i * 0.25 + 0.2, 100) for i in range(16)]
    bass = [(76, i * 0.5, i * 0.5 + 0.4, 100) for i in range(4)]  # far too high
    out, report = arrange(_pm([(0, lead, False), (33, bass, False)]), max_instruments=2)
    bass_inst = next(i for i in out.instruments if i.program == 33)
    assert bass_inst.notes
    assert max(n.pitch for n in bass_inst.notes) <= 55
    assert report.folded > 0


def test_velocities_are_balanced_by_role() -> None:
    lead = [(72, i * 0.25, i * 0.25 + 0.2, 100) for i in range(16)]
    pad = [(60, i * 1.0, i * 1.0 + 0.9, 100) for i in range(4)]
    out, _ = arrange(_pm([(0, lead, False), (48, pad, False)]), max_instruments=2)
    # Lead target 96, pad target 70.
    assert 90 <= max(n.velocity for n in out.instruments[0].notes) <= 100
    pad_inst = next(i for i in out.instruments if i.program == 48)
    assert all(60 <= n.velocity <= 80 for n in pad_inst.notes)


def test_dedupes_identical_notes() -> None:
    notes = [(60, 0.0, 1.0, 100), (60, 0.0, 1.0, 90), (62, 1.0, 2.0, 100)]
    out, report = arrange(_pm([(0, notes, False)]), max_instruments=1)
    assert report.deduped == 1
    assert len(out.instruments[0].notes) == 2


def test_monophonic_lead_does_not_overlap() -> None:
    notes = [(60, 0.0, 1.0, 100), (62, 0.5, 1.5, 100), (64, 1.0, 2.0, 100)]
    out, _ = arrange(_pm([(0, notes, False)]), max_instruments=1)
    lead = out.instruments[0]
    for a, b in zip(lead.notes, lead.notes[1:]):
        assert a.end <= b.start + 1e-6


def test_arrange_does_not_mutate_input() -> None:
    notes = [(60, 0.0, 1.0, 100), (61, 1.0, 2.0, 100), (62, 2.0, 3.0, 100)]
    pm = _pm([(0, notes, False)])
    before = [(n.pitch, n.start) for n in pm.instruments[0].notes]
    arrange(pm, snap_key=True)
    assert [(n.pitch, n.start) for n in pm.instruments[0].notes] == before


def test_tempo_is_preserved() -> None:
    notes = [(60, 0.0, 1.0, 100), (64, 1.0, 2.0, 100), (67, 2.0, 3.0, 100)]
    out, _ = arrange(_pm([(0, notes, False)], tempo=90.0))
    _, tempos = out.get_tempo_changes()
    assert tempos[0] == pytest.approx(90.0)


def test_arrange_midi_writes_file(tmp_path) -> None:
    notes = [(60, i * 0.5, i * 0.5 + 0.4, 100) for i in range(4)]
    src = tmp_path / "in.mid"
    _pm([(0, notes, False)]).write(str(src))
    out, report = arrange_midi(src)
    assert out.is_file()
    assert out.name == "in_arranged.mid"
    assert report.notes_after > 0


def test_sustain_lengthens_short_notes() -> None:
    lead = [(72, 0.0, 0.12, 100), (74, 1.0, 1.12, 100), (76, 2.0, 2.12, 100)]
    out, report = arrange(_pm([(0, lead, False)]), max_instruments=1)
    durations = [n.end - n.start for n in out.instruments[0].notes]
    assert report.sustained >= 2
    assert durations[0] == pytest.approx(0.98, abs=0.02)
    assert durations[1] == pytest.approx(0.98, abs=0.02)


def test_sustain_leaves_drums_alone() -> None:
    drums = [(36, 0.0, 0.12, 100), (38, 0.5, 0.62, 100), (36, 1.0, 1.12, 100)]
    out, report = arrange(_pm([(0, drums, True)]), max_instruments=1)
    assert report.sustained == 0
    assert all(n.end - n.start == pytest.approx(0.12) for n in out.instruments[0].notes)


def test_pad_holds_a_long_chord() -> None:
    lead = [(72, i * 0.5, i * 0.5 + 0.12, 100) for i in range(16)]
    pad = [(60, 0.0, 0.12, 100), (64, 0.0, 0.12, 100), (67, 8.0, 8.12, 100)]
    out, _ = arrange(_pm([(0, lead, False), (48, pad, False)]), max_instruments=2)
    pad_inst = next(i for i in out.instruments if i.program == 48)
    chord = [n.end - n.start for n in pad_inst.notes if n.start < 1.0]
    assert len(chord) == 2
    assert min(chord) > 2.0  # chord rings until the next event


def test_chords_are_not_collapsed_to_clicks() -> None:
    # Three simultaneous notes every two seconds. The mono cleanup used to
    # read a chord as a "self-overlap" and truncate all but the last note.
    notes = []
    for beat in range(4):
        for pitch in (60, 64, 67):
            notes.append((pitch, beat * 2.0, beat * 2.0 + 0.12, 100))
    out, _ = arrange(_pm([(0, notes, False)]), max_instruments=1)
    durations = [n.end - n.start for n in out.instruments[0].notes]
    assert min(durations) > 0.02  # no 0.02 s clicks
    assert sum(d > 1.5 for d in durations) >= 9  # every chord except the last rings
