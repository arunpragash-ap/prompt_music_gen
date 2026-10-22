"""Evaluation metrics: correctness, edge cases, and the serialisable report."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from promptmusic.evaluate import (
    EvaluationResult,
    evaluate_against_reference,
    evaluate_files,
    save_report,
    stats_for_midi,
)


def _write_midi(path: Path, pitches, tempo=120, gap=0.5, velocity=80, duration=0.4):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)
    for i, p in enumerate(pitches):
        inst.notes.append(
            pretty_midi.Note(velocity=velocity, pitch=p, start=i * gap, end=i * gap + duration)
        )
    pm.instruments.append(inst)
    pm.write(str(path))
    return path


# --------------------------------------------------------------------------- #
# stats_for_midi
# --------------------------------------------------------------------------- #
def test_stats_for_valid_midi(midi_file):
    s = stats_for_midi(midi_file)
    assert s["ok"] is True
    assert s["notes"] == 12          # 8 melody + 4 bass in the fixture
    assert s["duration_sec"] > 3.0
    assert s["tempo"] == pytest.approx(120, abs=1)
    assert 0 in s["programs"]        # piano
    assert 33 in s["programs"]       # bass
    assert len(s["pitch_classes"]) == 12


def test_stats_for_broken_file_returns_error_dict(tmp_path):
    bad = tmp_path / "bad.mid"
    bad.write_bytes(b"this is not midi")
    s = stats_for_midi(bad)
    assert s["ok"] is False
    assert "error" in s


def test_stats_for_empty_midi(tmp_path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    pm.write(str(tmp_path / "empty.mid"))
    s = stats_for_midi(tmp_path / "empty.mid")
    assert s["ok"] is False


# --------------------------------------------------------------------------- #
# Reference-free evaluation
# --------------------------------------------------------------------------- #
def test_evaluate_single_file(midi_file):
    r = evaluate_files([midi_file])
    assert r.num_files == 1
    assert r.validity_rate == 1.0
    assert r.notes_per_second > 0
    # The fixture is a C-major arpeggio: only 3 pitch classes (C, E, G).
    assert r.unique_pitch_classes >= 3
    assert 0 <= r.pitch_class_entropy <= math.log2(12) + 0.01


def test_evaluate_empty_list():
    r = evaluate_files([])
    assert r.num_files == 0
    assert any("no files" in n for n in r.notes)


def test_evaluate_counts_parse_failures(tmp_path, midi_file):
    (tmp_path / "broken.mid").write_bytes(b"garbage")
    r = evaluate_files([midi_file, tmp_path / "broken.mid"])
    assert r.num_files == 2
    assert r.parse_errors == 1
    assert r.validity_rate == 0.5


def test_tempo_match_and_mae(tmp_path):
    on_time = _write_midi(tmp_path / "on.mid", [60, 62, 64, 65], tempo=120)
    off_time = _write_midi(tmp_path / "off.mid", [60, 62, 64, 65], tempo=175)
    r = evaluate_files([on_time, off_time], expect_tempo=120)
    assert r.tempo_match == 0.5          # one of two obeyed
    assert r.tempo_mae == pytest.approx((0 + 55) / 2, abs=1)


def test_key_match_c_major(tmp_path):
    major = _write_midi(tmp_path / "cmaj.mid", [60, 64, 67, 72, 67, 64, 60, 64])
    r = evaluate_files([major], expect_key="C major")
    assert r.key_match == 1.0


def test_key_match_tolerates_enharmonics(tmp_path):
    _write_midi(tmp_path / "a.mid", [69, 72, 76, 81, 76, 72, 69, 72])
    r = evaluate_files([tmp_path / "a.mid"], expect_key="Bb major")
    # Whatever the estimate, a Bb request must not crash the enharmonic alias.
    assert r.key_match in (0.0, 1.0)


def test_duration_error(tmp_path):
    short = _write_midi(tmp_path / "s.mid", [60, 62, 64])
    r = evaluate_files([short], expect_duration=30.0)
    assert r.duration_error == pytest.approx(30.0 - 2 * 0.5 - 0.4, abs=0.5)


def test_instrument_coverage(tmp_path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    for program in (0, 73):     # piano, flute
        inst = pretty_midi.Instrument(program=program)
        for i in range(4):
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=60 + i, start=i * 0.5, end=i * 0.5 + 0.4)
            )
        pm.instruments.append(inst)
    path = tmp_path / "two.mid"
    pm.write(str(path))

    # Asking for flute (73) and sitar (105): half covered.
    r = evaluate_files([path], expect_instruments=[73, 105])
    assert r.instrument_coverage == pytest.approx(0.5)


def test_per_file_table_is_optional(midi_file):
    lean = evaluate_files([midi_file])
    assert lean.per_file == []
    full = evaluate_files([midi_file], keep_per_file=True)
    assert len(full.per_file) == 1
    assert full.per_file[0]["notes"] == 12


def test_repetition_rate_high_for_looped_phrase(tmp_path):
    loop = [60, 64, 67, 72] * 10
    looped = _write_midi(tmp_path / "loop.mid", loop)
    varied = _write_midi(
        tmp_path / "varied.mid",
        [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86],
    )
    r_loop = evaluate_files([looped])
    r_var = evaluate_files([varied])
    assert r_loop.repetition_rate > r_var.repetition_rate


def test_compression_ratio_is_normalised(midi_file):
    r = evaluate_files([midi_file])
    # Ratio of empirical entropy to log2(12): must sit in [0, 1].
    assert r.compression_ratio is not None
    assert 0.0 <= r.compression_ratio <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# Against a reference corpus
# --------------------------------------------------------------------------- #
def test_against_reference_identical_corpus(tmp_path):
    files = [
        _write_midi(tmp_path / f"r{i}.mid", [60, 64, 67, 72, 67, 64], tempo=120)
        for i in range(4)
    ]
    r = evaluate_against_reference(files, files)
    # Identical distributions -> divergence essentially zero.
    assert r.pitch_divergence == pytest.approx(0.0, abs=1e-6)
    assert r.rhythm_divergence == pytest.approx(0.0, abs=1e-6)
    assert r.tempo_wasserstein == pytest.approx(0.0, abs=1e-6)


def test_against_reference_detects_pitch_gap(tmp_path):
    gen = [_write_midi(tmp_path / f"g{i}.mid", [60, 62, 64, 65], tempo=100) for i in range(3)]
    ref = [_write_midi(tmp_path / f"f{i}.mid", [61, 63, 66, 68], tempo=100) for i in range(3)]
    r = evaluate_against_reference(gen, ref)
    assert r.pitch_divergence > 0.2


def test_against_reference_with_no_reference_files(midi_file):
    r = evaluate_against_reference([midi_file], [])
    assert r.pitch_divergence is None
    assert any("no reference" in n for n in r.notes)


# --------------------------------------------------------------------------- #
# Report shape
# --------------------------------------------------------------------------- #
def test_result_serialises_to_json(midi_file):
    r = evaluate_files([midi_file], expect_tempo=120)
    d = r.to_dict()
    json.dumps(d)   # must not raise
    assert d["num_files"] == 1
    assert "validity_rate" in d


def test_save_report_writes_file(tmp_path, midi_file):
    r = evaluate_files([midi_file])
    out = save_report(r, tmp_path / "report.json")
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["num_files"] == 1


def test_summary_renders_without_error(midi_file):
    text = evaluate_files([midi_file], expect_tempo=120).summary()
    assert "EVALUATION" in text
    assert "validity" in text


def test_notes_flag_high_repetition(tmp_path):
    looped = _write_midi(tmp_path / "loop.mid", [60, 64, 67] * 12)
    r = evaluate_files([looped])
    assert any("repetition" in n for n in r.notes)
