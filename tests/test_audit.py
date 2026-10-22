"""Instrument audit and rebalancing (Phase 6)."""

from __future__ import annotations

import json

import pytest

from promptmusic.data.audit import (
    audit_instrument_distribution,
    example_weights,
    filter_by_instrument,
    gm_program_name,
    load_rebalanced_index,
    rebalance_report,
    write_rebalanced_index,
)


def test_gm_program_names():
    assert gm_program_name(0) == "Acoustic Grand Piano"
    assert gm_program_name(73) == "Flute"
    # GM program numbers are 0-indexed: 104 = Sitar, 105 = Banjo.
    assert gm_program_name(104) == "Sitar"
    assert gm_program_name(127) == "Gunshot"
    assert gm_program_name(200) == "program_200"     # out of range is safe


def test_audit_counts_programs(multi_instrument_midi):
    audit = audit_instrument_distribution(paths=[multi_instrument_midi])
    assert audit.total_files == 1
    used = set(audit.used_programs)
    assert {0, 73, 104, 48} <= used
    # Programs we did not use must appear in the "absent" list.
    assert 60 in audit.unused_gm_programs
    assert len(audit.unused_gm_programs) == 128 - len(used)


def test_audit_note_counts(multi_instrument_midi):
    audit = audit_instrument_distribution(paths=[multi_instrument_midi])
    stat = audit.stats[73]
    assert stat.name == "Flute"
    assert stat.notes == 8
    assert stat.tracks == 1
    assert stat.files == 1
    assert stat.file_rate == 1.0


def test_audit_rejects_missing_argument():
    with pytest.raises(ValueError):
        audit_instrument_distribution()


def test_audit_skips_broken_files(tmp_path, multi_instrument_midi):
    (tmp_path / "junk.mid").write_bytes(b"not midi at all")
    audit = audit_instrument_distribution(
        paths=[multi_instrument_midi, tmp_path / "junk.mid"]
    )
    assert audit.total_files == 1     # the broken one was skipped, not fatal


def test_audit_scans_a_directory(tmp_path, multi_instrument_midi):
    import shutil

    shutil.copy(multi_instrument_midi, tmp_path / "copy.mid")
    audit = audit_instrument_distribution(root=tmp_path)
    assert audit.total_files == 1


def test_audit_role_classification(multi_instrument_midi):
    audit = audit_instrument_distribution(paths=[multi_instrument_midi], roles=True)
    roles = audit.stats[0].roles
    # Whatever role piano got, the counter must have exactly one entry summing
    # to the number of tracks.
    assert sum(roles.values()) == 1


def test_entropy_and_concentration(multi_instrument_midi):
    audit = audit_instrument_distribution(paths=[multi_instrument_midi])
    # Four equally-used programs: entropy = log2(4) = 2 bits, concentration 0.25.
    assert audit.entropy_bits == pytest.approx(2.0, abs=0.01)
    assert audit.concentration == pytest.approx(0.25)


def test_report_is_readable(multi_instrument_midi):
    audit = audit_instrument_distribution(paths=[multi_instrument_midi])
    text = rebalance_report(audit)
    assert "INSTRUMENT DISTRIBUTION AUDIT" in text
    assert "Flute" in text
    assert "Sitar" in text
    assert "RECOMMENDATION" in text


def test_write_weights_and_reload(tmp_path, multi_instrument_midi):
    audit = audit_instrument_distribution(paths=[multi_instrument_midi])
    path = write_rebalanced_index(audit, out_path=tmp_path / "w.json")
    assert path.exists()

    data = json.loads(path.read_text())
    assert data["strategy"] == "sqrt_invsqrt"
    assert data["mean_weight"] == 1.0

    weights = load_rebalanced_index(path)
    # With one file, every used program has equal rate -> equal weights.
    assert set(weights) == {0, 48, 73, 104}
    assert all(w == pytest.approx(1.0) for w in weights.values())


def test_weight_strategies_differ(tmp_path, midi_file, multi_instrument_midi):
    audit = audit_instrument_distribution(
        paths=[midi_file, multi_instrument_midi]
    )
    uniform = write_rebalanced_index(
        audit, out_path=tmp_path / "u.json", strategy="uniform"
    )
    inverse = write_rebalanced_index(
        audit, out_path=tmp_path / "i.json", strategy="full"
    )
    u = load_rebalanced_index(uniform)
    i = load_rebalanced_index(inverse)
    # piano appears in both files; flute only in the second -> full inversion
    # must give the rarer one a higher weight than uniform does.
    assert i[73] > u[73]


def test_unknown_strategy_raises(multi_instrument_midi):
    audit = audit_instrument_distribution(paths=[multi_instrument_midi])
    with pytest.raises(ValueError):
        write_rebalanced_index(audit, strategy="nonsense")


def test_load_missing_index_returns_empty(tmp_path):
    assert load_rebalanced_index(tmp_path / "nope.json") == {}


def test_filter_by_inclusion(tmp_path, multi_instrument_midi, midi_file):
    import shutil

    shutil.copy(multi_instrument_midi, tmp_path / "multi.mid")
    shutil.copy(midi_file, tmp_path / "piano.mid")
    # Only the multi file contains a flute.
    keep = filter_by_instrument(tmp_path, include=[73])
    assert len(keep) == 1
    assert keep[0].name == "multi.mid"


def test_filter_by_exclusion(tmp_path, multi_instrument_midi, midi_file):
    import shutil

    shutil.copy(multi_instrument_midi, tmp_path / "multi.mid")
    shutil.copy(midi_file, tmp_path / "piano.mid")
    keep = filter_by_instrument(tmp_path, exclude=[73])
    assert [p.name for p in keep] == ["piano.mid"]


def test_filter_require_all(tmp_path, multi_instrument_midi, midi_file):
    import shutil

    shutil.copy(multi_instrument_midi, tmp_path / "multi.mid")
    shutil.copy(midi_file, tmp_path / "piano.mid")
    # piano.mid has only program 0; requiring flute too excludes it.
    both = filter_by_instrument(tmp_path, include=[0, 73], require_all=True)
    assert [p.name for p in both] == ["multi.mid"]
    either = filter_by_instrument(tmp_path, include=[0, 73], require_all=False)
    assert len(either) == 2


def test_example_weights_normalised(multi_instrument_midi):
    audit = audit_instrument_distribution(paths=[multi_instrument_midi])
    weights = example_weights(audit, [{73}, {0}, set()])
    assert len(weights) == 3
    assert sum(weights) == pytest.approx(3.0, abs=1e-6)   # mean 1.0
    assert weights[2] == pytest.approx(1.0)               # no instruments -> 1.0
