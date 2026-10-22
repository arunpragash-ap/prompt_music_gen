"""Hindustani raga grammar: the table, the swara mapping, and the generator.

These tests are about *music correctness*, not just "does it run": a raga whose
generated notes fall outside its own scale is worse than no raga at all,
because the model would learn the wrong grammar.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from promptmusic.data.raga_gen import (
    DEFAULT_TABLE,
    Raga,
    RagaGenerator,
    build_scale,
    generate_raga_corpus,
    list_raga_names,
    load_raga_table,
    swara_to_semitone,
)

pytestmark = pytest.mark.usefixtures("midi_file")


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def test_table_file_ships_with_repo():
    assert DEFAULT_TABLE.is_file(), "data/ragas/raga_table.yaml is missing"


def test_table_loads_with_24_ragas():
    offsets, ragas = load_raga_table()
    assert len(ragas) == 24
    assert len(offsets) == 12   # one entry per swara


def test_every_raga_has_required_fields():
    _, ragas = load_raga_table()
    for raga in ragas:
        assert raga.name
        assert raga.thaat, f"{raga.name} has no thaat"
        assert raga.aroha, f"{raga.name} has no ascending scale"
        assert raga.avaroha, f"{raga.name} has no descending scale"
        assert raga.source, f"{raga.name} has no citation"
        # Every entry must be attributable.
        assert "[" in raga.source, f"{raga.name} source lacks a reference tag"


def test_raga_names_are_unique_and_filesystem_safe():
    names = list_raga_names()
    assert len(names) == len(set(names))
    for name in names:
        # Names become filenames, so no spaces or slashes.
        assert " " not in name
        assert "/" not in name


def test_all_ten_thaats_represented():
    """Bhatkhande's system has ten thaats; the table should cover them all."""
    _, ragas = load_raga_table()
    thaats = {r.thaat for r in ragas}
    expected = {
        "Bilawal", "Khamaj", "Kafi", "Asavari", "Bhairav",
        "Bhairavi", "Todi", "Purvi", "Marwa", "Kalyan",
    }
    assert expected <= thaats, f"missing thaats: {expected - thaats}"


def test_pentatonic_ragas_report_five_notes():
    _, ragas = load_raga_table()
    by_name = {r.name: r for r in ragas}
    assert by_name["Bhupali"].is_pentatonic
    assert by_name["Malkauns"].is_pentatonic
    # Bilawal is the all-shuddha scale = Western major / Ionian: 7 notes.
    assert by_name["Bilawal"].scale_classes == {0, 2, 4, 5, 7, 9, 11}
    # And a pentatonic really is five.
    assert len(by_name["Bhupali"].scale_classes) == 5
    assert len(by_name["Malkauns"].scale_classes) == 5


def test_vakra_ragas_flagged():
    _, ragas = load_raga_table()
    by_name = {r.name: r for r in ragas}
    assert by_name["Lalit"].vakra is True
    assert by_name["Darbari_Kanada"].vakra is True
    assert by_name["Yaman"].vakra is False


# --------------------------------------------------------------------------- #
# Swara -> semitone mapping
# --------------------------------------------------------------------------- #
def test_swara_semitones_match_the_documented_table():
    assert swara_to_semitone("S") == 0
    assert swara_to_semitone("r") == 1     # komal Re
    assert swara_to_semitone("R") == 2     # shuddha Re
    assert swara_to_semitone("g") == 3
    assert swara_to_semitone("G") == 4
    assert swara_to_semitone("M") == 5
    assert swara_to_semitone("M^") == 6    # teevra Ma
    assert swara_to_semitone("P") == 7
    assert swara_to_semitone("d") == 8
    assert swara_to_semitone("D") == 9
    assert swara_to_semitone("n") == 10
    assert swara_to_semitone("N") == 11


def test_octave_markers():
    assert swara_to_semitone("S'") == 12
    assert swara_to_semitone("S,") == -12
    assert swara_to_semitone("N'") == 23


def test_unknown_swara_raises():
    with pytest.raises(KeyError):
        swara_to_semitone("X")


def test_build_scale_maps_around_tonic():
    _, ragas = load_raga_table()
    yaman = next(r for r in ragas if r.name == "Yaman")
    scale = build_scale(yaman, tonic=60)
    assert scale["S"] == 60
    assert scale["S'"] == 72
    assert scale["S,"] == 48
    # Yaman's teevra Ma: 60 + 6 = 66.
    assert scale["M^"] == 66
    # And it begins on Ni (B below the tonic, i.e. 71 = 60 + 11).
    assert scale["N"] == 71


# --------------------------------------------------------------------------- #
# Generator output
# --------------------------------------------------------------------------- #
def test_generated_notes_respect_the_raga_scale():
    """The core music-theory guarantee: every pitch class is a legal swara."""
    offsets, ragas = load_raga_table()
    for raga in ragas:
        gen = RagaGenerator(raga, offsets=offsets, tempo=80, tonic=60, seed=1)
        pm = gen.build()
        legal = raga.scale_classes
        pitches = [n.pitch for inst in pm.instruments if not inst.is_drum
                   for n in inst.notes]
        assert pitches, f"{raga.name} generated no notes"
        bad = {p % 12 for p in pitches if (p % 12) not in legal}
        assert not bad, (
            f"{raga.name} emitted pitch classes {sorted(bad)} "
            f"outside its scale {sorted(legal)}"
        )


def test_generated_structure_has_alap_jor_gat():
    """Drone + lead + tabla, with the tabla entering late (the build-up)."""
    offsets, ragas = load_raga_table()
    raga = next(r for r in ragas if r.name == "Yaman")
    pm = RagaGenerator(raga, offsets=offsets, tempo=80, seed=3).build()

    names = [i.name for i in pm.instruments]
    assert any("lead" in n for n in names)
    assert any("tanpura" in n or "drone" in n for n in names)
    assert any("tabla" in n for n in names)

    tabla = next(i for i in pm.instruments if "tabla" in i.name)
    lead = next(i for i in pm.instruments if "lead" in i.name)
    # Tabla starts after the alap+jor opening, not at time zero.
    first_tabla = min(n.start for n in tabla.notes)
    first_lead = min(n.start for n in lead.notes)
    assert first_tabla > first_lead
    assert len(tabla.notes) > 16     # at least one full Teental cycle


def test_drone_spans_the_whole_piece():
    offsets, ragas = load_raga_table()
    raga = ragas[0]
    pm = RagaGenerator(raga, offsets=offsets, seed=5).build()
    lead = next(i for i in pm.instruments if "lead" in i.name)
    drone = next(i for i in pm.instruments if "tanpura" in i.name)
    lead_end = max(n.end for n in lead.notes)
    drone_end = max(n.end for n in drone.notes)
    assert drone_end >= lead_end * 0.9


def test_generator_is_deterministic_given_seed():
    offsets, ragas = load_raga_table()
    raga = next(r for r in ragas if r.name == "Kafi")

    def pitches(seed):
        pm = RagaGenerator(raga, offsets=offsets, tempo=70, seed=seed).build()
        return [n.pitch for i in pm.instruments for n in i.notes]

    assert pitches(11) == pitches(11)
    assert pitches(11) != pitches(12)


def test_tonic_jitter_moves_everything_transposed():
    offsets, ragas = load_raga_table()
    raga = next(r for r in ragas if r.name == "Bhupali")
    pm = RagaGenerator(raga, offsets=offsets, tonic=64, seed=2).build()
    pitches = [n.pitch for i in pm.instruments if not i.is_drum for n in i.notes]
    assert pitches
    # Sa = 64 (E), so every pitch class must be E-relative legal for Bhupali.
    legal = raga.scale_classes          # {0, 2, 4, 7, 9} relative to Sa
    sa_pc = 64 % 12
    for p in pitches:
        assert ((p % 12) - sa_pc) % 12 in legal


def test_corpus_generation_writes_files_and_index(tmp_path):
    out = tmp_path / "raga_corpus"
    summary = generate_raga_corpus(
        out_dir=out, variations_per_raga=1, seed=7,
    )
    assert summary["num_files"] == 24
    assert not summary["failures"]
    assert (out / "captions.json").exists()

    import json
    index = json.loads((out / "captions.json").read_text())
    assert len(index) == 24
    # Captions must name the raga - that is how the model learns the term.
    sample = next(r for r in index if r["raga"] == "Yaman")
    assert "Yaman" in sample["caption"]
    assert "raga" in sample["caption"].lower()
    # MidiCaps-compatible shape so corpora can be concatenated.
    for key in ("location", "caption", "tempo", "duration", "genre"):
        assert key in sample


def test_corpus_is_reproducible(tmp_path):
    a = generate_raga_corpus(out_dir=tmp_path / "a", variations_per_raga=1, seed=5)
    b = generate_raga_corpus(out_dir=tmp_path / "b", variations_per_raga=1, seed=5)
    assert a["num_files"] == b["num_files"]
    fa = sorted((tmp_path / "a").glob("*.mid"))
    fb = sorted((tmp_path / "b").glob("*.mid"))
    assert [f.name for f in fa] == [f.name for f in fb]
    # Byte-identical output for the same seed.
    assert fa[0].read_bytes() == fb[0].read_bytes()


def test_tempo_stays_in_range(tmp_path):
    generate_raga_corpus(out_dir=tmp_path / "t", variations_per_raga=2, seed=1)
    import pretty_midi
    for path in (tmp_path / "t").glob("*.mid"):
        pm = pretty_midi.PrettyMIDI(str(path))
        tempo = pm.get_tempo_changes()[1][0]
        assert 40 <= tempo <= 160, f"{path.name} tempo {tempo} out of range"
