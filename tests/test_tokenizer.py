"""REMI tokenization: round-trips, vocabulary structure, lyric interleaving."""

from __future__ import annotations

import pytest

from promptmusic.data.tokenizer import (
    LyricVocab,
    build_tokenizer,
    contains_tamil,
    is_tamil_char,
)


# --------------------------------------------------------------------------- #
# Vocabulary structure
# --------------------------------------------------------------------------- #
def test_vocab_ids_are_contiguous_from_zero(tiny_tokenizer):
    """miditok must hand us an id space 0..N-1; padding assumes id 0."""
    ids = set(tiny_tokenizer.vocab.values())
    assert min(ids) == 0
    assert max(ids) == len(ids) - 1
    assert len(ids) == tiny_tokenizer.vocab_size


def test_special_token_positions(tiny_tokenizer):
    """train.py hard-codes pad=0, bos=1, eos=2; this is the contract."""
    assert tiny_tokenizer.pad_id == 0
    assert tiny_tokenizer.bos_id == 1
    assert tiny_tokenizer.eos_id == 2
    assert tiny_tokenizer.unk_id >= 0


def test_vocab_contains_expected_token_families(tiny_tokenizer):
    tokens = set(tiny_tokenizer.vocab)
    # REMI families we rely on for multi-track, tempo-aware generation.
    families = ["Bar_None", "Position_", "Pitch_", "Velocity_", "Duration_",
                "Program_", "Tempo_", "TimeSig_"]
    for family in families:
        assert any(t.startswith(family) for t in tokens), (
            f"no token starts with {family!r}"
        )


def test_vocab_size_is_sane(tiny_tokenizer):
    """A REMI vocab of a few hundred to a few thousand tokens. Too small
    means configuration is broken; too large means beat_res exploded."""
    assert 100 < tiny_tokenizer.vocab_size < 20000


def test_token_to_id_roundtrip(tiny_tokenizer):
    for token, index in list(tiny_tokenizer.vocab.items())[:50]:
        assert tiny_tokenizer.token_to_id(token) == index
        assert tiny_tokenizer.id_to_token(index) == token


# --------------------------------------------------------------------------- #
# MIDI -> ids -> MIDI
# --------------------------------------------------------------------------- #
def test_encode_midi_returns_nonempty_ids(tiny_tokenizer, midi_file):
    ids = tiny_tokenizer.encode_midi(midi_file)
    assert len(ids) > 8
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < tiny_tokenizer.vocab_size for i in ids)


def test_encode_respects_max_tokens(tiny_tokenizer, midi_file):
    full = tiny_tokenizer.encode_midi(midi_file)
    capped = tiny_tokenizer.encode_midi(midi_file, max_tokens=16)
    assert len(capped) <= 16
    assert len(full) >= len(capped)


def test_decode_ids_returns_a_score(tiny_tokenizer, midi_file):
    ids = tiny_tokenizer.encode_midi(midi_file)
    score = tiny_tokenizer.decode_ids(ids)
    assert score is not None
    # A score with tracks or notes survives the round trip.
    assert score.note_num() > 0


def test_save_midi_roundtrip_preserves_music(tiny_tokenizer, midi_file, tmp_path):
    """ids -> decode -> .mid -> re-encode must still be valid music."""
    ids = tiny_tokenizer.encode_midi(midi_file)
    out = tiny_tokenizer.save_midi(ids, tmp_path / "roundtrip.mid")
    assert out.exists() and out.stat().st_size > 0

    reencoded = tiny_tokenizer.encode_midi(out)
    assert len(reencoded) > 0
    # Structure (bar markers at minimum) should survive.
    assert any("Bar_None" in tiny_tokenizer.id_to_token(i) for i in reencoded)


def test_encoding_is_deterministic(tiny_tokenizer, midi_file):
    """Same file, same vocab, same ids - a prerequisite for caching."""
    assert tiny_tokenizer.encode_midi(midi_file) == tiny_tokenizer.encode_midi(midi_file)


def test_two_instruments_produce_program_tokens(tiny_tokenizer, multi_instrument_midi):
    ids = tiny_tokenizer.encode_midi(multi_instrument_midi)
    tokens = [tiny_tokenizer.id_to_token(i) for i in ids]
    programs = {t for t in tokens if t.startswith("Program_")}
    assert len(programs) >= 2, "multi-track MIDI lost its instrument identity"


def test_encode_missing_file_raises(tmp_path, tiny_tokenizer):
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        tiny_tokenizer.encode_midi(tmp_path / "nope.mid")


# --------------------------------------------------------------------------- #
# Lyric vocabulary (Phase 3)
# --------------------------------------------------------------------------- #
def test_tamil_detection():
    assert contains_tamil("மகிழ்ச்சி")
    assert is_tamil_char("க")
    assert not contains_tamil("plain english")
    assert not is_tamil_char("a")


def test_lyric_vocab_hybrid_learns_english_and_tamil():
    vocab = LyricVocab(mode="hybrid", max_vocab=500)
    vocab.build([
        "the rain in tamil மழை falls soft",
        "another line of lyrics here",
    ])
    tokens = vocab.all_tokens()
    assert tokens, "vocab is empty"
    # Tamil characters must be representable in hybrid mode.
    encoded = vocab.encode("the rain மழை")
    assert encoded
    assert vocab.decode(encoded).replace("  ", " ") != ""


def test_lyric_vocab_save_load_roundtrip(tmp_path):
    vocab = LyricVocab(mode="hybrid", max_vocab=300)
    vocab.build(["hello world வணக்கம்"])
    path = vocab.save(tmp_path / "vocab.json")
    loaded = LyricVocab.load(path)
    assert loaded.all_tokens() == vocab.all_tokens()
    assert loaded.encode("hello") == vocab.encode("hello")


def test_lyric_vocab_unknown_words_are_spelled_out():
    vocab = LyricVocab(mode="hybrid", max_vocab=100)
    vocab.build(["alpha beta"])
    # An unseen word falls back to characters rather than being dropped.
    encoded = vocab.encode("zzzqqq")
    assert encoded


# --------------------------------------------------------------------------- #
# Lyric interleaving at bar boundaries
# --------------------------------------------------------------------------- #
def _lyric_tokenizer() -> "MusicTokenizer":
    """A tokenizer with lyrics enabled and a two-word English vocabulary."""
    from promptmusic.data.tokenizer import LyricVocab, MusicTokenizer

    vocab = LyricVocab(mode="hybrid").build(["the rain", "in the rain"])
    return MusicTokenizer({}, lyrics=vocab)


def test_interleave_lyrics_places_tokens_at_bars(midi_file):
    tok = _lyric_tokenizer()
    ids = tok.encode_midi(midi_file)
    lyric_tokens = ["lyric_w_the", "lyric_w_rain"]
    combined = tok.interleave_lyrics(ids, lyric_tokens)
    assert len(combined) > len(ids)
    # The original music must still be recoverable by dropping lyric ids.
    stripped = tok.remove_lyric_ids(combined)
    assert stripped == ids
    # And the lyric text round-trips through extract_lyrics.
    assert "the" in tok.extract_lyrics(combined)


def test_interleave_without_lyrics_vocab_is_identity(midi_file, tiny_tokenizer):
    """A tokenizer with lyrics=None silently no-ops rather than crashing."""
    ids = tiny_tokenizer.encode_midi(midi_file)
    assert tiny_tokenizer.interleave_lyrics(ids, ["lyric_w_the"]) == ids


def test_interleave_with_empty_lyrics_is_identity(tiny_tokenizer, midi_file):
    ids = tiny_tokenizer.encode_midi(midi_file)
    assert tiny_tokenizer.interleave_lyrics(ids, []) == ids
    assert tiny_tokenizer.remove_lyric_ids(ids) == ids


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def test_build_tokenizer_accepts_section_dicts():
    """Both a raw section dict and a Config must work."""
    tok = build_tokenizer({"beat_res": {"0,4": 4}, "num_velocities": 16})
    assert tok.vocab_size > 0


def test_build_tokenizer_is_cached_by_config():
    """Two calls with the same config return the same instance (vocabulary
    construction is not free)."""
    a = build_tokenizer(None)
    b = build_tokenizer(None)
    assert a is b
