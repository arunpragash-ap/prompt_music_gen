"""Shared fixtures for the promptmusic test suite.

Design rule: the default test run must be **fast and offline**. No test may
download a dataset or a transformer. Tests that genuinely need the frozen mT5
encoder (~1.1 GB) are marked ``slow`` and excluded from the default run by the
``-m "not slow"`` convention documented in the README.

Fixtures provided here:

``midi_file``
    A 4-second, two-instrument MIDI written to a temp directory. Everything
    that needs "some music" uses this rather than shipping a binary fixture.

``tiny_config``
    A Config whose decoder is 2 layers x 64 dims. Model tests use it so they
    run in milliseconds instead of seconds.

``tiny_tokenizer``
    A MusicTokenizer built from the default tokenizer settings.

``seeded_music_ids``
    A short, valid token id list (BOS + a few tokens + EOS) for decoder tests
    that should not pay for a real tokenization.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow running the tests without installing the package (no `pip install -e`).
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# --------------------------------------------------------------------------- #
# MIDI fixture
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def midi_file(tmp_path_factory) -> Path:
    """A short two-track MIDI file: melody over a bass line."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    melody = pretty_midi.Instrument(program=0, name="melody")   # piano
    bass = pretty_midi.Instrument(program=33, name="bass")      # electric bass

    # C major arpeggio, one note per beat, over 4 seconds.
    for i, pitch in enumerate([60, 64, 67, 72, 67, 64, 60, 64]):
        start = i * 0.5
        melody.notes.append(
            pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=start + 0.45)
        )
    for i, pitch in enumerate([48, 48, 55, 55]):
        start = i * 1.0
        bass.notes.append(
            pretty_midi.Note(velocity=70, pitch=pitch, start=start, end=start + 0.9)
        )

    pm.instruments.extend([bass, melody])
    path = tmp_path_factory.mktemp("midi") / "fixture.mid"
    pm.write(str(path))
    return path


@pytest.fixture(scope="session")
def multi_instrument_midi(tmp_path_factory) -> Path:
    """Four distinct GM programs, for the instrument-audit tests."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=100)
    for program in (0, 73, 104, 48):   # piano, flute, sitar, strings
        inst = pretty_midi.Instrument(program=program, name=f"p{program}")
        for i in range(8):
            inst.notes.append(
                pretty_midi.Note(
                    velocity=80, pitch=60 + program % 12 + i,
                    start=i * 0.25, end=i * 0.25 + 0.2,
                )
            )
        pm.instruments.append(inst)
    path = tmp_path_factory.mktemp("multi") / "multi.mid"
    pm.write(str(path))
    return path


# --------------------------------------------------------------------------- #
# Config / tokenizer fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def tiny_config():
    """A Config with a 2-layer, 64-wide decoder (fast model tests)."""
    from promptmusic.config import load_config

    cfg = load_config("smoke")   # smoke gives us the small data/step settings
    cfg.set("model.decoder.num_layers", 2)
    cfg.set("model.decoder.d_model", 64)
    cfg.set("model.decoder.nhead", 2)
    cfg.set("model.decoder.dim_feedforward", 128)
    cfg.set("model.decoder.max_len", 512)
    return cfg


@pytest.fixture()
def tiny_tokenizer():
    from promptmusic.data.tokenizer import build_tokenizer

    return build_tokenizer(None)


@pytest.fixture()
def seeded_music_ids(tiny_tokenizer) -> list[int]:
    """BOS + tokens for a few real musical events + EOS."""
    tok = tiny_tokenizer
    ids = [tok.bos_id]
    for token in ("Tempo_120.0", "TimeSig_4/4", "Program_0", "Bar_None",
                  "Position_0", "Pitch_60", "Velocity_80", "Duration_1.0.4"):
        try:
            ids.append(tok.token_to_id(token))
        except KeyError:
            pytest.skip(f"tokenizer vocabulary lacks {token}")
    ids.append(tok.eos_id)
    return ids


# --------------------------------------------------------------------------- #
# Decoder-only model fixture (no text encoder, no network)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def tiny_decoder(tiny_config, tiny_tokenizer):
    """A bare MusicDecoder - exercises all transformer logic offline."""
    from promptmusic.model.decoder import MusicDecoder

    return MusicDecoder(
        vocab_size=tiny_tokenizer.vocab_size,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        max_len=512,
        pad_id=0,
    )
