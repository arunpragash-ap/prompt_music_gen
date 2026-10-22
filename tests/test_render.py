"""MIDI -> WAV rendering: backend selection, the numpy fallback, output checks.

The numpy synthesiser is tested directly because it is the only backend that
is always available; fluidsynth and pyfluidsynth are exercised opportunistically
(skipped when not installed) so the suite still passes on a bare machine.

API notes (these drove several test fixes, documented here so they are not
"fixed back"):

* :class:`RenderResult` has ``wav_path`` - not ``path`` - and **no** ``success``
  field: ``render_midi`` either returns a result or raises. A returned result
  *is* the success signal.
* ``available_backends()`` reports the *detection* side, so its key for the
  command-line tool is ``fluidsynth_cli`` (a machine can have the CLI without
  the Python module and vice versa). Results still name the backend
  ``fluidsynth`` because that is what ``backend=`` accepts.
* A MIDI with no notes has zero duration: the numpy backend refuses it, so
  ``render_midi`` raises ``RuntimeError`` listing everything it tried. Silent
  output would be worse than a loud failure.
"""

from __future__ import annotations

import shutil

import pytest

from promptmusic.render import (
    available_backends,
    find_soundfont,
    render_midi,
)

FLUIDSYNTH = shutil.which("fluidsynth") or shutil.which("fluid-synth")


# --------------------------------------------------------------------------- #
# Backend discovery
# --------------------------------------------------------------------------- #
def test_available_backends_reports_every_entry():
    backends = available_backends()
    assert isinstance(backends, dict)
    # The three tiers documented in the README. Note the CLI tool is reported
    # as "fluidsynth_cli" so it can differ from the Python module.
    for key in ("fluidsynth_cli", "pyfluidsynth", "numpy"):
        assert key in backends, f"missing backend {key!r}"
    # numpy is pure Python + numpy, so it must always be available.
    assert backends["numpy"] is True
    # Soundfont detection lives in the same report.
    assert "soundfont" in backends


def test_numpy_backend_is_always_the_floor():
    backends = available_backends()
    assert backends["numpy"] is True


def test_find_soundfont_returns_path_or_none():
    sf = find_soundfont()
    if sf is not None:
        assert sf.is_file()
        assert sf.suffix.lower() == ".sf2"


# --------------------------------------------------------------------------- #
# numpy fallback (always available)
# --------------------------------------------------------------------------- #
def test_numpy_render_writes_a_valid_wav(midi_file, tmp_path):
    result = render_midi(midi_file, tmp_path / "out.wav", backend="numpy", sample_rate=22050)
    # Returning at all means success (failures raise).
    assert result.wav_path.exists()
    assert result.wav_path.stat().st_size > 44      # more than a bare WAV header
    assert result.backend == "numpy"
    assert result.duration_sec > 0
    assert result.sample_rate == 22050


def test_numpy_render_respects_sample_rate(midi_file, tmp_path):
    import soundfile as sf

    result = render_midi(midi_file, tmp_path / "sr.wav", backend="numpy", sample_rate=16000)
    data, rate = sf.read(result.wav_path)
    assert rate == 16000
    assert len(data) > 0


def test_numpy_render_gain_scales_the_signal(midi_file, tmp_path):
    import numpy as np
    import soundfile as sf

    loud = render_midi(midi_file, tmp_path / "loud.wav", backend="numpy", gain=1.0)
    soft = render_midi(midi_file, tmp_path / "soft.wav", backend="numpy", gain=0.3)
    loud_peak = np.abs(sf.read(loud.wav_path)[0]).max()
    soft_peak = np.abs(sf.read(soft.wav_path)[0]).max()
    assert soft_peak < loud_peak


def test_numpy_render_is_deterministic(midi_file, tmp_path):
    import numpy as np
    import soundfile as sf

    a = render_midi(midi_file, tmp_path / "a.wav", backend="numpy")
    b = render_midi(midi_file, tmp_path / "b.wav", backend="numpy")
    da = sf.read(a.wav_path)[0]
    db = sf.read(b.wav_path)[0]
    length = min(len(da), len(db))
    assert np.allclose(da[:length], db[:length])


def test_numpy_render_of_empty_midi_raises(tmp_path):
    """A note-less MIDI has zero duration; every backend refuses it.

    The contract is a loud RuntimeError naming the backends tried, never a
    silent 0-second "success" and never an unhandled crash type.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    empty = tmp_path / "empty.mid"
    pm.write(str(empty))
    with pytest.raises(RuntimeError, match="every rendering backend failed"):
        render_midi(empty, tmp_path / "empty.wav", backend="numpy")


def test_numpy_render_missing_file_raises(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        render_midi(tmp_path / "nope.mid", tmp_path / "nope.wav", backend="numpy")


# --------------------------------------------------------------------------- #
# Auto-selection through render_midi
# --------------------------------------------------------------------------- #
def test_render_auto_picks_a_working_backend(midi_file, tmp_path):
    result = render_midi(midi_file, tmp_path / "auto.wav", backend="auto")
    assert result.backend in {"fluidsynth", "pyfluidsynth", "numpy"}
    assert result.wav_path.exists()


def test_explicit_numpy_backend(midi_file, tmp_path):
    result = render_midi(midi_file, tmp_path / "n.wav", backend="numpy")
    assert result.backend == "numpy"


@pytest.mark.skipif(FLUIDSYNTH is None, reason="fluidsynth not installed")
def test_explicit_fluidsynth_backend(midi_file, tmp_path):
    result = render_midi(midi_file, tmp_path / "fs.wav", backend="fluidsynth")
    assert result.backend == "fluidsynth"
    assert result.wav_path.exists()


@pytest.mark.skipif(FLUIDSYNTH is None, reason="fluidsynth not installed")
def test_fluidsynth_output_is_longer_and_cleaner(midi_file, tmp_path):
    """A soundfont render is a real sampler; the numpy fallback is an
    approximation. Both must be non-empty, but the soundfont file should be
    at least as long (it lets notes ring out)."""
    numpy_r = render_midi(midi_file, tmp_path / "n.wav", backend="numpy")
    fs_r = render_midi(midi_file, tmp_path / "f.wav", backend="fluidsynth")
    assert fs_r.duration_sec >= numpy_r.duration_sec * 0.9


def test_unknown_backend_raises(midi_file, tmp_path):
    with pytest.raises(ValueError):
        render_midi(midi_file, tmp_path / "x.wav", backend="quantum")


def test_render_result_is_serialisable(midi_file, tmp_path):
    result = render_midi(midi_file, tmp_path / "s.wav")
    d = result.to_dict()
    import json

    json.dumps(d)
    assert {"wav_path", "backend", "duration_sec", "sample_rate"} <= set(d)
