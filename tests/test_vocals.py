"""Tests for the vocal layer (vocals.py) and tempo sanitization.

All tests here are OFFLINE: the TTS model is never loaded. Only the pure
functions (placement, mixing, resampling, lyric parsing, MIDI timing) and
the ``save_midi`` tempo fix are exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from promptmusic.vocals import (
    _fade_edges,
    _midi_timing,
    _read_lyrics,
    _resample,
    mix_vocals,
    plan_placements,
)


# --------------------------------------------------------------------------- #
# plan_placements
# --------------------------------------------------------------------------- #
class TestPlanPlacements:
    def test_zero_lines_returns_empty(self):
        assert plan_placements(0, bar_seconds=2.0, total_seconds=60.0) == []

    def test_lines_spread_over_song_on_bar_boundaries(self):
        starts = plan_placements(4, bar_seconds=2.0, total_seconds=80.0)
        assert len(starts) == 4
        assert starts[0] == 0.0
        # default stride = n_bars // n_lines = 40 // 4 = 10 bars = 20 s
        assert starts == [0.0, 20.0, 40.0, 60.0]
        # every start is a multiple of the bar length
        assert all(abs(s % 2.0) < 1e-9 for s in starts)

    def test_explicit_stride_bars(self):
        starts = plan_placements(3, bar_seconds=1.5, total_seconds=60.0, stride_bars=4)
        assert starts == [0.0, 6.0, 12.0]

    def test_colliding_lines_are_pushed_apart(self):
        # 4-second lines every 2 seconds would overlap -> pushed forward
        starts = plan_placements(
            3,
            bar_seconds=2.0,
            total_seconds=120.0,
            line_seconds=[4.0, 4.0, 1.0],
        )
        assert starts[1] >= starts[0] + 4.0
        assert starts[2] >= starts[1] + 4.0

    def test_last_line_never_exceeds_song(self):
        total = 30.0
        starts = plan_placements(5, bar_seconds=2.0, total_seconds=total)
        assert all(0.0 <= s <= total for s in starts)

    def test_many_lines_still_fit(self):
        # more lines than bars: stride floors at 1 bar, starts stay sorted
        starts = plan_placements(50, bar_seconds=2.0, total_seconds=60.0)
        assert starts == sorted(starts)
        assert len(starts) == 50


# --------------------------------------------------------------------------- #
# mix_vocals / _resample / _fade_edges
# --------------------------------------------------------------------------- #
class TestMixing:
    def test_mix_adds_scaled_vocal(self):
        inst = np.zeros((100, 2), dtype=np.float32)
        vocal = np.ones(100, dtype=np.float32)
        mixed = mix_vocals(inst, vocal, vocal_gain_db=0.0)  # unity gain
        assert mixed.shape == (100, 2)
        np.testing.assert_allclose(mixed[:, 0], 1.0, rtol=1e-6)

    def test_negative_gain_attenuates(self):
        inst = np.zeros((10, 2), dtype=np.float32)
        vocal = np.ones(10, dtype=np.float32)
        mixed = mix_vocals(inst, vocal, vocal_gain_db=-6.0)
        assert 0.4 < mixed[0, 0] < 0.6  # -6 dB ~= 0.5

    def test_mono_instrumental_is_promoted(self):
        inst = np.zeros(50, dtype=np.float32)
        vocal = np.ones(50, dtype=np.float32)
        mixed = mix_vocals(inst, vocal, vocal_gain_db=0.0)
        assert mixed.shape == (50, 1)

    def test_vocal_longer_than_instrumental_is_clipped(self):
        inst = np.zeros((10, 2), dtype=np.float32)
        vocal = np.ones(100, dtype=np.float32)
        mixed = mix_vocals(inst, vocal, vocal_gain_db=0.0)
        assert mixed.shape == (10, 2)  # no broadcast blow-up

    def test_resample_noop_when_rates_match(self):
        x = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        assert _resample(x, 44100, 44100) is x

    def test_resample_16k_to_44k1_changes_length(self):
        x = np.zeros(16000, dtype=np.float32)  # 1 s at 16 kHz
        y = _resample(x, 16000, 44100)
        assert abs(len(y) - 44100) <= 2

    def test_fade_edges_zeroes_boundaries(self):
        x = np.ones(1000, dtype=np.float32)
        faded = _fade_edges(x, 100)
        assert faded[0] == 0.0
        assert faded[-1] == 0.0
        assert faded[500] == 1.0  # middle untouched


# --------------------------------------------------------------------------- #
# _read_lyrics
# --------------------------------------------------------------------------- #
class TestReadLyrics:
    def test_blank_lines_are_dropped(self):
        assert _read_lyrics("line one\n\n\n  line two  \n", None) == [
            "line one",
            "line two",
        ]

    def test_empty_input_is_empty_list(self):
        assert _read_lyrics(None, None) == []

    def test_file_wins_over_inline_text(self, tmp_path):
        f = tmp_path / "lyrics.txt"
        f.write_text("from file\n", encoding="utf-8")
        assert _read_lyrics("inline", f) == ["from file"]

    def test_tamil_utf8_survives(self, tmp_path):
        f = tmp_path / "ta.txt"
        f.write_text("காதல் எனும் நதியில்\n", encoding="utf-8")
        lines = _read_lyrics(None, f)
        assert lines == ["காதல் எனும் நதியில்"]


# --------------------------------------------------------------------------- #
# _midi_timing (tempo robustness)
# --------------------------------------------------------------------------- #
class TestMidiTiming:
    def _write_midi(self, path, qpm: float):
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=qpm if qpm > 0 else 1.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=2.0)
        )
        pm.instruments.append(inst)
        pm.write(str(path))
        if qpm <= 0:
            # pretty_midi refuses to write a 0-tempo file directly, so
            # simulate the broken generated MIDI by patching the tempo
            # event post-write via the in-memory object instead.
            pass
        return pm

    def test_valid_tempo_gives_expected_bar(self, tmp_path):
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(80, 60, 0.0, 2.0))
        pm.instruments.append(inst)
        p = tmp_path / "ok.mid"
        pm.write(str(p))

        bar, total = _midi_timing(p)
        assert abs(bar - 2.0) < 0.05  # 4 beats at 120 BPM
        assert total > 0

    def test_waltz_time_signature_adjusts_bar(self, tmp_path):
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        pm.time_signature_changes.append(
            pretty_midi.TimeSignature(numerator=3, denominator=4, time=0.0)
        )
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(80, 60, 0.0, 2.0))
        pm.instruments.append(inst)
        p = tmp_path / "waltz.mid"
        pm.write(str(p))

        bar, _ = _midi_timing(p)
        assert abs(bar - 1.5) < 0.05  # 3 beats at 120 BPM


# --------------------------------------------------------------------------- #
# save_midi tempo sanitization (the 0-BPM bug)
# --------------------------------------------------------------------------- #
class TestTempoSanitization:
    def test_zero_qpm_is_replaced(self, tmp_path):
        import symusic

        from promptmusic.data.tokenizer import MusicTokenizer

        tok = MusicTokenizer({"tempo_range": (40, 250)})
        score = symusic.Score(480)
        score.tempos.append(symusic.Tempo(time=0, qpm=0.0))

        tok._sanitize_tempos(score)

        assert score.tempos[0].qpm == pytest.approx(145.0)  # (40+250)/2

    def test_valid_tempo_is_kept(self):
        import symusic

        from promptmusic.data.tokenizer import MusicTokenizer

        tok = MusicTokenizer({})
        score = symusic.Score(480)
        score.tempos.append(symusic.Tempo(time=0, qpm=96.0))

        tok._sanitize_tempos(score)

        assert score.tempos[0].qpm == pytest.approx(96.0)

    def test_out_of_range_tempo_is_clamped(self):
        import symusic

        from promptmusic.data.tokenizer import MusicTokenizer

        tok = MusicTokenizer({"tempo_range": (40, 250)})
        score = symusic.Score(480)
        score.tempos.append(symusic.Tempo(time=0, qpm=900.0))

        tok._sanitize_tempos(score)

        assert score.tempos[0].qpm == pytest.approx(145.0)


# --------------------------------------------------------------------------- #
# add_vocals (offline: monkeypatched synthesis)
# --------------------------------------------------------------------------- #
class TestAddVocalsOffline:
    def _prepare(self, tmp_path):
        """A 8 s instrumental WAV + MIDI at 120 BPM."""
        import pretty_midi
        import soundfile as sf

        sr = 44100
        wav = tmp_path / "inst.wav"
        # 0.5 s of tone, 7.5 s of silence -> energy only at the start
        t = np.arange(int(0.5 * sr)) / sr
        tone = 0.3 * np.sin(2 * np.pi * 220 * t)
        audio = np.concatenate([tone, np.zeros(8 * sr - len(t))])
        sf.write(wav, np.stack([audio, audio], axis=1), sr)

        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(80, 60, 0.0, 1.0))
        pm.instruments.append(inst)
        midi = tmp_path / "song.mid"
        pm.write(str(midi))
        return midi, wav

    def test_mixed_output_contains_speech_energy(self, tmp_path, monkeypatch):
        from promptmusic import vocals as V

        midi, wav = self._prepare(tmp_path)

        # fake TTS: 1 s 440 Hz tone per line at 16 kHz
        fake_rate = 16000
        fake = np.sin(2 * np.pi * 440 * np.arange(fake_rate) / fake_rate)
        monkeypatch.setattr(
            V, "synthesize_lines", lambda lines, **kw: ([fake.copy() for _ in lines], fake_rate)
        )

        result = V.add_vocals(
            midi, wav, lyrics="line one\nline two", out_path=tmp_path / "out.wav"
        )
        assert result.n_lines == 2
        assert result.duration_sec == pytest.approx(8.0, abs=0.1)

        import soundfile as sf

        mixed, sr = sf.read(result.wav_path, dtype="float32")
        inst, _ = sf.read(wav, dtype="float32")
        n = min(len(mixed), len(inst))
        diff = mixed[:n, 0] - inst[:n, 0]
        # the vocal layer must be audible at the planned placements
        for start in result.placements:
            i = int(start * sr)
            seg = diff[i : i + int(0.4 * sr)]
            assert np.sqrt((seg**2).mean()) > 0.05

    def test_raises_without_lyrics(self, tmp_path):
        from promptmusic import vocals as V

        midi, wav = self._prepare(tmp_path)
        with pytest.raises(ValueError, match="no lyrics"):
            V.add_vocals(midi, wav)

    def test_no_peak_clipping(self, tmp_path, monkeypatch):
        from promptmusic import vocals as V

        midi, wav = self._prepare(tmp_path)
        loud = np.ones(16000, dtype=np.float32)  # full-scale fake speech
        monkeypatch.setattr(
            V, "synthesize_lines", lambda lines, **kw: ([loud.copy() for _ in lines], 16000)
        )
        result = V.add_vocals(midi, wav, lyrics="loud\nlines", out_path=tmp_path / "loud.wav")
        import soundfile as sf

        mixed, _ = sf.read(result.wav_path, dtype="float32")
        assert float(np.abs(mixed).max()) <= 1.0 + 1e-6
