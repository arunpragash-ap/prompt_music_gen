"""Offline tests for melody-aware singing (singing.py) and mastering.

Nothing here downloads a model: the TTS call is monkeypatched with a
synthetic voiced tone. The tests cover the parts where bugs are silent and
expensive - syllabification, syllable/note alignment, pitch shifting, gain
staging, and MIDI humanisation.
"""

from __future__ import annotations

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

import promptmusic.singing as singing
from promptmusic.master import (
    compress,
    convolve_reverb,
    duck,
    make_impulse_response,
    master_mix,
    soft_limit,
)
from promptmusic.render import humanize_midi
from promptmusic.singing import (
    MelodyNote,
    SungUnit,
    allocate_line_sections,
    extract_lead_track,
    extract_melody,
    fit_duration,
    fit_pitch,
    estimate_f0,
    pitch_shift,
    plan_line,
    plan_line_grid,
    quantize_melody,
    render_line,
    segment_phrases,
    simplify_melody,
    smooth_pitches,
    split_english_syllables,
    split_syllables,
    split_tamil_syllables,
    time_stretch,
    vibrato,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tone(freq: float = 180.0, seconds: float = 0.5, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    wave = sum(np.sin(2 * np.pi * freq * h * t) / h for h in (1, 2, 3, 4))
    env = np.minimum(1.0, np.minimum(t / 0.02, (seconds - t) / 0.02))
    return (wave * env).astype(np.float32)


def _fake_pm(notes: list[tuple[float, float, int]], *, program: int = 0, drum: bool = False):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=program, is_drum=drum)
    for start, end, pitch in notes:
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=pitch, start=float(start), end=float(end))
        )
    pm.instruments.append(inst)
    return pm


# --------------------------------------------------------------------------- #
# Syllabification
# --------------------------------------------------------------------------- #
class TestTamilSyllables:
    @pytest.mark.parametrize(
        "word,expected",
        [
            ("காதல்", ["கா", "தல்"]),
            ("நதியில்", ["ந", "தி", "யில்"]),
            ("என்", ["என்"]),
            ("ஸ்திரம்", ["ஸ்தி", "ரம்"]),
            ("மிதக்கிறேன்", ["மி", "தக்", "கி", "றேன்"]),
            ("நான்", ["நான்"]),
        ],
    )
    def test_words(self, word, expected):
        assert split_tamil_syllables(word) == expected

    def test_whole_line_keeps_order(self):
        line = "காதல் எனும் நதியில் நான் மிதக்கிறேன்"
        sylls = split_tamil_syllables(line)
        assert len(sylls) == 12
        assert "".join(sylls).replace(" ", "") == line.replace(" ", "")
        assert sylls[0] == "கா" and sylls[-1] == "றேன்"

    def test_punctuation_and_digits_survive(self):
        assert split_tamil_syllables("காதல், 2 முறை!") == ["கா", "தல்", "2", "மு", "றை"]

    def test_empty_and_punctuation_only(self):
        assert split_tamil_syllables("") == []
        assert split_tamil_syllables("... , !") == []


class TestEnglishSyllables:
    @pytest.mark.parametrize(
        "word,count",
        [("love", 1), ("singing", 2), ("beautiful", 3), ("tonight", 2), ("a", 1)],
    )
    def test_counts(self, word, count):
        assert len(split_english_syllables(word)) == count

    def test_preserves_letters(self):
        words = "a romantic love song under the stars".split()
        out = split_english_syllables(" ".join(words))
        assert "".join(out) == "".join(words)

    def test_detects_script(self):
        assert split_syllables("காதல்")[0] == "கா"
        assert split_syllables("love") == ["love"]


# --------------------------------------------------------------------------- #
# Melody extraction
# --------------------------------------------------------------------------- #
class TestMelody:
    def test_skyline_takes_the_highest_note(self):
        pm = _fake_pm([(0.0, 1.0, 60), (0.0, 1.0, 67)])
        mel = extract_melody(pm)
        assert len(mel) == 1 and mel[0].pitch == 67

    def test_drums_are_ignored(self):
        pm = _fake_pm([(0.0, 1.0, 90)], drum=True)
        pm.instruments[0].notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0)
        )
        assert all(n.pitch != 90 for n in extract_melody(pm))

    def test_lead_track_prefers_monophonic_instrument(self):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        piano = pretty_midi.Instrument(program=0)
        for i in range(12):
            piano.notes.append(pretty_midi.Note(90, 60 + i % 5, i * 0.2, i * 0.2 + 0.18))
        pad = pretty_midi.Instrument(program=89)
        pad.notes.append(pretty_midi.Note(90, 84, 0.0, 2.0))
        pm.instruments.extend([piano, pad])
        lead = extract_lead_track(pm)
        assert len(lead) == 12
        assert max(n.pitch for n in lead) <= 70

    def test_simplify_drops_slivers_and_caps_density(self):
        notes = [MelodyNote(i * 0.05, i * 0.05 + 0.03, 60 + i % 7) for i in range(40)]
        out = simplify_melody(notes, min_duration=0.1, max_rate=2.0)
        assert all(n.duration >= 0.1 for n in out)
        # 2 s of source at 2 notes/s -> at most ~5 notes
        assert len(out) <= 5

    def test_quantize_puts_notes_on_the_grid(self):
        notes = [
            MelodyNote(0.00, 0.12, 60),
            MelodyNote(0.12, 0.24, 62),
            MelodyNote(0.36, 0.48, 64),
        ]
        out = quantize_melody(notes, bpm=120, subdivision=2)
        assert out
        grid = 60 / 120 / 2
        for n in out:
            assert abs(n.start / grid - round(n.start / grid)) < 1e-6

    def test_quantize_merges_repeats(self):
        notes = [MelodyNote(i * 0.12, i * 0.12 + 0.12, 60) for i in range(8)]
        out = quantize_melody(notes, bpm=120, subdivision=2)
        assert len(out) == 1
        assert out[0].duration > 0.3

    def test_phrases_split_at_rests(self):
        notes = [
            MelodyNote(0.0, 0.4, 60),
            MelodyNote(0.4, 0.8, 62),
            MelodyNote(2.0, 2.4, 64),
            MelodyNote(2.4, 2.8, 65),
        ]
        phrases = segment_phrases(notes, min_gap=0.45, min_phrase=0.5)
        assert len(phrases) == 2
        assert phrases[0][0].pitch == 60 and phrases[1][0].pitch == 64

    def test_smooth_pitches_caps_large_leaps(self):
        out = smooth_pitches([60, 61, 90, 62, 63])
        assert len(out) == 5
        assert all(abs(b - a) <= 7 for a, b in zip(out, out[1:]))

    def test_smooth_pitches_keeps_octave_direction(self):
        # Regression: folding by octaves turned an octave leap into a unison,
        # so a melody alternating 55/67 became one repeated pitch.
        out = smooth_pitches([55] * 7 + [67] * 6)
        assert out[-1] > out[0]
        assert out[-1] - out[0] >= 5


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
class TestAlignment:
    def test_more_notes_than_syllables_holds_notes(self):
        notes = [MelodyNote(i * 0.5, i * 0.5 + 0.45, 60 + i) for i in range(6)]
        units = plan_line(["ka", "dhal", "um"], notes)
        assert [u.syllable for u in units] == ["ka", "dhal", "um"]
        assert [u.n_notes for u in units] == [2, 2, 2]
        assert units[0].start == 0.0
        assert units[1].start == 1.0

    def test_fewer_notes_than_syllables_splits_notes(self):
        notes = [MelodyNote(0.0, 1.0, 60), MelodyNote(1.0, 2.0, 64)]
        units = plan_line(list("abcdef"), notes)
        assert len(units) == 6
        assert [u.syllable for u in units] == list("abcdef")
        assert all(u.duration > 0 for u in units)
        assert units[0].start < units[1].start < units[2].start

    def test_grid_plan_uses_every_syllable_once(self):
        notes = [MelodyNote(i * 0.4, i * 0.4 + 0.35, 60 + (i % 4)) for i in range(6)]
        sylls = list("abcdefghij")
        units = plan_line_grid(sylls, notes)
        assert [u.syllable for u in units] == sylls
        assert all(u.duration >= 0.1 for u in units)
        assert all(a.start <= b.start for a, b in zip(units, units[1:]))

    def test_sections_are_ordered_and_cover_notes(self):
        notes = [MelodyNote(i * 0.5, i * 0.5 + 0.4, 60 + i % 5) for i in range(20)]
        sections = allocate_line_sections(notes, 3, [4, 4, 4])
        assert len(sections) == 3
        assert all(sections)
        assert sections[0][0].start < sections[1][0].start < sections[2][0].start
        covered = sum(len(s) for s in sections)
        assert covered <= len(notes)
        assert covered >= len(notes) - 1

    def test_sections_survive_more_lines_than_notes(self):
        notes = [MelodyNote(i * 0.5, i * 0.5 + 0.4, 60) for i in range(3)]
        sections = allocate_line_sections(notes, 5, [2] * 5)
        assert 0 < len(sections) <= 5
        assert all(sections)

    def test_fit_pitch_keeps_voice_in_range(self):
        for pitch in (30, 60, 83, 100):
            f = fit_pitch(pitch, natural_f0=170.0)
            assert 170 * 2 ** (-7 / 12) <= f <= 170 * 2 ** (7 / 12)

    def test_fit_pitch_preserves_direction(self):
        # Regression: octave folding mapped an octave leap to a *unison*,
        # turning repeated-octave melodies into a monotone chant.
        low = fit_pitch(60, natural_f0=170.0, center_pitch=64)
        high = fit_pitch(72, natural_f0=170.0, center_pitch=64)
        assert 12 * np.log2(high / low) > 5.0
        # The melody's centre lands on the speaker's F0.
        assert fit_pitch(64, natural_f0=170.0, center_pitch=64) == pytest.approx(170.0)


# --------------------------------------------------------------------------- #
# Audio primitives
# --------------------------------------------------------------------------- #
class TestAudioPrimitives:
    def test_estimate_f0_on_a_sine(self):
        sr = 16000
        f = estimate_f0(_tone(180.0, 0.5, sr), sr)
        assert abs(f - 180.0) / 180.0 < 0.05

    def test_pitch_shift_preserves_length_and_changes_pitch(self):
        sr = 16000
        x = _tone(180.0, 0.5, sr)
        y = pitch_shift(x, 4.0, sr)
        assert len(y) == len(x)
        f = estimate_f0(y, sr)
        assert abs(f - 180.0 * 2 ** (4 / 12)) / (180.0 * 2 ** (4 / 12)) < 0.08

    def test_negative_pitch_shift(self):
        sr = 16000
        x = _tone(180.0, 0.5, sr)
        y = pitch_shift(x, -5.0, sr)
        assert abs(estimate_f0(y, sr) - 180.0 * 2 ** (-5 / 12)) < 10

    def test_time_stretch_preserves_pitch(self):
        sr = 16000
        x = _tone(180.0, 0.5, sr)
        y = time_stretch(x, 1.6, sr)
        assert abs(len(y) - int(len(x) * 1.6)) < int(0.02 * sr)
        assert abs(estimate_f0(y, sr) - 180.0) < 10

    def test_vibrato_is_finite_and_same_length(self):
        sr = 16000
        x = _tone(180.0, 0.5, sr)
        y = vibrato(x, sr)
        assert len(y) == len(x) and np.isfinite(y).all()

    def test_fit_duration_hits_target(self):
        sr = 16000
        x = _tone(180.0, 0.5, sr)
        for target in (0.3, 0.9, 2.4):
            y = fit_duration(x, target, sr)
            assert abs(len(y) / sr - target) < 0.06

    def test_render_line_places_sung_notes(self):
        sr = 16000
        units = [
            SungUnit("ka", 0.0, 0.5, 60),
            SungUnit("dhal", 0.55, 0.5, 64),
            SungUnit("um", 1.1, 0.5, 67),
        ]
        out = render_line(_tone(180.0, 1.0, sr), units, sr=sr, natural_f0=180.0)
        assert len(out) / sr > 1.4
        assert np.isfinite(out).all() and np.max(np.abs(out)) <= 1.0
        # Each note centre should be near its target frequency.
        for u in units:
            seg = out[int((u.start + 0.15) * sr) : int((u.start + 0.35) * sr)]
            if np.max(np.abs(seg)) < 1e-4:
                continue
            f = estimate_f0(seg, sr)
            target = fit_pitch(u.pitch, 180.0)
            assert abs(12 * np.log2(max(f, 1) / target)) < 2.0


# --------------------------------------------------------------------------- #
# Mastering
# --------------------------------------------------------------------------- #
class TestMastering:
    def test_impulse_response_decays(self):
        ir = make_impulse_response(8000, rt60=1.0, length=1.0)
        assert ir.shape[1] == 2 and len(ir) == 8000
        assert np.isfinite(ir).all()
        early = np.abs(ir[:1000]).mean()
        late = np.abs(ir[-1000:]).mean()
        assert early > late

    def test_soft_limit_only_reduces_peaks(self):
        quiet = np.array([0.1, -0.2], dtype=np.float32)
        assert np.allclose(soft_limit(quiet, ceiling=0.891), quiet)
        loud = np.linspace(-2.0, 2.0, 1000, dtype=np.float32)
        out = soft_limit(loud, ceiling=0.891)
        assert np.max(np.abs(out)) <= 0.891 + 1e-5

    def test_compress_reduces_dynamic_range(self):
        sr = 8000
        x = np.zeros(sr, dtype=np.float32)
        x[1000:2000] = 0.9
        x[3000:5000] = 0.1
        y = compress(x, sr, threshold_db=-20, ratio=4.0, attack=0.005, release=0.05)
        assert np.max(np.abs(y)) < np.max(np.abs(x))
        assert np.sqrt(np.mean(y**2)) > 0.0

    def test_duck_reduces_target_under_sidechain(self):
        sr = 8000
        target = np.ones(sr, dtype=np.float32)
        sc = np.zeros(sr, dtype=np.float32)
        sc[: sr // 2] = 1.0
        out = duck(target, sc, sr, depth=0.5, attack=0.005, release=0.05)
        assert out[: sr // 2].mean() < 0.8
        assert out[-100:].mean() > 0.9

    def test_convolve_reverb_keeps_shape(self):
        sr = 8000
        x = np.zeros((sr, 2), dtype=np.float32)
        x[100, 0] = 1.0
        ir = make_impulse_response(sr, length=0.5)
        y = convolve_reverb(x, ir, wet=1.0)
        assert y.shape == x.shape
        assert np.isfinite(y).all()
        assert np.sum(np.abs(y)) > 0

    def test_master_mix_balances_and_limits(self):
        sr = 8000
        n = sr * 3
        t = np.arange(n) / sr
        inst = np.stack(
            [0.2 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 330 * t)], axis=1
        ).astype(np.float32)
        voc = np.zeros(n, dtype=np.float32)
        voc[int(1.0 * sr) : int(2.0 * sr)] = 0.9 * np.sin(
            2 * np.pi * 180 * t[: sr]
        ).astype(np.float32)
        mix = master_mix(inst, voc, sr, vocal_gain_db=1.5, reverb=0.0)
        assert mix.shape == inst.shape
        assert np.isfinite(mix).all()
        assert np.max(np.abs(mix)) <= 0.892
        rms = float(np.sqrt(np.mean(mix**2)))
        peak = float(np.max(np.abs(mix)))
        assert 0.02 < rms < 0.5
        assert 20 * np.log10(peak / rms) > 5.0  # not a crushed wall


# --------------------------------------------------------------------------- #
# Humanisation
# --------------------------------------------------------------------------- #
class TestHumanize:
    def _write(self, path):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        piano = pretty_midi.Instrument(program=0)
        for i in range(4):
            piano.notes.append(pretty_midi.Note(90, 60 + i, i * 0.5, i * 0.5 + 0.4))
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        drums.notes.append(pretty_midi.Note(100, 36, 0.0, 0.1))
        pm.instruments.extend([piano, drums])
        pm.write(str(path))
        return pm

    def test_timing_and_velocity_change_within_bounds(self, tmp_path):
        src = tmp_path / "in.mid"
        self._write(src)
        out = humanize_midi(src, tmp_path / "out.mid", timing=0.006, velocity=5, seed=1)
        a = pretty_midi.PrettyMIDI(str(src))
        b = pretty_midi.PrettyMIDI(str(out))
        for ai, bi in zip(a.instruments, b.instruments):
            for an, bn in zip(ai.notes, bi.notes):
                assert abs(bn.start - an.start) < 0.05
                assert 1 <= bn.velocity <= 127
        # drums are untouched by default
        assert b.instruments[1].notes[0].start == a.instruments[1].notes[0].start

    def test_same_seed_is_deterministic(self, tmp_path):
        src = tmp_path / "in.mid"
        self._write(src)
        p1 = humanize_midi(src, tmp_path / "a.mid", seed=7)
        p2 = humanize_midi(src, tmp_path / "b.mid", seed=7)
        a, b = pretty_midi.PrettyMIDI(str(p1)), pretty_midi.PrettyMIDI(str(p2))
        assert [n.start for n in a.instruments[0].notes] == [
            n.start for n in b.instruments[0].notes
        ]


# --------------------------------------------------------------------------- #
# End-to-end (no model download)
# --------------------------------------------------------------------------- #
def test_sing_song_end_to_end(tmp_path, monkeypatch):
    sr = 16000

    def fake_tts(lines, *, voice, device):
        out = []
        for _ in lines:
            t = np.arange(int(0.9 * sr)) / sr
            wave = sum(np.sin(2 * np.pi * 180 * h * t) / h for h in (1, 2, 3))
            env = np.minimum(1.0, np.minimum(t / 0.02, (0.9 - t) / 0.02))
            out.append((wave * env).astype(np.float32))
        return out, sr

    monkeypatch.setattr("promptmusic.vocals.synthesize_lines", fake_tts)

    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    for i in range(14):
        inst.notes.append(pretty_midi.Note(90, 60 + (i % 5), 0.5 + i * 0.45, 0.9 + i * 0.45))
    pm.instruments.append(inst)
    midi = tmp_path / "song.mid"
    pm.write(str(midi))

    n = int(8 * 44100)
    t = np.arange(n) / 44100
    pad = np.stack([0.03 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32)
    wav = tmp_path / "song.wav"
    sf.write(str(wav), pad, 44100)

    result = singing.sing_song(
        midi,
        lyrics="காதல் எனும் நதியில்\nஉன் கண்களின் நட்சத்திரம்",
        lyrics_file=None,
        out_path=tmp_path / "out.wav",
        instrumental_wav=wav,
        master=True,
        humanize=False,
    )
    assert (tmp_path / "out.wav").exists()
    assert result.n_lines == 2
    assert result.n_syllables == 7 + 8
    assert result.duration_sec > 7.0
    info = sf.info(str(tmp_path / "out.wav"))
    assert info.samplerate == 44100 and info.channels == 2
    assert result.n_melody_notes > 0


def test_tempo_is_read_from_the_tempo_map(tmp_path) -> None:
    # Regression: get_tempo_changes() returns (times, tempos). Unpacking it
    # backwards read the *time* of the first change (0.0), always failed the
    # sanity range, and silently fell back to 120 BPM - so every song was
    # quantised and bar-aligned at the wrong tempo.
    pm = pretty_midi.PrettyMIDI(initial_tempo=145.0)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    pm.instruments.append(inst)
    path = tmp_path / "tempo.mid"
    pm.write(str(path))

    assert singing._tempo_of(pm) == pytest.approx(145.0, abs=0.01)

    from promptmusic.vocals import _midi_timing

    bar, _total = _midi_timing(path)
    assert bar == pytest.approx(4 * 60.0 / 145.0, rel=1e-3)
