"""Analysing a reference music file so the model can imitate its character.

Two kinds of reference
----------------------
* **A MIDI file.** We can read everything: tempo, key, time signature, which
  instruments play, how dense the notes are, the pitch range. That analysis is
  turned into words and appended to the prompt, and the file's opening bars
  can also be used to *prime* the decoder (see
  :func:`promptmusic.generate.generate` with ``reference_midi=``).

* **An audio file** (WAV/MP3/FLAC). Much harder - there is no note list, only a
  waveform. We therefore extract *statistical* descriptors with librosa:
  tempo, estimated key, energy over time, spectral centroid (brightness) and
  spectral rolloff. These do not tell us the melody, but they do capture the
  feel well enough to steer the prompt.

Why not transcribe the audio to MIDI automatically?
----------------------------------------------------
Automatic music transcription is a research field of its own, and current
open models are unreliable on multi-instrument music. Rather than bolt on a
weak transcription step we analyse the audio and describe it in words, which
the text encoder already understands. A transcription backend can be plugged
in later without changing this module's interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import logger

LOG = logger("promptmusic.analyze")

__all__ = [
    "analyze_midi",
    "analyze_audio",
    "analysis_to_prompt_fragment",
    "describe_analysis",
    "analyze_to_file",
]


# --------------------------------------------------------------------------- #
# MIDI analysis
# --------------------------------------------------------------------------- #
def analyze_midi(path: str | Path) -> dict[str, Any]:
    """Extract musical descriptors from a MIDI file.

    Returns a dict with keys like ``tempo``, ``key``, ``time_signature``,
    ``duration_sec``, ``num_tracks``, ``num_notes``, ``instruments``,
    ``pitch_range``, ``note_density`` and ``velocity_mean``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"reference MIDI not found: {p}")

    # --- symusic is the fast, dependency-light path ------------------------ #
    try:
        analysis = _analyze_midi_symusic(p)
        if analysis:
            return analysis
    except Exception as exc:  # noqa: BLE001 - fall through to pretty_midi
        LOG.debug("symusic analysis failed (%s); trying pretty_midi", exc)

    # --- pretty_midi fallback ---------------------------------------------- #
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(p))
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        raise ValueError(f"no notes in {p}")

    tempos = pm.get_tempo_changes()[1]
    pitches = [n.pitch for n in notes]
    return {
        "path": str(p),
        "backend": "pretty_midi",
        "tempo": float(tempos[0]) if len(tempos) else 0.0,
        "key": _estimate_key_midi(pitches),
        "time_signature": _first_time_signature(pm),
        "duration_sec": float(pm.get_end_time()),
        "num_tracks": len(pm.instruments),
        "num_notes": len(notes),
        "instruments": [inst.name or f"program_{inst.program}" for inst in pm.instruments],
        "programs": sorted({inst.program for inst in pm.instruments}),
        "pitch_range": [int(min(pitches)), int(max(pitches))],
        "note_density": round(len(notes) / max(pm.get_end_time(), 1e-6), 2),
        "velocity_mean": round(sum(n.velocity for n in notes) / len(notes), 1),
    }


def _analyze_midi_symusic(path: Path) -> dict[str, Any] | None:
    from symusic import Score

    score = Score(str(path), ttype="second")
    notes = [n for track in score.tracks for n in track.notes]
    if not notes:
        return None

    tempo = 0.0
    if len(score.tempos) > 0:
        tempo = float(score.tempos[0].tempo)

    timesig = ""
    if len(score.time_signatures) > 0:
        ts = score.time_signatures[0]
        timesig = f"{ts.numerator}/{ts.denominator}"

    pitches = [int(n.pitch) for n in notes]
    programs = sorted({int(t.program) for t in score.tracks if getattr(t, "program", None) is not None})

    return {
        "path": str(path),
        "backend": "symusic",
        "tempo": tempo,
        "key": _estimate_key_midi(pitches),
        "time_signature": timesig,
        "duration_sec": float(score.end - score.start),
        "num_tracks": len(score.tracks),
        "num_notes": len(notes),
        "programs": programs,
        "instruments": [_gm_name(p) for p in programs],
        "pitch_range": [min(pitches), max(pitches)],
        "note_density": round(len(notes) / max(float(score.end - score.start), 1e-6), 2),
        "velocity_mean": round(sum(int(n.velocity) for n in notes) / len(notes), 1),
    }


def _estimate_key_midi(pitches: list[int]) -> str:
    """Krumhansl-Schmuckler key estimate from a MIDI pitch histogram."""
    if not pitches:
        return ""
    import numpy as np

    hist = np.zeros(12, dtype=np.float64)
    for p in pitches:
        hist[p % 12] += 1
    return _krumhansl_key(hist)


# --------------------------------------------------------------------------- #
# Audio analysis
# --------------------------------------------------------------------------- #
def analyze_audio(path: str | Path, sr: int = 22050) -> dict[str, Any]:
    """Extract statistical descriptors from an audio file with librosa.

    Returns ``tempo``, ``key``, ``duration_sec``, ``rms_mean``, ``rms_std``,
    ``spectral_centroid_mean`` (brightness) and ``spectral_rolloff_mean``.

    Raises:
        ImportError: if ``librosa`` is not installed
            (``pip install 'promptmusic[analysis]'``).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"reference audio not found: {p}")
    try:
        import librosa
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "librosa is required for audio analysis. Install it with:\n"
            "    pip install 'promptmusic[analysis]'\n"
            "or:  pip install librosa soundfile"
        ) from exc

    y, _sr = librosa.load(str(p), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"audio file is empty: {p}")

    tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = _krumhansl_key(chroma.mean(axis=1)) if chroma.size else ""

    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    return {
        "path": str(p),
        "backend": "librosa",
        "tempo": tempo_val,
        "key": key,
        "duration_sec": float(len(y) / sr),
        "rms_mean": round(float(rms.mean()), 5),
        "rms_std": round(float(rms.std()), 5),
        "spectral_centroid_mean": round(float(centroid.mean()), 1),
        "spectral_rolloff_mean": round(float(rolloff.mean()), 1),
    }


# --------------------------------------------------------------------------- #
# Key estimation (shared)
# --------------------------------------------------------------------------- #
_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _krumhansl_key(profile_12: Any) -> str:
    """Match a 12-bin pitch/chroma profile against the Krumhansl key profiles."""
    import numpy as np

    p = np.asarray(profile_12, dtype=np.float64)
    if p.sum() <= 0:
        return ""
    p = p / p.sum()
    best, best_name = -2.0, ""

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / denom) if denom > 0 else 0.0

    for shift in range(12):
        rolled = np.roll(p, -shift)
        maj = _corr(rolled, np.asarray(_MAJOR) / sum(_MAJOR))
        mnr = _corr(rolled, np.asarray(_MINOR) / sum(_MINOR))
        if maj > best:
            best, best_name = maj, f"{_NAMES[shift]} major"
        if mnr > best:
            best, best_name = mnr, f"{_NAMES[shift]} minor"
    return best_name


# --------------------------------------------------------------------------- #
# Describing an analysis in words
# --------------------------------------------------------------------------- #
def analysis_to_prompt_fragment(analysis: dict[str, Any]) -> str:
    """Turn an analysis dict into a short phrase to append to a prompt.

    Example: ``"(tempo 120 bpm, key C major, instruments: piano, strings)"``.
    """
    bits: list[str] = []
    if analysis.get("tempo"):
        bits.append(f"tempo {float(analysis['tempo']):.0f} bpm")
    if analysis.get("key"):
        bits.append(f"key {analysis['key']}")
    if analysis.get("time_signature"):
        bits.append(f"time signature {analysis['time_signature']}")
    if analysis.get("instruments"):
        names = ", ".join(_gm_name(p) if isinstance(p, int) else str(p)
                          for p in list(analysis["instruments"])[:6])
        bits.append(f"instruments: {names}")
    if analysis.get("pitch_range"):
        lo, hi = analysis["pitch_range"]
        bits.append(f"pitch range {lo}-{hi}")
    if analysis.get("spectral_centroid_mean") is not None:
        c = float(analysis["spectral_centroid_mean"])
        bits.append("bright timbre" if c > 1800 else "dark timbre")
    if analysis.get("duration_sec"):
        bits.append(f"{float(analysis['duration_sec']):.0f} seconds")
    return "(" + ", ".join(bits) + ")" if bits else ""


def describe_analysis(analysis: dict[str, Any]) -> str:
    """A one-line human-readable summary for the CLI."""
    parts = [
        f"tempo {analysis.get('tempo', 0):.0f} bpm" if analysis.get("tempo") else "tempo unknown",
        analysis.get("key") or "key unknown",
        f"{analysis.get('num_notes', 0)} notes",
        f"{analysis.get('duration_sec', 0):.1f}s",
    ]
    return ", ".join(parts)


def analyze_to_file(analysis: dict[str, Any], path: str | Path) -> Path:
    """Write an analysis dict to JSON (used by ``promptmusic analyze --json``).

    The analysis may contain numpy scalars (e.g. ``np.int64`` note counts from
    symusic), which the stdlib ``json`` encoder rejects. ``default`` converts
    them (and anything else exotic) to plain Python values instead of raising.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _coerce(obj: Any) -> Any:
        # numpy scalar -> python scalar
        item = getattr(obj, "item", None)
        if callable(item):
            try:
                return item()
            except Exception:  # noqa: BLE001 - fall through to str()
                pass
        return str(obj)

    p.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, default=_coerce),
        encoding="utf-8",
    )
    return p


# --------------------------------------------------------------------------- #
# General MIDI instrument names (used to make reports readable)
# --------------------------------------------------------------------------- #
_GM_NAMES = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)",
    "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar",
    "Distortion Guitar", "Guitar Harmonics",
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    "Violin", "Viola", "Cello", "Contrabass",
    "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
    "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
    "Oboe", "English Horn", "Bassoon", "Clarinet",
    "Piccolo", "Flute", "Recorder", "Pan Flute",
    "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass+lead)",
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    "Sitar", "Banjo", "Shamisen", "Koto",
    "Kalimba", "Bag pipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
]


def _gm_name(program: int) -> str:
    """General MIDI program number -> readable instrument name."""
    if isinstance(program, str):
        return program
    idx = int(program)
    return _GM_NAMES[idx] if 0 <= idx < len(_GM_NAMES) else f"program_{idx}"


def _first_time_signature(pm: Any) -> str:
    ts = getattr(pm, "time_signature_changes", None) or []
    if not ts:
        return ""
    first = ts[0]
    return f"{first.numerator}/{first.denominator}"
