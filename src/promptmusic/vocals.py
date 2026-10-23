"""Vocal layer: put the lyrics *into* the audio, not just behind it.

Why this module exists
----------------------
The music model generates **MIDI** - notation, not audio. A MIDI file has
no voice track, and no soundfont can articulate words, so ``generate``
with lyrics produces an *instrumental* whose mood was conditioned on the
text but which never sings it. This module closes that gap with the best
approach that runs locally on an 8 GB laptop today:

    lyric line -> Tamil TTS -> bar-aligned placement -> mixed under the
    instrumental WAV

Honest limits (please read before judging the output)
------------------------------------------------------
This is a **spoken-sung vocal layer**, not singing voice synthesis (SVS).
True SVS aligns phonemes to a melody and synthesises sung pitch
contours; no open Tamil SVS model exists - DiffSinger-class systems need
datasets of sung Tamil that have not been released. Here the words are
audible and land on bar boundaries, but they do **not** follow the
generated melody. The upgrade path (melody-aware pitch shaping, or a
Tamil SVS model if one appears) is described in docs/07.

Default voice: ``facebook/mms-tts-tam`` (36 M params, 16 kHz, VITS).
**License: CC-BY-NC 4.0 - non-commercial use only.** Swap it for a
permissively licensed voice with ``--voice`` before shipping anything
commercial; see the licence table in docs/05.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .paths import HF_CACHE_DIR

LOG = logging.getLogger("promptmusic.vocals")

__all__ = [
    "DEFAULT_VOICE",
    "VocalResult",
    "add_vocals",
    "synthesize_lines",
    "plan_placements",
    "mix_vocals",
]

#: Tamil VITS speech model (16 kHz). CC-BY-NC 4.0 - see module docstring.
DEFAULT_VOICE = "facebook/mms-tts-tam"

#: Fallback when the MIDI carries no usable tempo (broken/zero events).
_FALLBACK_BPM = 120.0
#: Gap inserted when a line would otherwise collide with the next one.
_GAP_SEC = 0.25


@dataclass
class VocalResult:
    """What :func:`add_vocals` produced."""

    wav_path: str
    n_lines: int
    duration_sec: float
    voice: str
    backend: str
    placements: list[float] = field(default_factory=list)
    instrumental_path: str = ""
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Speech synthesis
# --------------------------------------------------------------------------- #
def synthesize_lines(
    lines: list[str],
    *,
    voice: str = DEFAULT_VOICE,
    device: str = "cpu",
) -> tuple[list[np.ndarray], int]:
    """Speak each lyric line; returns (mono float32 arrays, sample rate).

    Loaded lazily so the rest of the package (train, generate, render)
    never pays for the TTS model, and so tests can import this module
    without a transformers download.
    """
    import torch
    from transformers import AutoTokenizer, VitsModel

    LOG.info("loading speech model %s (one-time ~80 MB download on first use)", voice)
    # cache_dir follows the project convention (see paths.HF_CACHE_DIR).
    model = VitsModel.from_pretrained(voice, cache_dir=str(HF_CACHE_DIR / "hub"))
    tokenizer = AutoTokenizer.from_pretrained(voice, cache_dir=str(HF_CACHE_DIR / "hub"))
    model.eval()
    try:
        model.to(device)
    except Exception:  # noqa: BLE001 - MPS quirks must not kill the run
        LOG.warning("could not move speech model to %s; using CPU", device)
        device = "cpu"

    rate = int(getattr(model.config, "sampling_rate", 16000))
    out: list[np.ndarray] = []
    with torch.no_grad():
        for line in lines:
            inputs = tokenizer(line, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            waveform = model(**inputs).waveform
            audio = waveform[0].detach().cpu().numpy().astype(np.float32)
            # Guard against a rare VITS failure mode: NaN/inf samples.
            if not np.isfinite(audio).all():
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            out.append(audio)
    return out, rate


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #
def _midi_timing(midi_path: Path | str) -> tuple[float, float]:
    """Return ``(bar_seconds, total_seconds)`` for a MIDI file.

    Defensive on tempo: a generated MIDI can carry a 0 BPM event (the
    tokenizer now sanitises on save, but files predating that fix, or
    hand-edited ones, still circulate). Any tempo outside 20-400 BPM is
    replaced by the fallback.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    _, tempos = pm.get_tempo_changes()
    valid = [float(t) for t in np.atleast_1d(tempos) if 20.0 <= float(t) <= 400.0]
    bpm = valid[0] if valid else _FALLBACK_BPM
    # pretty_midi meters are almost always 4/4 in MidiCaps; read the first
    # time signature when one exists so waltzes still land on bar lines.
    beats_per_bar = 4.0
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        beats_per_bar = 4.0 * ts.numerator / ts.denominator
    bar = beats_per_bar * 60.0 / bpm
    return bar, float(pm.get_end_time())


def plan_placements(
    n_lines: int,
    *,
    bar_seconds: float,
    total_seconds: float,
    line_seconds: list[float] | None = None,
    stride_bars: int | None = None,
) -> list[float]:
    """Decide when each lyric line starts (seconds from song start).

    Lines are spread over the song and snapped to bar boundaries - the
    ear reads that as "singing between phrases". If a line runs past the
    next planned start it is pushed forward rather than overlapping
    (unless it is the last line, where running to the end is fine).
    """
    if n_lines <= 0:
        return []
    if stride_bars is None or stride_bars < 1:
        n_bars = max(1, int(total_seconds / max(bar_seconds, 1e-6)))
        stride_bars = max(1, n_bars // n_lines)

    starts: list[float] = []
    prev_end = 0.0
    for i in range(n_lines):
        bar_i = i * stride_bars
        t = min(bar_i * bar_seconds, max(0.0, total_seconds - 1.0))
        if starts and line_seconds is not None:
            t = max(t, prev_end + _GAP_SEC)
        starts.append(t)
        dur = line_seconds[i] if line_seconds and i < len(line_seconds) else 0.0
        prev_end = t + dur
    return starts


# --------------------------------------------------------------------------- #
# Mixing
# --------------------------------------------------------------------------- #
def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Polyphase resample (scipy); no-op when rates already match."""
    if src_rate == dst_rate:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(src_rate), int(dst_rate))
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)


def mix_vocals(
    instrumental: np.ndarray,
    vocal_track: np.ndarray,
    *,
    vocal_gain_db: float = -4.0,
) -> np.ndarray:
    """Add a mono vocal track to (N, C) instrumental audio."""
    gain = float(10.0 ** (vocal_gain_db / 20.0))
    mixed = instrumental.astype(np.float32, copy=True)
    if mixed.ndim == 1:
        mixed = mixed[:, None]
    vocal = vocal_track.astype(np.float32)
    if vocal.ndim > 1:
        vocal = vocal.reshape(-1)
    n = min(len(mixed), len(vocal))
    mixed[:n] += vocal[:n, None] * gain
    return mixed


def _fade_edges(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Short fade in/out so placed lines never click."""
    n = min(n_samples, len(audio) // 2)
    if n <= 0:
        return audio
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
def _read_lyrics(lyrics: str | None, lyrics_file: str | Path | None) -> list[str]:
    """Normalise lyrics input to a list of non-empty lines."""
    text = lyrics or ""
    if lyrics_file:
        text = Path(lyrics_file).read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines


def add_vocals(
    midi_path: str | Path,
    wav_path: str | Path,
    *,
    lyrics: str | None = None,
    lyrics_file: str | Path | None = None,
    out_path: str | Path | None = None,
    voice: str = DEFAULT_VOICE,
    device: str = "cpu",
    stride_bars: int | None = None,
    vocal_gain_db: float = -4.0,
    sample_rate: int = 44100,
) -> VocalResult:
    """Speak the lyrics, align them to the MIDI's bars, mix under the music.

    Args:
        midi_path: the generated MIDI (used only for tempo/bar timing).
        wav_path: the rendered instrumental to sing over.
        lyrics / lyrics_file: sung text; ``lyrics_file`` wins when given.
        voice: any ``transformers`` VITS TTS repo id.
        stride_bars: bars between line starts (default: spread evenly).
        vocal_gain_db: vocal level relative to the instrumental.

    Returns:
        A :class:`VocalResult`; the mixed WAV is written next to the
        instrumental as ``<stem>_vocal.wav`` unless ``out_path`` says
        otherwise.
    """
    import soundfile as sf

    lines = _read_lyrics(lyrics, lyrics_file)
    if not lines:
        raise ValueError("no lyrics given - pass lyrics= or lyrics_file=")

    midi_path, wav_path = Path(midi_path), Path(wav_path)
    if out_path is None:
        out_path = wav_path.with_name(f"{wav_path.stem}_vocal.wav")
    out_path = Path(out_path)

    bar_seconds, midi_seconds = _midi_timing(midi_path)
    instrumental, inst_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    total = len(instrumental) / inst_rate
    warnings: list[str] = []
    if midi_seconds <= 0:
        warnings.append("MIDI reported zero duration; timing may be wrong")

    LOG.info("speaking %d lyric line(s) with %s", len(lines), voice)
    segments, voice_rate = synthesize_lines(lines, voice=voice, device=device)

    # Model rate -> instrumental rate, with per-line length in seconds.
    resampled = [_resample(s, voice_rate, sample_rate) for s in segments]
    line_seconds = [len(s) / sample_rate for s in resampled]

    starts = plan_placements(
        len(lines),
        bar_seconds=bar_seconds,
        total_seconds=total,
        line_seconds=line_seconds,
        stride_bars=stride_bars,
    )

    vocal_track = np.zeros(len(instrumental), dtype=np.float32)
    fade = int(0.01 * sample_rate)
    for start, segment in zip(starts, resampled):
        i0 = int(start * sample_rate)
        if i0 >= len(vocal_track):
            warnings.append(f"line at {start:.1f}s fell past the end; skipped")
            continue
        seg = _fade_edges(segment.copy(), fade)
        n = min(len(seg), len(vocal_track) - i0)
        vocal_track[i0 : i0 + n] += seg[:n]

    mixed = mix_vocals(instrumental, vocal_track, vocal_gain_db=vocal_gain_db)
    # The instrumental was already peak-normalised by the renderer; keep
    # the mix legal by scaling only if WE pushed it over full scale.
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mixed, inst_rate)
    LOG.info("wrote vocal mix -> %s", out_path)

    return VocalResult(
        wav_path=str(out_path),
        n_lines=len(lines),
        duration_sec=len(mixed) / inst_rate,
        voice=voice,
        backend="tts+vocal-layer",
        placements=[round(s, 2) for s in starts],
        instrumental_path=str(wav_path),
        warnings=warnings,
    )
