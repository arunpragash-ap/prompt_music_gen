"""Mastering: turn a dry MIDI render into something that sounds produced.

Why this exists
---------------
A soundfont render is *flat*: every instrument is dry, the vocal sits on
top instead of inside the track, and nothing controls the dynamics. Even a
well-generated arrangement sounds like a MIDI file, not a record. This
module is a small, dependency-free mastering chain:

* high-pass the vocal and add presence,
* compress the vocal so every syllable is audible,
* gently widen the instrumental (mid/side),
* duck the instrumental under the vocal (a manual sidechain),
* one convolution reverb shared by both, so they sound like one room,
* bus compression and a soft limiter, normalised to -1 dBFS.

Everything is numpy + scipy; no plugins, no extra packages. The reverb is a
procedurally generated impulse response convolved with FFTs (fast even for
an 80 s stereo track).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOG = logging.getLogger("promptmusic.master")

__all__ = [
    "master_mix",
    "compress",
    "duck",
    "make_impulse_response",
    "convolve_reverb",
    "soft_limit",
    "highpass",
    "bandpass",
]


# --------------------------------------------------------------------------- #
# Small DSP helpers
# --------------------------------------------------------------------------- #
def _as_stereo(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x[:, :2]


def _match_length(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) >= n:
        return a[:n]
    return np.pad(a, ((0, n - len(a)), (0, 0))) if a.ndim == 2 else np.pad(a, (0, n - len(a)))


def highpass(x: np.ndarray, sr: int, cutoff: float, order: int = 2) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    if cutoff <= 0 or cutoff >= sr / 2:
        return x
    sos = butter(order, cutoff / (sr / 2), btype="highpass", output="sos")
    return sosfilt(sos, x, axis=0).astype(np.float32)


def bandpass(x: np.ndarray, sr: int, low: float, high: float, order: int = 2) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    low = max(20.0, low)
    high = min(sr / 2 - 100.0, high)
    if high <= low:
        return np.zeros_like(x)
    sos = butter(order, [low / (sr / 2), high / (sr / 2)], btype="bandpass", output="sos")
    return sosfilt(sos, x, axis=0).astype(np.float32)


def _envelope(x: np.ndarray, sr: int, attack: float, release: float, block: int = 64) -> np.ndarray:
    """Block-RMS envelope with attack/release smoothing, back at sample rate."""
    mono = np.abs(x).mean(axis=1) if x.ndim == 2 else np.abs(x)
    n_blocks = len(mono) // block
    if n_blocks == 0:
        return np.zeros(len(mono), dtype=np.float32)
    rms = np.sqrt(np.mean(mono[: n_blocks * block].reshape(n_blocks, block) ** 2, axis=1))
    a_att = float(np.exp(-block / max(1e-4, attack * sr)))
    a_rel = float(np.exp(-block / max(1e-4, release * sr)))
    env = np.empty(n_blocks, dtype=np.float64)
    e = 0.0
    for i in range(n_blocks):
        v = float(rms[i])
        a = a_att if v > e else a_rel
        e = a * e + (1.0 - a) * v
        env[i] = e
    # Linear interpolation of the block envelope back to audio samples.
    xs = (np.arange(n_blocks) + 0.5) * block
    full = np.interp(np.arange(len(mono)), xs, env)
    return full.astype(np.float32)


def compress(
    x: np.ndarray,
    sr: int,
    *,
    threshold_db: float = -20.0,
    ratio: float = 3.0,
    attack: float = 0.01,
    release: float = 0.12,
    makeup_db: float = 0.0,
) -> np.ndarray:
    """Feed-forward compressor with a block envelope (vectorised gain)."""
    if x.size == 0:
        return x
    env = np.maximum(_envelope(x, sr, attack, release), 1e-7)
    level_db = 20.0 * np.log10(env)
    over = np.maximum(0.0, level_db - threshold_db)
    gain_db = -over * (1.0 - 1.0 / max(ratio, 1.0)) + makeup_db
    gain = (10.0 ** (gain_db / 20.0)).astype(np.float32)
    if x.ndim == 2:
        gain = gain[:, None]
    return (x * gain).astype(np.float32)


def duck(
    target: np.ndarray,
    sidechain: np.ndarray,
    sr: int,
    *,
    depth: float = 0.35,
    attack: float = 0.02,
    release: float = 0.28,
) -> np.ndarray:
    """Turn ``target`` down while ``sidechain`` (the vocal) is active."""
    if target.size == 0 or sidechain.size == 0:
        return target
    n = len(target)
    sc = _match_length(np.asarray(sidechain, dtype=np.float32).reshape(-1), n)
    env = _envelope(sc, sr, attack, release)
    peak = float(env.max()) or 1.0
    # Smoothstep so the gain change has no corners.
    u = np.clip(env / peak, 0.0, 1.0)
    u = u * u * (3.0 - 2.0 * u)
    gain = (1.0 - depth * u).astype(np.float32)
    if target.ndim == 2:
        gain = gain[:, None]
    return (target * gain).astype(np.float32)


def make_impulse_response(
    sr: int,
    *,
    rt60: float = 1.7,
    length: float = 2.2,
    predelay: float = 0.018,
    seed: int = 7,
    damping_hz: float = 6500.0,
) -> np.ndarray:
    """A stereo hall impulse response: decaying, damped noise + early taps."""
    n = int(length * sr)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    env = np.exp(-6.9078 * t / max(rt60, 0.1))
    ir = rng.standard_normal((n, 2)).astype(np.float32) * env[:, None]
    # Early reflections give the reverb a sense of size instead of a wash.
    for delay_ms, gain in ((11, 0.5), (17, 0.38), (29, 0.27), (41, 0.2), (53, 0.14)):
        i = int(delay_ms / 1000.0 * sr)
        if i < n:
            ir[i, 0] += gain
            ir[i, 1] += gain * 0.83
    from scipy.signal import butter, sosfilt

    sos = butter(2, min(damping_hz, sr / 2 - 500) / (sr / 2), btype="lowpass", output="sos")
    ir = sosfilt(sos, ir, axis=0).astype(np.float32)
    pd = int(predelay * sr)
    if pd > 0:
        ir = np.pad(ir, ((pd, 0), (0, 0)))[:n]
    energy = np.sqrt(np.sum(ir**2, axis=0, keepdims=True)) + 1e-9
    return (ir / energy * 0.55).astype(np.float32)


def convolve_reverb(x: np.ndarray, ir: np.ndarray, *, wet: float = 1.0) -> np.ndarray:
    """FFT-convolve ``x`` (mono or stereo) with ``ir`` (stereo IR)."""
    from scipy.signal import fftconvolve

    x = _as_stereo(x)
    n = len(x)
    out = np.zeros_like(x)
    for c in range(2):
        conv = fftconvolve(x[:, c], ir[:, c], mode="full")
        out[:, c] = conv[:n]
    return (out * wet).astype(np.float32)


def soft_limit(x: np.ndarray, ceiling: float = 0.891) -> np.ndarray:
    """Peak limiter: reduce peaks to the ceiling, never boost quiet audio.

    ``tanh(k x)/k`` is solved for the ``k`` that maps the current peak to the
    ceiling. Unlike a normaliser this leaves low-level material untouched, so
    a sparse mix is not slammed into a wall of sound.
    """
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak <= ceiling or peak <= 1e-8:
        return x
    target = min(ceiling, 0.999)
    lo, hi = 1e-3, 50.0
    for _ in range(60):
        k = 0.5 * (lo + hi)
        if np.tanh(peak * k) / k > target:
            hi = k
        else:
            lo = k
    k = 0.5 * (lo + hi)
    y = np.tanh(x * k) / k
    y_peak = float(np.max(np.abs(y))) or 1.0
    return (y * (target / y_peak)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
def _active_rms(x: np.ndarray, floor_db: float = -45.0) -> float:
    """RMS of the parts that are actually sounding (not the silence)."""
    mono = np.abs(x).mean(axis=1) if x.ndim == 2 else np.abs(x)
    if mono.size == 0:
        return 0.0
    thresh = float(mono.max()) * 10 ** (floor_db / 20.0)
    active = mono[mono > max(thresh, 1e-6)]
    if active.size == 0:
        return float(np.sqrt(np.mean(mono**2)))
    return float(np.sqrt(np.mean(active**2)))


def _fit_peak(x: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (ceiling / peak) if peak > ceiling else x


def master_mix(
    instrumental: np.ndarray,
    vocals: np.ndarray,
    sr: int,
    *,
    vocal_gain_db: float = -1.0,
    reverb: float = 0.16,
    duck_depth: float = 0.35,
    width: float = 1.18,
    warnings: list[str] | None = None,
) -> np.ndarray:
    """Mix vocals under an instrumental and master the result.

    Args:
        instrumental: ``(N,)`` or ``(N, 2)`` backing track.
        vocals: ``(N,)`` or ``(N, 2)`` vocal track, same timeline.
        sr: sample rate.
        vocal_gain_db: vocal level relative to the instrumental.
        reverb: wet send amount (0 disables the reverb).
        duck_depth: how much the instrumental dips under the vocal (0-1).
        width: side-channel gain for the instrumental (>1 = wider).
        warnings: optional list that receives non-fatal notes.

    Returns:
        ``(N, 2)`` float32 audio, peaked at -1 dBFS.
    """
    warnings = warnings if warnings is not None else []
    inst = _as_stereo(instrumental)
    n = len(inst)
    voc = np.asarray(vocals, dtype=np.float32)
    if voc.ndim == 1:
        voc = voc[:, None]
    if voc.shape[1] == 1:
        voc = np.repeat(voc, 2, axis=1)
    voc = _match_length(voc, n)

    # --- vocal chain ------------------------------------------------------- #
    voc = highpass(voc, sr, 85.0)
    presence = bandpass(voc, sr, 2200.0, 7000.0)
    voc = voc + 0.22 * presence
    voc = compress(
        voc, sr, threshold_db=-20.0, ratio=3.2, attack=0.008, release=0.13, makeup_db=2.0
    )

    # A voice is a point source. Keep it centred; widening a single mono
    # track with a delay creates comb filtering in mono. The reverb provides
    # the width instead.
    voc = np.repeat(voc[:, :1], 2, axis=1)
    voc_l = voc[:, 0]

    # --- instrumental: width and a little air ------------------------------ #
    inst = highpass(inst, sr, 32.0)
    mid = 0.5 * (inst[:, 0] + inst[:, 1])
    side = 0.5 * (inst[:, 0] - inst[:, 1]) * width
    inst = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
    air = bandpass(inst, sr, 8000.0, min(16000.0, sr / 2 - 500.0))
    inst = inst + 0.08 * air
    # Level the backing track before balancing the voice against it. A
    # soundfont render of a sparse arrangement is quiet (RMS ~0.06) while raw
    # TTS is near full scale; mixing them raw puts the vocal ~20 dB too loud.
    inst_rms = _active_rms(inst)
    if inst_rms > 1e-5:
        inst = _fit_peak(inst * (0.09 / inst_rms), 0.70)

    # --- set the vocal level relative to the music ------------------------ #
    # vocal_gain_db is now a *ratio* between the two RMS levels, not an
    # absolute gain on a full-scale TTS waveform.
    voc_rms = _active_rms(voc)
    target_rms = _active_rms(inst) * 10.0 ** (vocal_gain_db / 20.0)
    if voc_rms > 1e-5 and target_rms > 1e-6:
        voc = _fit_peak(voc * (target_rms / voc_rms), 0.70)

    # --- duck the backing track under the voice --------------------------- #
    voc_l = voc[:, 0]
    inst = duck(inst, voc_l, sr, depth=duck_depth)

    # --- sit the voice in the same room ----------------------------------- #
    mix = inst + voc
    if reverb > 0:
        ir = make_impulse_response(sr)
        mix = mix + convolve_reverb(voc, ir, wet=reverb)
        mix = mix + convolve_reverb(inst, ir, wet=reverb * 0.45)

    # --- master bus -------------------------------------------------------- #
    mix = highpass(mix, sr, 28.0)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.95:
        mix *= 0.95 / peak  # leave headroom for the bus compressor
    mix = compress(
        mix, sr, threshold_db=-14.0, ratio=1.8, attack=0.025, release=0.25, makeup_db=1.0
    )
    mix = soft_limit(mix, ceiling=0.891)
    # Normalise the finished mix, but never by more than +3.5 dB: a master
    # that lifts a quiet, dynamic mix by 13 dB is just distortion.
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1e-6:
        mix *= min(0.891 / peak, 1.5)

    # Fade the very edges so a hard cut never clicks.
    fade = min(len(mix) // 2, max(1, int(0.01 * sr)))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        mix[:fade] *= ramp[:, None]
        mix[-fade:] *= ramp[::-1, None]

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.891:
        mix *= 0.891 / peak
    LOG.info(
        "mastered %d samples (%.1fs), peak %.2f, reverb %.0f%%",
        n, n / sr, peak, reverb * 100,
    )
    return mix.astype(np.float32)
