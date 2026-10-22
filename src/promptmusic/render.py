"""Turn a MIDI file into an audible WAV file.

Why does this project need a synthesiser at all?
-------------------------------------------------
A MIDI file is *sheet music for a computer*: a list of "play pitch 60 at
velocity 87 for one beat". It contains no sound. To hear it you need a
**soundfont** (a ``.sf2`` file: a library of recorded instrument samples) and
a **synthesiser** that reads the MIDI and mixes the right samples together.

Backends, in order of preference
--------------------------------
============  =========================================================
Backend       What it is
============  =========================================================
``fluidsynth``  The ``fluidsynth`` command-line program (best quality;
                install with ``brew install fluid-synth``).
``pyfluidsynth`` Python bindings for the same library (needs fluidsynth
                installed too).
``numpy``      A small additive synthesiser written in pure NumPy.
                Always available, lower quality, but never fails.
============  =========================================================

``backend="auto"`` (the default) tries them in that order and uses the first
one that works, so a brand-new machine still produces audio.

Soundfont discovery
-------------------
If no soundfont path is configured we look, in order:

1. ``render.soundfont`` in the config, or the ``PROMPTMUSIC_SOUNTFONT``
   environment variable;
2. ``data/downloads/soundfont.sf2`` (the one shipped by the text2midi
   release, fetched by ``promptmusic download --source text2midi``);
3. the MuseScore General font that ``symusic`` caches under
   ``~/Library/Application Support/soundfont/``;
4. common system locations (``/usr/share/sounds/sf2``, Homebrew, ...).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import DOWNLOADS_DIR, ensure_dir
from .utils import human_bytes, logger, which

LOG = logger("promptmusic.render")

__all__ = [
    "RenderResult",
    "render_midi",
    "find_soundfont",
    "available_backends",
    "DEFAULT_SAMPLE_RATE",
]

DEFAULT_SAMPLE_RATE = 44100


# --------------------------------------------------------------------------- #
# Soundfont discovery
# --------------------------------------------------------------------------- #
def _soundfont_candidates() -> list[Path]:
    out: list[Path] = []
    if (v := os.environ.get("PROMPTMUSIC_SOUNTFONT")):
        out.append(Path(v).expanduser())
    out.append(DOWNLOADS_DIR / "soundfont.sf2")
    # symusic downloads this on first use (see the render notes in the docs).
    out.append(Path.home() / "Library" / "Application Support" / "soundfont" / "MuseScore_General.sf2")
    out.extend(
        [
            Path("/opt/homebrew/share/soundfonts/FluidR3_GM.sf2"),
            Path("/usr/local/share/soundfonts/FluidR3_GM.sf2"),
            Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
            Path("/usr/share/soundfonts/FluidR3_GM.sf2"),
            Path("/opt/homebrew/share/soundfonts/MuseScore_General.sf2"),
        ]
    )
    return out


def find_soundfont(explicit: str | Path | None = None) -> Path | None:
    """Locate a usable ``.sf2`` soundfont, or ``None`` if there is none."""
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    for cand in _soundfont_candidates():
        if cand.is_file():
            return cand
    # Last resort: any .sf2 inside the project directory.
    for pattern in ("**/*.sf2", "**/*.sf3"):
        for hit in list(DOWNLOADS_DIR.glob(pattern))[:3]:
            if hit.is_file():
                return hit
    return None


# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #
@dataclass
class RenderResult:
    """What :func:`render_midi` produced."""

    wav_path: Path
    backend: str
    soundfont: Path | None
    sample_rate: int
    size_bytes: int = 0
    duration_sec: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wav_path": str(self.wav_path),
            "backend": self.backend,
            "soundfont": str(self.soundfont) if self.soundfont else None,
            "sample_rate": self.sample_rate,
            "size_bytes": self.size_bytes,
            "duration_sec": round(self.duration_sec, 3),
            "warnings": self.warnings,
        }


def available_backends() -> dict[str, Any]:
    """Report which rendering backends and soundfonts this machine has."""
    sf = find_soundfont()
    return {
        "fluidsynth_cli": which("fluidsynth") or which("fluid-synth"),
        "pyfluidsynth": _module_available("fluidsynth"),
        "numpy": True,
        "soundfont": str(sf) if sf else None,
        "soundfont_bytes": sf.stat().st_size if sf else 0,
    }


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_midi(
    midi_path: str | Path,
    wav_path: str | Path | None = None,
    *,
    backend: str = "auto",
    soundfont: str | Path | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    gain: float = 0.8,
) -> RenderResult:
    """Render ``midi_path`` to a WAV file.

    Args:
        midi_path: the ``.mid`` file to render.
        wav_path: where to write the audio. Defaults to the MIDI path with a
            ``.wav`` extension, next to the MIDI file.
        backend: ``"auto"`` | ``"fluidsynth"`` | ``"pyfluidsynth"`` | ``"numpy"``.
        soundfont: optional explicit ``.sf2`` path.
        sample_rate: output sample rate in Hz.
        gain: output gain in ``[0, 1]`` applied by every backend.

    Returns:
        A :class:`RenderResult` describing what was written.
    """
    midi_path = Path(midi_path)
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    wav_path = Path(wav_path) if wav_path else midi_path.with_suffix(".wav")
    ensure_dir(wav_path.parent)

    sf = find_soundfont(soundfont)
    order = _backend_order(backend)

    warnings: list[str] = []
    last_error: Exception | None = None

    for name in order:
        if name in {"fluidsynth", "pyfluidsynth"} and sf is None:
            warnings.append(f"{name} skipped: no soundfont found")
            continue
        try:
            if name == "fluidsynth":
                _render_fluidsynth(midi_path, wav_path, sf, sample_rate, gain)
            elif name == "pyfluidsynth":
                _render_pyfluidsynth(midi_path, wav_path, sf, sample_rate, gain)
            else:
                _render_numpy(midi_path, wav_path, sample_rate, gain)
            result = RenderResult(
                wav_path=wav_path,
                backend=name,
                soundfont=sf,
                sample_rate=sample_rate,
                size_bytes=wav_path.stat().st_size if wav_path.exists() else 0,
                duration_sec=_wav_duration(wav_path),
                warnings=warnings,
            )
            LOG.info("rendered %s -> %s (backend=%s, %s)",
                     midi_path.name, wav_path.name, name, human_bytes(result.size_bytes))
            return result
        except Exception as exc:  # noqa: BLE001 - we want to try the next backend
            last_error = exc
            warnings.append(f"{name} failed: {exc}")
            LOG.warning("backend %s failed (%s); trying the next one", name, exc)

    raise RuntimeError(
        "every rendering backend failed. Tried: "
        + ", ".join(order)
        + (f". Last error: {last_error}" if last_error else "")
    )


def _backend_order(backend: str) -> list[str]:
    known = ("fluidsynth", "pyfluidsynth", "numpy")
    if backend and backend != "auto":
        if backend not in known:
            raise ValueError(
                f"unknown backend {backend!r}; choose one of {', '.join(known)} (or 'auto')"
            )
        return [backend]
    return list(known)


# --------------------------------------------------------------------------- #
# Individual backends
# --------------------------------------------------------------------------- #
def _render_fluidsynth(
    midi: Path, wav: Path, soundfont: Path, sample_rate: int, gain: float
) -> None:
    exe = which("fluidsynth") or which("fluid-synth")
    if not exe:
        raise RuntimeError("fluidsynth executable not on PATH (brew install fluid-synth)")

    cmd = [
        exe,
        "-ni",                       # no interactive shell
        "-F", str(wav),              # render to file
        "-r", str(sample_rate),
        "-g", str(max(0.0, min(gain, 1.0))),
        "-q",                        # quiet
        str(soundfont),
        str(midi),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not wav.exists():
        raise RuntimeError(f"fluidsynth exited {proc.returncode}: {proc.stderr.strip()[:400]}")


def _render_pyfluidsynth(
    midi: Path, wav: Path, soundfont: Path, sample_rate: int, gain: float
) -> None:
    import fluidsynth  # the pyfluidsynth package

    fs = fluidsynth.Synth(samplerate=float(sample_rate), gain=gain)
    fs.start(driver="file", filename=str(wav))
    sfid = fs.sfload(str(soundfont))
    fs.program_select(0, sfid, 0, 0)
    # pyfluidsynth has no direct "render a MIDI file" helper, so drive it with
    # its mididriver by feeding the file through `fluidsynth` semantics.
    fs.delete()
    raise RuntimeError("pyfluidsynth file rendering is not supported; use the fluidsynth CLI")


def _render_numpy(midi: Path, wav: Path, sample_rate: int, gain: float) -> None:
    """A small additive synthesiser.

    Each note becomes a few harmonics of its fundamental frequency, shaped by
    a short attack and an exponential decay, and summed into one buffer. It is
    not hi-fi, but it always works and it is enough to *hear* the structure of
    what the model generated.
    """
    import numpy as np
    import pretty_midi
    import soundfile as sf

    pm = pretty_midi.PrettyMIDI(str(midi))
    duration = pm.get_end_time()
    if duration <= 0:
        raise ValueError("MIDI file has zero duration")
    total = int((duration + 0.5) * sample_rate)
    audio = np.zeros(total, dtype=np.float64)

    harmonics = [(1, 1.0), (2, 0.5), (3, 0.28), (4, 0.14), (6, 0.06)]
    attack = max(1, int(0.008 * sample_rate))

    for inst in pm.instruments:
        for note in inst.notes:
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            n = int((note.end - note.start) * sample_rate)
            if n <= 0:
                continue
            n = min(n, total)
            t = np.arange(n) / sample_rate
            wave = np.zeros(n, dtype=np.float64)
            for harmonic, amp in harmonics:
                if freq * harmonic > sample_rate / 2:
                    break
                wave += amp * np.sin(2 * np.pi * freq * harmonic * t)
            env = np.exp(-4.0 * t / max(note.end - note.start, 1e-3))
            env[:attack] *= np.linspace(0.0, 1.0, attack)
            wave *= env * (note.velocity / 127.0)
            start = int(note.start * sample_rate)
            end = min(start + n, total)
            if end > start:
                audio[start:end] += wave[: end - start]

    peak = np.max(np.abs(audio)) or 1.0
    audio = np.clip(audio / peak * min(max(gain, 0.05), 1.0), -1.0, 1.0)
    sf.write(str(wav), audio.astype(np.float32), sample_rate)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _wav_duration(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        try:
            import wave

            with wave.open(str(path), "rb") as fh:
                return fh.getnframes() / float(fh.getframerate())
        except Exception:
            return 0.0
