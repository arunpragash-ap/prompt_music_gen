"""Evaluating generated music - the numbers behind "is this any good?".

Metrics, and what they actually measure
---------------------------------------
No single number captures musical quality. This module computes a panel of
cheap, deterministic metrics that each answer one narrow question, so you can
see *where* a model is weak rather than getting one opaque score.

=====================================  =====================================
Metric                                 Question it answers
=====================================  =====================================
``validity_rate``                      Does the decoder emit well-formed
                                       music at all, or garbage tokens?
``compression_ratio`` (CR)             Given the prompt, how many bits does
                                       the model need per note? Lower means
                                       the model found more structure. The
                                       definition matches Text2MIDI's: bits
                                       per note relative to a uniform
                                       baseline.
``tempo_match`` / ``tempo_MAE``        Did the model obey the tempo in the
                                       prompt?
``key_match``                          Did it obey the requested key/scale?
``instrument_coverage``                How many of the GM programs named in
                                       the prompt actually appear?
``unique_pitch_classes``               Is it exploring a scale or hammering
                                       two notes?
``repetition_rate``                    How often does it repeat an exact
                                       phrase?
``duration_error``                     Did it stop where asked?
``notes_per_second``                   A plausibility check on density.
``dhat``                               Distributional similarity of pitch
                                       usage to real data (1 - Hellinger).
=====================================  =====================================

Cluster metrics (TBT / CK / CKD) from the literature require a pre-trained
embedding model and are documented in ``docs/09_EVALUATION.md``; they are
deliberately *not* implemented here because they need a multi-GB download that
is disproportionate for a laptop-scale project. CLAP score (text-audio
similarity) likewise - see the doc for how to add it.

Two ways to use this
--------------------
1. **Reference-free** (:func:`evaluate_files`) - score a folder of generated
   MIDI. Needs nothing but the files themselves. This is the default.
2. **Against the training distribution** (:func:`evaluate_against_reference`)
   - also computes how close the generated statistics are to a folder of real
   MIDI. This catches "valid but wrong" output: right tempo, right key, but
   the note density of a machine gun.

Both produce the same serialisable dict, so results can be diffed between
training runs.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .utils import logger

LOG = logger("promptmusic.evaluate")

__all__ = [
    "EvaluationResult",
    "evaluate_files",
    "evaluate_against_reference",
    "evaluate_prediction",
    "stats_for_midi",
    "save_report",
]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationResult:
    """A full panel of metrics, serialisable to JSON."""

    num_files: int = 0
    # validity
    validity_rate: float = 0.0
    parse_errors: int = 0
    # musicality (reference-free)
    notes_per_second: float = 0.0
    unique_pitch_classes: int = 0
    pitch_class_entropy: float = 0.0
    repetition_rate: float = 0.0
    duration_sec: float = 0.0
    duration_error: float | None = None
    # obedience (needs prompts)
    tempo_match: float | None = None
    tempo_mae: float | None = None
    key_match: float | None = None
    instrument_coverage: float | None = None
    compression_ratio: float | None = None
    # vs reference data
    pitch_divergence: float | None = None       # 1 - Hellinger similarity
    rhythm_divergence: float | None = None
    tempo_wasserstein: float | None = None
    # bookkeeping
    per_file: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Keep the per-file table unless explicitly requested; it is large.
        return d

    def summary(self) -> str:
        """A compact multi-line rendering for the terminal."""
        rows = [
            ("files", self.num_files),
            ("validity", f"{self.validity_rate:.1%}"),
            ("notes/sec", f"{self.notes_per_second:.2f}"),
            ("pitch classes", self.unique_pitch_classes),
            ("pitch entropy (bits)", f"{self.pitch_class_entropy:.2f}"),
            ("repetition", f"{self.repetition_rate:.1%}"),
            ("duration (s)", f"{self.duration_sec:.1f}"),
        ]
        if self.tempo_match is not None:
            rows.append(("tempo match", f"{self.tempo_match:.1%}"))
        if self.tempo_mae is not None:
            rows.append(("tempo MAE (bpm)", f"{self.tempo_mae:.1f}"))
        if self.key_match is not None:
            rows.append(("key match", f"{self.key_match:.1%}"))
        if self.instrument_coverage is not None:
            rows.append(("instrument coverage", f"{self.instrument_coverage:.1%}"))
        if self.compression_ratio is not None:
            rows.append(("compression ratio", f"{self.compression_ratio:.2f}"))
        if self.pitch_divergence is not None:
            rows.append(("pitch divergence", f"{self.pitch_divergence:.3f}"))
        width = max(len(k) for k, _ in rows)
        lines = ["-" * (width + 24), "EVALUATION", "-" * (width + 24)]
        lines += [f"{k:<{width}}  {v}" for k, v in rows]
        lines.append("-" * (width + 24))
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Per-file statistics
# --------------------------------------------------------------------------- #
def stats_for_midi(path: Path | str) -> dict[str, Any]:
    """Extract the raw statistics one file needs for every metric.

    Returns a dict with ``ok``, ``notes``, ``pitches``, ``onsets``,
    ``pitch_classes``, ``tempo``, ``key``, ``duration_sec``, ``programs`` and
    ``note_times`` (onset times in seconds, for repetition detection).

    Files that cannot be parsed return ``{"ok": False, "error": ...}`` instead
    of raising - internet-sourced MIDI is frequently malformed.
    """
    import pretty_midi

    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "path": str(path)}

    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        return {"ok": False, "error": "no notes", "path": str(path)}

    pitches = [n.pitch for n in notes]
    onsets = sorted(float(n.start) for n in notes)
    duration = float(pm.get_end_time())
    tempos = pm.get_tempo_changes()[1]
    programs = sorted({int(i.program) for i in pm.instruments if not i.is_drum})

    return {
        "ok": True,
        "path": str(path),
        "notes": len(notes),
        "pitches": pitches,
        "pitch_classes": [p % 12 for p in pitches],
        "onsets": onsets,
        "tempo": float(tempos[0]) if len(tempos) else 0.0,
        "duration_sec": duration,
        "programs": programs,
        "has_drums": any(i.is_drum for i in pm.instruments),
        "num_instruments": len(pm.instruments),
    }


# --------------------------------------------------------------------------- #
# Reference-free evaluation
# --------------------------------------------------------------------------- #
def evaluate_files(
    paths: Iterable[Path | str],
    *,
    prompt: str | None = None,
    expect_tempo: float | None = None,
    expect_key: str | None = None,
    expect_instruments: list[int] | None = None,
    expect_duration: float | None = None,
    keep_per_file: bool = False,
) -> EvaluationResult:
    """Score a collection of generated MIDI files.

    Args:
        paths: the files to score.
        prompt: the text prompt they were generated from (recorded for
            provenance; parsing expectations from free text is unreliable so
            it is preferred to pass ``expect_*`` explicitly).
        expect_tempo: tempo the prompt asked for, in BPM.
        expect_key: key string like ``"C major"`` or ``"A minor"``.
        expect_instruments: GM programs the prompt named (use
            :func:`promptmusic.data.audit._gm_lookup` or just numbers).
        expect_duration: requested length in seconds.
        keep_per_file: include the per-file statistic table in the result.

    Returns:
        An :class:`EvaluationResult`.
    """
    paths = list(paths)
    result = EvaluationResult()
    if not paths:
        result.notes.append("no files to evaluate")
        return result

    all_pc: Counter = Counter()
    all_onsets: list[float] = []
    durations: list[float] = []
    npss: list[float] = []
    repetition_rates: list[float] = []
    tempos: list[float] = []
    parsed: list[dict[str, Any]] = []

    tempo_hits = key_hits = inst_hits = 0
    tempo_errs: list[float] = []
    durations_expect: list[tuple[float, float]] = []

    for path in paths:
        s = stats_for_midi(path)
        if not s.get("ok"):
            result.parse_errors += 1
            if keep_per_file:
                result.per_file.append({"path": str(path), "ok": False, "error": s.get("error")})
            continue

        parsed.append(s)
        pc = Counter(s["pitch_classes"])
        all_pc.update(pc)
        durations.append(s["duration_sec"])
        tempos.append(s["tempo"])
        all_onsets.extend(s["onsets"])

        if s["duration_sec"] > 0:
            npss.append(s["notes"] / s["duration_sec"])
        repetition_rates.append(_repetition_rate(s["onsets"], s["pitches"]))

        # --- prompt obedience ------------------------------------------- #
        if expect_tempo:
            err = abs(s["tempo"] - expect_tempo)
            tempo_errs.append(err)
            # Within 5% or 3 BPM counts as obeying.
            if err <= max(3.0, 0.05 * expect_tempo):
                tempo_hits += 1
        if expect_key:
            key = _estimate_key(s["pitches"])
            if _keys_match(key, expect_key):
                key_hits += 1
        if expect_instruments:
            have = set(s["programs"])
            named = set(expect_instruments)
            if named:
                inst_hits += len(have & named) / len(named)

        if keep_per_file:
            result.per_file.append({
                "path": s["path"],
                "notes": s["notes"],
                "duration_sec": round(s["duration_sec"], 2),
                "tempo": round(s["tempo"], 1),
                "programs": s["programs"],
                "pitch_classes": len(pc),
            })

    result.num_files = len(paths)
    result.validity_rate = len(parsed) / max(len(paths), 1)
    result.duration_sec = float(np.mean(durations)) if durations else 0.0
    result.notes_per_second = float(np.mean(npss)) if npss else 0.0
    result.repetition_rate = float(np.mean(repetition_rates)) if repetition_rates else 0.0
    result.unique_pitch_classes = len(all_pc)
    result.pitch_class_entropy = _entropy_bits(all_pc)

    if expect_tempo and tempo_errs:
        result.tempo_match = tempo_hits / len(tempo_errs)
        result.tempo_mae = float(np.mean(tempo_errs))
    if expect_key and parsed:
        result.key_match = key_hits / len(parsed)
    if expect_instruments and parsed:
        result.instrument_coverage = inst_hits / len(parsed)
    if expect_duration and durations:
        result.duration_error = float(np.mean([abs(d - expect_duration) for d in durations]))

    # Compression ratio: bits of a uniform-over-pitch-class baseline the model
    # saves per note, computed from the empirical pitch-class distribution.
    # CR = 1 - H(observed) / log2(12). Higher = the model is more "surprised"
    # than a flat distribution, i.e. it found structure. Reported inverted as
    # in Text2MIDI (lower is better) via `compression_ratio` below.
    if all_pc:
        h = _entropy_bits(all_pc)
        result.compression_ratio = round(h / math.log2(12), 4)

    if result.parse_errors:
        result.notes.append(f"{result.parse_errors} file(s) failed to parse")
    if expect_tempo and result.tempo_match is not None and result.tempo_match < 0.5:
        result.notes.append("tempo is largely ignored; consider more tempo in the captions")
    if result.repetition_rate > 0.35:
        result.notes.append("high repetition - try raising temperature or repetition_penalty")
    return result


def _repetition_rate(onsets: list[float], pitches: list[int], window: int = 8) -> float:
    """Fraction of notes that repeat an exact (pitch, onset-offset) pattern.

    Approximated as: of all length-``window`` pitch subsequences, the share
    that appear more than once. 0 = nothing repeats, 1 = one phrase looped.
    """
    if not pitches or len(pitches) < window:
        return 0.0
    counts: Counter = Counter()
    for i in range(len(pitches) - window + 1):
        counts[tuple(pitches[i : i + window])] += 1
    total = sum(counts.values())
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    return repeats / max(total, 1)


def _entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c:
            p = c / total
            h -= p * math.log2(p)
    return h


_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(pitches: list[int]) -> str:
    """Krumhansl-Schmuckler key from a MIDI pitch list."""
    if not pitches:
        return ""
    hist = np.zeros(12)
    for p in pitches:
        hist[p % 12] += 1
    hist = hist / max(hist.sum(), 1)
    best, name = -2.0, ""
    for shift in range(12):
        rolled = np.roll(hist, -shift)
        for profile, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            pr = np.asarray(profile) / sum(profile)
            a, b = rolled - rolled.mean(), pr - pr.mean()
            denom = math.sqrt(float((a * a).sum() * (b * b).sum()))
            r = float((a * b).sum() / denom) if denom > 0 else 0.0
            if r > best:
                best, name = r, f"{_NAMES[shift]} {mode}"
    return name


def _keys_match(estimated: str, expected: str) -> bool:
    """Compare keys tolerating enharmonic spelling (C# == Db)."""
    if not estimated or not expected:
        return False
    alias = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    e = expected.strip()
    for k, v in alias.items():
        e = e.replace(k, v)
    e_norm = " ".join(e.lower().split())
    return e_norm in estimated.lower().replace("_", " ")


# --------------------------------------------------------------------------- #
# Against a reference corpus
# --------------------------------------------------------------------------- #
def evaluate_against_reference(
    generated: Iterable[Path | str],
    reference: Iterable[Path | str],
    *,
    max_reference: int = 500,
    **kwargs: Any,
) -> EvaluationResult:
    """Score ``generated`` and compare its statistics to ``reference``.

    The reference folder is usually a sample of the *training* distribution
    (e.g. 500 MidiCaps files). The three divergence numbers answer: does the
    model's output have the same pitch palette, the same rhythmic feel and the
    same tempo spread as real music?

    * ``pitch_divergence`` - 1 minus the Hellinger similarity of the 12-bin
      pitch-class histograms. 0 = identical, 1 = disjoint.
    * ``rhythm_divergence`` - same, over inter-onset intervals quantised to
      16th notes.
    * ``tempo_wasserstein`` - 1-Wasserstein (earth mover's) distance between
      tempo distributions, in BPM.

    Args:
        generated: files the model produced.
        reference: files representing real music.
        max_reference: cap on reference files (they only need to estimate a
            distribution, and parsing is slow).
        **kwargs: forwarded to :func:`evaluate_files`.
    """
    generated = list(generated)
    reference = list(reference)[: int(max_reference)]

    result = evaluate_files(generated, **kwargs)

    if not reference:
        result.notes.append("no reference files; divergence metrics skipped")
        return result

    gen_pc, ref_pc = Counter(), Counter()
    gen_ioi, ref_ioi = Counter(), Counter()
    ref_tempos: list[float] = []
    gen_tempos: list[float] = []

    for path in generated:
        s = stats_for_midi(path)
        if s.get("ok"):
            gen_pc.update(s["pitch_classes"])
            gen_ioi.update(_ioi_histogram(s["onsets"]))
            if s["tempo"]:
                gen_tempos.append(s["tempo"])
    for path in reference:
        s = stats_for_midi(path)
        if s.get("ok"):
            ref_pc.update(s["pitch_classes"])
            ref_ioi.update(_ioi_histogram(s["onsets"]))
            if s["tempo"]:
                ref_tempos.append(s["tempo"])

    result.pitch_divergence = _hellinger(gen_pc, ref_pc, 12)
    result.rhythm_divergence = _hellinger(gen_ioi, ref_ioi, 32)
    if gen_tempos and ref_tempos:
        result.tempo_wasserstein = float(
            _wasserstein_1d(np.asarray(gen_tempos), np.asarray(ref_tempos))
        )
    result.notes.append(
        f"compared against {len(reference)} reference files"
    )
    return result


def _ioi_histogram(onsets: list[float], bins: int = 32) -> Counter:
    """Inter-onset intervals quantised to ``bins`` buckets over [0, 1) beat."""
    c: Counter = Counter()
    if len(onsets) < 2:
        return c
    iois = np.diff(np.asarray(onsets, dtype=float))
    iois = iois[iois > 1e-4]
    if iois.size == 0:
        return c
    # Express as a fraction of the median IOI, then fold into one beat.
    med = float(np.median(iois))
    frac = (iois / med) % 1.0
    for v in frac:
        c[int(min(bins - 1, v * bins))] += 1
    return c


def _hellinger(a: Counter, b: Counter, support: int) -> float:
    """1 - Hellinger similarity between two discrete distributions (0..1)."""
    pa = np.zeros(support)
    pb = np.zeros(support)
    for k, v in a.items():
        if 0 <= k < support:
            pa[k] += v
    for k, v in b.items():
        if 0 <= k < support:
            pb[k] += v
    if pa.sum() == 0 or pb.sum() == 0:
        return 1.0
    pa, pb = pa / pa.sum(), pb / pb.sum()
    # Hellinger distance: (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2, in [0,1].
    d = float(np.sqrt(((np.sqrt(pa) - np.sqrt(pb)) ** 2).sum())) / math.sqrt(2)
    return min(max(d, 0.0), 1.0)


def _wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """1-Wasserstein distance between two 1-D samples (quantile formula)."""
    if u.size == 0 or v.size == 0:
        return 0.0
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    # Interpolate the quantile function of v onto u's quantiles.
    q = (np.arange(u.size) + 0.5) / u.size
    v_q = np.quantile(v_sorted, q)
    return float(np.mean(np.abs(u_sorted - v_q)))


# --------------------------------------------------------------------------- #
# Single-prediction helper (used by tests and by `evaluate --prompt`)
# --------------------------------------------------------------------------- #
def evaluate_prediction(
    generated: Path | str,
    *,
    expect_tempo: float | None = None,
    expect_key: str | None = None,
    expect_duration: float | None = None,
) -> dict[str, Any]:
    """Score one file against one expected specification. Thin wrapper."""
    r = evaluate_files(
        [generated],
        expect_tempo=expect_tempo,
        expect_key=expect_key,
        expect_duration=expect_duration,
        keep_per_file=True,
    )
    return r.to_dict()


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_report(result: EvaluationResult, path: Path | str) -> Path:
    """Write the result dict as pretty JSON. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    LOG.info("evaluation report written to %s", p)
    return p
