"""Melody-aware singing: put the *words* on the *notes* of the generated song.

What changed from the first vocal layer
---------------------------------------
``vocals.py`` speaks each lyric line and drops it on a bar boundary. The
words are audible, but they are *spoken over* the music: no note, no
rhythm, no relation to the melody. That is why it sounds like a narrator.

This module does the thing that makes it read as singing:

1. **Extract the lead melody** from the generated MIDI (skyline over the
   non-drum tracks, simplified to a singable density).
2. **Break the lyrics into syllables** (Tamil or Latin script).
3. **Assign every syllable to a note** in a phrase of that melody, so the
   syllable count and the phrase length agree by construction.
4. **Pitch-shape each syllable**: the spoken syllable is pitch-shifted to
   the note's fundamental, stretched to the note's duration, and given
   vibrato on long notes. This is the classic "pitch-shaped speech"
   approach to singing - not a neural singing voice, but the words now
   follow the tune instead of talking over it.

Honest ceiling (read this before judging the result)
----------------------------------------------------
This is **not** a singing-voice synthesis (SVS) model. An SVS network
takes phonemes + pitch + duration and renders a human timbre with correct
vowel formants; no open Tamil SVS model exists, and the DiffSinger
voicebanks that do exist are Chinese/Japanese/English. Pitch-shaped TTS
keeps the speaker's original formants and adds a musical F0, so it sounds
like a synthetic voice *singing*, not a human. It is a large step up from
speech-over-music and it is honest about what it is.

Dependencies: numpy + scipy only (no librosa, no numba). The TTS model is
the same one ``vocals.py`` uses (``facebook/mms-tts-tam``, CC-BY-NC).
"""

from __future__ import annotations

import logging
import math
import re
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

LOG = logging.getLogger("promptmusic.singing")

__all__ = [
    "DEFAULT_VOICE",
    "MelodyNote",
    "SungUnit",
    "SingResult",
    "split_syllables",
    "split_tamil_syllables",
    "split_english_syllables",
    "extract_melody",
    "extract_lead_track",
    "quantize_melody",
    "simplify_melody",
    "segment_phrases",
    "choose_phrases",
    "allocate_line_sections",
    "plan_line",
    "plan_line_grid",
    "smooth_pitches",
    "estimate_f0",
    "pitch_shift",
    "time_stretch",
    "vibrato",
    "fit_duration",
    "render_line",
    "sing_song",
]

#: Same Tamil VITS model as ``vocals.py``. CC-BY-NC 4.0 - see docs/05.
DEFAULT_VOICE = "facebook/mms-tts-tam"

#: Processing rate. The TTS model is 16 kHz; doing the pitch work there and
#: resampling once at the end is both faster and avoids resample artefacts.
PROCESS_RATE = 16000

_TAMIL_START, _TAMIL_END = 0x0B80, 0x0BFF
_VIRAMA = "\u0BCD"
_AYTHAM = "\u0B83"
#: Dependent vowel signs (ா .. ௌ).
_VOWEL_SIGNS = set("\u0BBE\u0BBF\u0Bc0\u0BC1\u0BC2\u0BC6\u0BC7\u0BC8\u0BCA\u0BCB\u0BCC\u0BD7")
#: Independent vowels (அ .. ஔ).
_INDEPENDENT_VOWELS = set(chr(c) for c in range(0x0B85, 0x0B95))
#: Tamil consonants (க .. ஹ, including the Grantha letters).
_CONSONANTS = set(chr(c) for c in range(0x0B95, 0x0BBA))
#: Long vowels / long vowel signs get roughly twice the weight.
_LONG_MARKS = set("\u0BBE\u0BC0\u0BC2\u0BC7\u0BCB\u0BCC\u0BD7\u0B86\u0B88\u0B8A\u0B8E\u0B8F\u0B90\u0B92\u0B93\u0B94")


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class MelodyNote:
    """One note of the extracted lead line, in seconds / MIDI pitch."""

    start: float
    end: float
    pitch: int
    velocity: int = 80

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SungUnit:
    """One syllable placed on the song timeline."""

    syllable: str
    start: float
    duration: float
    pitch: int
    n_notes: int = 1


@dataclass
class SingResult:
    """What :func:`sing_song` produced."""

    wav_path: str
    duration_sec: float
    n_lines: int
    n_syllables: int
    n_melody_notes: int
    n_phrases: int
    voice: str
    natural_f0_hz: float
    melody_range: tuple[int, int]
    placements: list[float] = field(default_factory=list)
    instrumental_path: str = ""
    midi_path: str = ""
    stems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wav_path": self.wav_path,
            "duration_sec": round(self.duration_sec, 2),
            "n_lines": self.n_lines,
            "n_syllables": self.n_syllables,
            "n_melody_notes": self.n_melody_notes,
            "n_phrases": self.n_phrases,
            "voice": self.voice,
            "natural_f0_hz": round(self.natural_f0_hz, 1),
            "melody_range": list(self.melody_range),
            "placements": self.placements,
            "instrumental_path": self.instrumental_path,
            "midi_path": self.midi_path,
            "stems": self.stems,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# 1. Syllabification
# --------------------------------------------------------------------------- #
def _is_tamil(ch: str) -> bool:
    return _TAMIL_START <= ord(ch) <= _TAMIL_END


def _tamil_graphemes(word: str) -> list[str]:
    """Split a Tamil word into base + combining-mark clusters."""
    out: list[str] = []
    for ch in word:
        if out and (ch in _VOWEL_SIGNS or ch == _VIRAMA or ch == _AYTHAM):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def split_tamil_syllables(text: str) -> list[str]:
    """Break Tamil text into singable syllables.

    Tamil is close to syllabic: a syllable is a consonant cluster plus a
    vowel (written as a dependent sign or inherent ``அ``), optionally with
    a coda consonant marked by the virama (``்``). The rules implemented
    here are the ones that matter for singing::

        காதல்      -> கா | தல்
        நதியில்    -> ந | தி | யில்
        என்        -> என்
        ஸ்திரம்    -> ஸ்தி | ரம்

    Latin words, digits and punctuation are passed through as units so a
    mixed line still produces something singable.
    """
    syllables: list[str] = []
    for token in re.findall(r"[\u0B80-\u0BFF]+|[A-Za-z']+|\d+|[^\s]", text):
        if not _is_tamil(token[0]):
            if token.strip() and not re.fullmatch(r"[^\w\u0B80-\u0BFF]+", token):
                syllables.append(token)
            continue

        cur = ""
        has_nucleus = False

        def flush() -> None:
            nonlocal cur, has_nucleus
            if cur:
                syllables.append(cur)
            cur, has_nucleus = "", False

        for cluster in _tamil_graphemes(token):
            base = cluster[0]
            if base in _INDEPENDENT_VOWELS:
                flush()
                cur, has_nucleus = cluster, True
            elif base in _CONSONANTS:
                if cluster.endswith(_VIRAMA) or cluster.endswith(_AYTHAM):
                    # Pure consonant: either the onset of the next syllable
                    # (ஸ் + தி) or the coda of this one (தல்). Appending
                    # covers both; the following nucleus decides.
                    cur += cluster
                else:
                    if has_nucleus:
                        flush()
                    cur += cluster
                    has_nucleus = True
            else:  # pragma: no cover - digits/odd marks
                cur += cluster
                if not cluster.endswith(_VIRAMA):
                    has_nucleus = True
        flush()
    return [s for s in syllables if s and not re.fullmatch(r"[\W_]+", s)]


_EN_VOWELS = set("aeiouy")


def split_english_syllables(text: str) -> list[str]:
    """A pragmatic English syllabifier (enough for singing, not phonology).

    Vowel groups are the nuclei. Consonants between two nuclei split by the
    usual maximum-onset rule (one consonant goes to the next syllable, a
    cluster keeps the last one as the onset) with common digraphs kept
    together. Silent word-final ``e`` is not treated as its own syllable.
    """
    out: list[str] = []
    for word in re.findall(r"[A-Za-z']+|\d+", text):
        w = word.lower().replace("'", "")
        if not w:
            continue
        groups: list[tuple[int, int]] = []
        i = 0
        while i < len(w):
            if w[i] in _EN_VOWELS:
                j = i
                while j < len(w) and w[j] in _EN_VOWELS:
                    j += 1
                groups.append((i, j))
                i = j
            else:
                i += 1
        if not groups:
            out.append(word)
            continue
        # Word-final silent e ("love", "name") is not a syllable of its own.
        if (
            len(groups) > 1
            and groups[-1] == (len(w) - 1, len(w))
            and w[-1] == "e"
        ):
            groups.pop()
        boundaries = [0]
        for k in range(1, len(groups)):
            prev_end = groups[k - 1][1]
            next_start = groups[k][0]
            between = next_start - prev_end
            if between <= 0:
                cut = next_start
            elif between == 1:
                cut = next_start  # single consonant -> onset of next
            else:
                # Keep digraphs (th, sh, ch, ph, wh, ck, ng) whole.
                pair = w[prev_end : prev_end + 2]
                cut = prev_end + 1 if pair not in {"th", "sh", "ch", "ph", "wh"} else prev_end + 2
                if cut >= next_start:
                    cut = next_start
            boundaries.append(cut)
        boundaries.append(len(w))
        for a, b in zip(boundaries, boundaries[1:]):
            if b > a:
                out.append(w[a:b])
    return out


def split_syllables(text: str) -> list[str]:
    """Syllabify Tamil or Latin text (auto-detected per line)."""
    if any(_is_tamil(ch) for ch in text):
        return split_tamil_syllables(text)
    return split_english_syllables(text)


def _syllable_weight(syllable: str) -> float:
    """Relative duration: long vowels and codas are held longer."""
    if not any(_is_tamil(ch) for ch in syllable):
        # Latin: one unit per vowel group, a touch longer for silent tails.
        return 1.0 + 0.15 * len([c for c in syllable.lower() if c in _EN_VOWELS])
    weight = 1.0
    if any(ch in _LONG_MARKS for ch in syllable):
        weight *= 1.8
    if syllable.endswith(_VIRAMA):
        weight *= 0.85
    if len(syllable) >= 4:
        weight *= 1.15  # consonant clusters take longer to articulate
    return weight


# --------------------------------------------------------------------------- #
# 2. Melody extraction
# --------------------------------------------------------------------------- #
def _skyline(notes: Iterable[MelodyNote]) -> list[MelodyNote]:
    """Highest sounding pitch at every moment (the classic skyline melody)."""
    events: list[tuple[float, int, int, int]] = []
    for n in notes:
        if n.end <= n.start:
            continue
        events.append((n.start, 1, n.pitch, n.velocity))
        events.append((n.end, -1, n.pitch, n.velocity))
    if not events:
        return []
    # At equal times, note-ons first so a re-articulated pitch survives.
    events.sort(key=lambda e: (e[0], -e[1]))

    active: dict[int, int] = {}
    segments: list[MelodyNote] = []
    prev_time: float | None = None
    prev_top: int | None = None
    for t, kind, pitch, vel in events:
        if prev_time is not None and prev_top is not None and t > prev_time + 1e-6:
            segments.append(MelodyNote(prev_time, t, prev_top, active.get(prev_top, vel)))
        if kind == 1:
            active[pitch] = vel
        else:
            active.pop(pitch, None)
        prev_time, prev_top = t, (max(active) if active else None)
    return segments


def _merge_repeats(notes: Sequence[MelodyNote], gap: float = 0.08) -> list[MelodyNote]:
    """Join adjacent segments of the same pitch separated by a tiny gap."""
    out: list[MelodyNote] = []
    for n in notes:
        if (
            out
            and n.pitch == out[-1].pitch
            and n.start - out[-1].end <= gap
            and n.start >= out[-1].start
        ):
            out[-1] = MelodyNote(
                out[-1].start, max(out[-1].end, n.end), n.pitch, max(out[-1].velocity, n.velocity)
            )
        else:
            out.append(MelodyNote(n.start, n.end, n.pitch, n.velocity))
    return out


def extract_lead_track(pm: Any) -> list[MelodyNote]:
    """Pick the most "lead-like" instrument and return its notes.

    The skyline works when one instrument carries the tune, but it also
    chops that tune into 40 ms slivers whenever a pad or a high arpeggio
    sounds above it. In a generated arrangement the lead is usually the
    instrument with the most notes that is *monophonic* and sits in a
    singable register - that is exactly what this scores.
    """
    best: list[MelodyNote] | None = None
    best_score = float("-inf")
    for inst in pm.instruments:
        if inst.is_drum or not inst.notes:
            continue
        notes = sorted(inst.notes, key=lambda n: n.start)
        span = notes[-1].end - notes[0].start
        if span <= 1.0 or len(notes) < 8:
            continue
        polyphony = sum(n.end - n.start for n in notes) / max(span, 1e-6)
        median_pitch = float(np.median([n.pitch for n in notes]))
        monophonic = 1.0 if polyphony <= 1.4 else max(0.35, 1.4 / polyphony)
        score = len(notes) * monophonic - 0.6 * abs(median_pitch - 67.0)
        if score > best_score:
            best_score = score
            best = [
                MelodyNote(float(n.start), float(n.end), int(n.pitch), int(n.velocity))
                for n in notes
                if n.end - n.start >= 0.03
            ]
    return best or []


def extract_melody(pm: Any, *, min_note: float = 0.05) -> list[MelodyNote]:
    """Extract a monophonic lead line from a ``pretty_midi`` object."""
    notes: list[MelodyNote] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.end - n.start >= min_note:
                notes.append(MelodyNote(float(n.start), float(n.end), int(n.pitch), int(n.velocity)))
    return _merge_repeats(_skyline(notes))


def _tempo_of(pm: Any, fallback: float = 120.0) -> float:
    """First sane tempo in a MIDI file, or ``fallback``."""
    try:
        _, tempos = pm.get_tempo_changes()
        valid = [float(t) for t in np.atleast_1d(tempos) if 20.0 <= float(t) <= 400.0]
        return valid[0] if valid else fallback
    except Exception:  # noqa: BLE001 - broken tempo maps are common
        return fallback


def quantize_melody(
    notes: Sequence[MelodyNote],
    bpm: float = 120.0,
    *,
    subdivision: int = 2,
    merge_gap_cells: int = 1,
) -> list[MelodyNote]:
    """Turn a busy instrumental line into a steady, singable vocal rhythm.

    The generated lead is often a stream of 16th notes. A singer cannot
    articulate 8 syllables a second, so we sample the line on an eighth-note
    grid (at the song tempo), keeping the most prominent sounding pitch in
    each cell, merging repeated pitches, and then joining the result
    legato. Rests survive: cells with nothing sounding stay empty.
    """
    if not notes:
        return []
    grid = 60.0 / max(20.0, min(400.0, bpm)) / max(1, subdivision)
    start = min(n.start for n in notes)
    end = max(n.end for n in notes)
    first = int(math.floor(start / grid))
    last = int(math.ceil(end / grid))
    cells: list[MelodyNote] = []
    for c in range(first, last + 1):
        t = (c + 0.5) * grid
        active = [n for n in notes if n.start - 1e-6 <= t < n.end + 1e-6]
        if not active:
            continue
        best = max(active, key=lambda n: (n.duration, n.velocity))
        cells.append(MelodyNote(c * grid, (c + 1) * grid, best.pitch, best.velocity))

    # Merge repeated pitches across small gaps (a held note, not a retrigger).
    merged: list[MelodyNote] = []
    for n in cells:
        if (
            merged
            and n.pitch == merged[-1].pitch
            and n.start - merged[-1].end <= merge_gap_cells * grid + 1e-6
        ):
            merged[-1].end = n.end
        else:
            merged.append(n)
    # Legato: hold each note until the next one starts (up to 1.2 s), so the
    # voice phrases instead of stuttering.
    for a, b in zip(merged, merged[1:]):
        gap = b.start - a.start
        if gap <= 1.2:
            a.end = max(a.start + grid * 0.5, b.start - 0.02)
    return merged


def simplify_melody(
    notes: Sequence[MelodyNote],
    *,
    min_duration: float = 0.09,
    max_rate: float = 3.0,
    merge_gap: float = 0.08,
) -> list[MelodyNote]:
    """Make the skyline singable: no machine-gun notes, no tiny slivers.

    Overlapping/short skyline segments are common when a piano arpeggio and
    a pad sound at once. Singing one syllable per 40 ms is impossible, so
    notes shorter than ``min_duration`` are dropped and the local density is
    capped at ``max_rate`` notes per second (sliding window, keeping the
    longest note in each window rather than skipping whole groups).
    """
    notes = [n for n in _merge_repeats(notes, merge_gap) if n.duration >= min_duration]
    if not notes:
        return []
    window = 1.0 / max(max_rate, 0.5)
    kept: list[MelodyNote] = []
    for n in notes:
        if not kept or n.start - kept[-1].start >= window:
            if kept and n.start < kept[-1].end - 0.02:
                n = MelodyNote(n.start, max(n.start + min_duration, kept[-1].end), n.pitch, n.velocity)
            kept.append(n)
        elif n.duration > kept[-1].duration:
            # Replace the previous pick but keep the time slot monotonic.
            kept[-1] = n
    # Close accidental overlaps so each note owns a slice of time.
    for a, b in zip(kept, kept[1:]):
        if a.end > b.start:
            a.end = max(a.start + min_duration, b.start)
    return [n for n in kept if n.duration >= min_duration]


def segment_phrases(
    notes: Sequence[MelodyNote],
    *,
    min_gap: float = 0.45,
    min_phrase: float = 1.2,
    max_phrase: float = 10.0,
) -> list[list[MelodyNote]]:
    """Split a melody into phrases at rests, then enforce length bounds."""
    if not notes:
        return []
    phrases: list[list[MelodyNote]] = [[notes[0]]]
    for prev, n in zip(notes, notes[1:]):
        if n.start - prev.end >= min_gap:
            phrases.append([])
        phrases[-1].append(n)

    def total(p: Sequence[MelodyNote]) -> float:
        return p[-1].end - p[0].start

    merged: list[list[MelodyNote]] = []
    for p in phrases:
        if merged and total(merged[-1]) < min_phrase and total(merged[-1]) + total(p) <= max_phrase:
            gap = p[0].start - merged[-1][-1].end
            if gap <= min_gap * 2:
                merged[-1].extend(p)
                continue
        merged.append(list(p))

    # Split anything still too long at its widest internal rest.
    out: list[list[MelodyNote]] = []
    for p in merged:
        stack = [p]
        while stack:
            cur = stack.pop()
            if total(cur) <= max_phrase or len(cur) < 3:
                out.append(cur)
                continue
            gaps = [(b.start - a.end, k) for k, (a, b) in enumerate(zip(cur, cur[1:]))]
            gap, k = max(gaps)
            if gap < 0.2:
                k = len(cur) // 2
            stack.append(cur[k + 1 :])
            stack.append(cur[: k + 1])
    out.sort(key=lambda p: p[0].start)
    return [p for p in out if p and total(p) >= 0.7]


def allocate_line_sections(
    notes: Sequence[MelodyNote],
    n_lines: int,
    syllable_counts: Sequence[int] | None = None,
    *,
    intro_frac: float = 0.05,
    outro_frac: float = 0.03,
    snap_window: float = 1.5,
) -> list[list[MelodyNote]]:
    """Split the melody timeline into one contiguous section per lyric line.

    Each line should occupy roughly its share of the song (by syllable
    count), and its notes must be contiguous so the vocal follows the tune
    in order. Boundaries are snapped to the nearest rest inside the section,
    which keeps phrases from being cut mid-note. If snapping starves a
    section, we fall back to splitting by note count rather than time.
    """
    if n_lines <= 0 or not notes:
        return []
    notes = sorted(notes, key=lambda n: n.start)
    start, end = notes[0].start, notes[-1].end
    span = max(end - start, 1e-6)
    lo = start + intro_frac * span
    hi = end - outro_frac * span
    weights = list(syllable_counts or [1] * n_lines)[:n_lines]
    if len(weights) < n_lines:
        weights += [1] * (n_lines - len(weights))
    total = float(sum(weights)) or 1.0
    cum = np.cumsum(weights) / total

    rests = [
        (a.end + b.start) * 0.5
        for a, b in zip(notes, notes[1:])
        if b.start - a.end >= 0.12
    ]
    bounds = [lo]
    for c in cum[:-1]:
        target = lo + (hi - lo) * float(c)
        lower = bounds[-1] + 0.4
        pick = target
        near = [r for r in rests if abs(r - target) <= snap_window and r > lower]
        if near:
            pick = min(near, key=lambda r: abs(r - target))
        else:
            starts = [n.start for n in notes if n.start > lower and abs(n.start - target) <= snap_window]
            if starts:
                pick = min(starts, key=lambda s: abs(s - target))
        bounds.append(max(pick, lower))
    bounds.append(hi)

    sections: list[list[MelodyNote]] = []
    for b0, b1 in zip(bounds, bounds[1:]):
        sections.append([n for n in notes if b0 <= n.start < b1])
    # A boundary that landed on a rest can leave an empty slice; fall back
    # to splitting by note count (weighted the same way) for all sections.
    if any(not s for s in sections):
        cuts = [0]
        for c in cum[:-1]:
            cuts.append(int(round(float(c) * len(notes))))
        cuts.append(len(notes))
        for i in range(1, len(cuts)):
            cuts[i] = max(cuts[i], cuts[i - 1] + 1)
        sections = [list(notes[a:b]) for a, b in zip(cuts, cuts[1:])]
        sections = [s for s in sections if s]
    return [s for s in sections if s]


def choose_phrases(
    phrases: Sequence[Sequence[MelodyNote]],
    n_lines: int,
    *,
    intro_frac: float = 0.06,
    outro_frac: float = 0.05,
) -> list[list[MelodyNote]]:
    """Pick ``n_lines`` non-overlapping phrases spread across the song."""
    if n_lines <= 0 or not phrases:
        return []
    song_start = phrases[0][0].start
    song_end = phrases[-1][-1].end
    span = max(song_end - song_start, 1e-6)
    usable = [
        list(p)
        for p in phrases
        if p[0].start >= song_start + intro_frac * span
        and p[-1].end <= song_end - outro_frac * span
    ]
    pool = usable or [list(p) for p in phrases]
    if len(pool) >= n_lines:
        idx = np.linspace(0, len(pool) - 1, n_lines).round().astype(int)
        picks: list[list[MelodyNote]] = []
        for k in idx:
            if not picks or pool[int(k)][0].start >= picks[-1][-1].end:
                picks.append(pool[int(k)])
        # np.linspace can repeat an index when there are just enough
        # phrases; fill any shortfall with the remaining usable phrases.
        used = {p[0].start for p in picks}
        for p in pool:
            if len(picks) >= n_lines:
                break
            if p[0].start not in used:
                picks.append(p)
                used.add(p[0].start)
        picks.sort(key=lambda p: p[0].start)
        return picks[:n_lines]

    # Fewer phrases than lines: cycle, but shift each reuse so the vocal
    # does not repeat the same notes at exactly the same pitch.
    picks = []
    for i in range(n_lines):
        p = list(pool[i % len(pool)])
        if picks and p[0].start < picks[-1][-1].end + 0.5:
            shift = picks[-1][-1].end + 0.5 - p[0].start
            p = [MelodyNote(n.start + shift, n.end + shift, n.pitch, n.velocity) for n in p]
        picks.append(p)
    return picks


# --------------------------------------------------------------------------- #
# 3. Audio primitives (numpy/scipy only)
# --------------------------------------------------------------------------- #
def _as_mono(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Polyphase resample; no-op when the rates already match."""
    x = _as_mono(audio)
    if src_rate == dst_rate or x.size == 0:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(src_rate), int(dst_rate))
    return resample_poly(x, dst_rate // g, src_rate // g).astype(np.float32)


def _stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    win = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad + hop), mode="reflect" if len(x) > pad else "constant")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[idx] * win, axis=1)


def _istft(spec: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.fft.irfft(spec, n=n_fft, axis=1).astype(np.float32) * win
    out = np.zeros((len(frames) - 1) * hop + n_fft, dtype=np.float32)
    wsum = np.zeros_like(out)
    for i, f in enumerate(frames):
        out[i * hop : i * hop + n_fft] += f
        wsum[i * hop : i * hop + n_fft] += win * win
    out /= np.maximum(wsum, 1e-4)
    pad = n_fft // 2
    return out[pad : len(out) - pad]


def _phase_vocoder(spec: np.ndarray, rate: float, n_fft: int, hop: int) -> np.ndarray:
    """Time-scale a complex STFT by ``rate`` (rate > 1 = longer).

    Standard phase-vocoder loop: the *analysis* phase (``prev_phase``) is
    kept separate from the accumulated *synthesis* phase. Mixing the two
    (a classic bug) smears transients and detunes the result.
    """
    n_bins = spec.shape[1]
    omega = 2.0 * np.pi * np.arange(n_bins) / n_fft
    mag = np.abs(spec)
    prev_phase = np.angle(spec[0])
    sum_phase = prev_phase.copy()
    synth_hop = max(1, int(round(hop * rate)))
    frames: list[np.ndarray] = []
    for i in range(1, len(spec)):
        dphi = np.angle(spec[i]) - prev_phase - omega * hop
        dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
        true_omega = omega + dphi / hop
        sum_phase = sum_phase + true_omega * synth_hop
        prev_phase = np.angle(spec[i])
        frames.append((mag[i] * np.exp(1j * sum_phase)).astype(np.complex64))
    if not frames:
        return spec
    return np.stack(frames)


def time_stretch(audio: np.ndarray, rate: float, sr: int = PROCESS_RATE) -> np.ndarray:
    """Change duration by ``rate`` (2.0 = twice as long) without pitch change."""
    x = _as_mono(audio)
    if x.size < 32 or abs(rate - 1.0) < 1e-3:
        n = max(1, int(round(len(x) * rate)))
        if n == len(x):
            return x.copy()
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    n_fft = 512 if sr <= 24000 else 1024
    hop = n_fft // 4
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)), mode="constant")
    spec = _stft(x, n_fft, hop)
    stretched = _istft(_phase_vocoder(spec, rate, n_fft, hop), n_fft, max(1, int(round(hop * rate))))
    target = max(1, int(round(len(x) * rate)))
    if len(stretched) >= target:
        return stretched[:target]
    return np.pad(stretched, (0, target - len(stretched))).astype(np.float32)


def pitch_shift(audio: np.ndarray, semitones: float, sr: int = PROCESS_RATE) -> np.ndarray:
    """Shift pitch by ``semitones`` keeping duration.

    Speed the waveform up (resample down) to raise the pitch, then stretch
    it back to the original duration with the phase vocoder. Getting the
    resample direction backwards is a classic bug: ``resample_poly(x, up,
    down)`` changes the *playback time base*, so upsampling lowers pitch.
    """
    x = _as_mono(audio)
    if x.size == 0 or abs(semitones) < 1e-3:
        return x.copy()
    ratio = float(2.0 ** (semitones / 12.0))
    frac = Fraction(ratio).limit_denominator(64)
    from scipy.signal import resample_poly

    # up=den, down=num -> shorter and higher by num/den = ratio.
    compressed = resample_poly(x, frac.denominator, frac.numerator).astype(np.float32)
    y = time_stretch(compressed, ratio, sr)
    # Match the original length exactly (rounding + vocoder padding).
    if len(y) > len(x):
        y = y[: len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def vibrato(
    audio: np.ndarray,
    sr: int = PROCESS_RATE,
    *,
    rate_hz: float = 5.4,
    depth_s: float = 0.0008,
    base_delay_s: float = 0.002,
) -> np.ndarray:
    """Add a real pitch vibrato via a modulated delay line.

    A delay ``d(t) = d0 + A sin(2πft)`` read back at the output rate makes
    the instantaneous pitch wobble by ``-d'(t)``. With A = 0.8 ms and
    f = 5.4 Hz that is about ±0.45 semitone - the depth of a gentle
    classical vibrato, not a warbling effect.
    """
    x = _as_mono(audio)
    if x.size < 8:
        return x
    n = np.arange(len(x), dtype=np.float64)
    delay = base_delay_s + depth_s * np.sin(2.0 * np.pi * rate_hz * n / sr)
    pos = n - delay * sr
    pos = np.clip(pos, 0, len(x) - 1.001)
    i0 = np.floor(pos).astype(np.int64)
    frac = (pos - i0).astype(np.float32)
    return ((1.0 - frac) * x[i0] + frac * x[np.minimum(i0 + 1, len(x) - 1)]).astype(np.float32)


def _trim_silence(audio: np.ndarray, sr: int, thresh_db: float = -42.0) -> np.ndarray:
    x = _as_mono(audio)
    if x.size == 0:
        return x
    env = np.abs(x)
    # 5 ms blocks: robust against single-sample spikes.
    block = max(1, int(0.005 * sr))
    n = len(env) // block * block
    if n == 0:
        return x
    rms = np.sqrt(np.mean(env[:n].reshape(-1, block) ** 2, axis=1))
    thresh = 10 ** (thresh_db / 20.0) * (rms.max() or 1.0)
    voiced = np.where(rms > thresh)[0]
    if voiced.size == 0:
        return x
    a = max(0, voiced[0] * block - int(0.01 * sr))
    b = min(len(x), (voiced[-1] + 1) * block + int(0.01 * sr))
    return x[a:b]


def estimate_f0(audio: np.ndarray, sr: int = PROCESS_RATE, *, fmin: float = 60.0, fmax: float = 500.0) -> float:
    """Median fundamental of voiced frames (autocorrelation), in Hz."""
    x = _as_mono(audio)
    if x.size < int(0.05 * sr):
        return 0.0
    frame = max(64, int(0.04 * sr))
    hop = max(32, int(0.01 * sr))
    lag_min = max(2, int(sr / fmax))
    lag_max = min(frame - 2, int(sr / fmin))
    vals: list[float] = []
    for start in range(0, len(x) - frame, hop):
        f = x[start : start + frame].astype(np.float64)
        f -= f.mean()
        e = float(np.dot(f, f))
        if e < 1e-6:
            continue
        ac = np.correlate(f, f, mode="full")[frame - 1 :]
        ac /= ac[0] if ac[0] else 1.0
        seg = ac[lag_min : lag_max + 1]
        if seg.size == 0:
            continue
        k = int(np.argmax(seg))
        peak = float(seg[k])
        if peak < 0.35:
            continue
        vals.append(sr / float(lag_min + k))
    if not vals:
        return 0.0
    return float(np.median(vals))


def fit_duration(
    audio: np.ndarray,
    target_sec: float,
    sr: int = PROCESS_RATE,
    *,
    max_stretch: float = 2.6,
    min_stretch: float = 0.45,
) -> np.ndarray:
    """Stretch to a target duration without absurd artefacts.

    Very long notes are sustained by looping the tail (a vowel) with
    crossfades rather than stretching 10x, which would smear the consonant
    into mush. Very short targets are compressed at most to ``min_stretch``;
    the caller can let the syllable run slightly into the next note.
    """
    x = _as_mono(audio)
    if x.size == 0 or target_sec <= 0:
        return x
    cur = len(x) / sr
    rate = target_sec / cur
    if rate <= max_stretch and rate >= min_stretch:
        return time_stretch(x, rate, sr)
    if rate > max_stretch:
        y = time_stretch(x, max_stretch, sr)
        need = int(round(target_sec * sr)) - len(y)
        if need <= 0:
            return y
        tail_len = max(int(0.08 * sr), int(0.35 * len(y)))
        tail = y[-tail_len:].copy()
        fade = max(1, int(0.03 * sr))
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        tail[:fade] *= ramp
        tail[-fade:] *= ramp[::-1]
        # Tile the tail with a periodic seam so the hold does not pulse.
        reps = int(math.ceil(need / max(1, len(tail) - fade))) + 1
        looped = np.tile(tail, reps)[: need + fade]
        y = np.concatenate([y, looped[fade : fade + need]])
        return y
    return time_stretch(x, min_stretch, sr)


# --------------------------------------------------------------------------- #
# 4. Syllable <-> note assignment
# --------------------------------------------------------------------------- #
def _active_runs(notes: Sequence[MelodyNote], gap: float = 0.45) -> list[list[MelodyNote]]:
    """Group notes into contiguous sounding runs separated by rests."""
    if not notes:
        return []
    runs: list[list[MelodyNote]] = [[notes[0]]]
    for n in notes[1:]:
        if n.start - runs[-1][-1].end >= gap:
            runs.append([])
        runs[-1].append(n)
    return [r for r in runs if r[-1].end - r[0].start >= 0.3]


def _pitch_at(notes: Sequence[MelodyNote], t: float) -> int:
    """Pitch of the note sounding at ``t``, else the nearest note's pitch."""
    for n in notes:
        if n.start <= t < n.end:
            return n.pitch
    return min(notes, key=lambda n: abs((n.start + n.end) * 0.5 - t)).pitch


def smooth_pitches(pitches: Sequence[int], *, max_leap: int = 7) -> list[int]:
    """Median-smooth a contour, then fold octaves to avoid un-singable leaps."""
    if not pitches:
        return []
    p = list(pitches)
    if len(p) >= 3:
        med = p[:]
        for i in range(1, len(p) - 1):
            med[i] = int(np.median(p[i - 1 : i + 2]))
        p = med
    out = [p[0]]
    for v in p[1:]:
        while v - out[-1] > max_leap:
            v -= 12
        while v - out[-1] < -max_leap:
            v += 12
        out.append(int(v))
    return out


def plan_line_grid(
    syllables: Sequence[str],
    notes: Sequence[MelodyNote],
    *,
    rest_gap: float = 0.45,
) -> list[SungUnit]:
    """Give every syllable its own sung note inside the line's phrases.

    This is the fallback (and the default for sparse generated arrangements)
    when the melody does not contain enough notes to carry the words. Each
    active run of the section is filled with an even, singable rhythm sized
    to the number of syllables assigned to it; the *pitch* at each syllable
    comes from the generated lead line, so the vocal still follows the tune's
    contour. Every syllable is sung exactly once, inside a phrase, never
    crammed into a 50 ms sliver.
    """
    if not syllables or not notes:
        return []
    runs = _active_runs(notes, rest_gap) or [list(notes)]
    n = len(syllables)
    span_start, span_end = notes[0].start, notes[-1].end
    span = max(0.3, span_end - span_start)

    # Split across the section's phrases only when every phrase can give a
    # syllable a humane duration (>=0.22 s). Otherwise sing across the whole
    # section span: stretching a line over a sparse bar is musical; cramming
    # 9 syllables a second is not.
    durations = np.array([max(0.3, r[-1].end - r[0].start) for r in runs])
    shares = durations / durations.sum() * n
    use_runs = len(runs) > 1 and bool(np.all(durations / np.maximum(shares, 1) >= 0.22))

    if use_runs:
        edges = np.round(np.cumsum(durations) / durations.sum() * n).astype(int)
        counts = np.diff(np.concatenate([[0], edges])).tolist()
        diff = n - sum(counts)
        if diff:
            k = int(np.argmax(counts))
            counts[k] = max(1, counts[k] + diff)
        plan = list(zip(runs, counts))
    else:
        plan = [([notes[0], notes[-1]], n)]

    units: list[SungUnit] = []
    idx = 0
    for run, count in plan:
        count = max(0, int(count))
        if count == 0:
            continue
        start, end = run[0].start, run[-1].end
        step = max(0.12, (end - start) / count)
        for k in range(count):
            if idx >= n:
                break
            t = start + k * step
            pitch = _pitch_at(notes, t + step * 0.5)
            units.append(
                SungUnit(
                    syllable=syllables[idx],
                    start=t,
                    duration=max(0.1, min(1.2, step * 0.9)),
                    pitch=pitch,
                    n_notes=1,
                )
            )
            idx += 1
    # Any syllables left over (degenerate rounds) go on the final note.
    while idx < n and units:
        last = units[-1]
        units.append(
            SungUnit(syllables[idx], last.start + last.duration, last.duration, last.pitch)
        )
        idx += 1

    smoothed = smooth_pitches([u.pitch for u in units])
    for u, p in zip(units, smoothed):
        u.pitch = p
    return units


def plan_line(
    syllables: Sequence[str],
    notes: Sequence[MelodyNote],
    *,
    line_start: float = 0.0,
    articulation_gap: float = 0.0,
) -> list[SungUnit]:
    """Place every syllable on the phrase's notes, in order.

    * More notes than syllables: a syllable may span several notes (a held
      note / melisma). It is sung at the pitch of the longest note in its
      slice, which keeps the strong beat rather than a passing tone.
    * More syllables than notes: the note is subdivided evenly. This is the
      case where lyrics would otherwise not fit; splitting guarantees every
      syllable is sung inside the phrase.
    """
    if not syllables or not notes:
        return []
    notes = [n for n in notes if n.duration > 0.02]
    if not notes:
        return []
    units: list[SungUnit] = []
    n_s, n_n = len(syllables), len(notes)

    if n_n >= n_s:
        for i, syl in enumerate(syllables):
            a = i * n_n // n_s
            b = max(a + 1, (i + 1) * n_n // n_s)
            group = notes[a:b]
            first = notes[a]
            last = group[-1]
            strongest = max(group, key=lambda n: (n.duration, n.velocity))
            units.append(
                SungUnit(
                    syllable=syl,
                    start=first.start + line_start,
                    duration=max(0.06, last.end - first.start - articulation_gap),
                    pitch=strongest.pitch,
                    n_notes=len(group),
                )
            )
    else:
        # Distribute the syllables over the notes, as evenly as possible.
        for j, note in enumerate(notes):
            a = j * n_s // n_n
            b = max(a + 1, (j + 1) * n_s // n_n)
            group = syllables[a:b]
            if not group:
                continue
            share = note.duration / len(group)
            for k, syl in enumerate(group):
                units.append(
                    SungUnit(
                        syllable=syl,
                        start=note.start + line_start + k * share,
                        duration=max(0.05, share - articulation_gap),
                        pitch=note.pitch,
                    )
                )
    return units


# --------------------------------------------------------------------------- #
# 5. Line rendering
# --------------------------------------------------------------------------- #
def _energy_curve(audio: np.ndarray, sr: int, hop_s: float = 0.005) -> np.ndarray:
    hop = max(1, int(hop_s * sr))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    frames = audio[: n * hop].reshape(n, hop)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12).astype(np.float32)


def _spoken_boundaries(
    audio: np.ndarray,
    weights: Sequence[float],
    sr: int,
) -> list[tuple[int, int]]:
    """Cut one spoken line into syllable slices.

    Proportions come from syllable weights; each cut is then nudged to the
    nearest energy valley (the gap between two spoken syllables), which is a
    cheap stand-in for forced alignment and noticeably better than a fixed
    split.
    """
    n = len(audio)
    total_w = float(sum(weights)) or 1.0
    expected = np.cumsum([0.0] + [w / total_w for w in weights]) * n
    energy = _energy_curve(audio, sr)
    hop = n / max(1, len(energy))

    cuts = [0.0]
    for k in range(1, len(weights)):
        e = float(expected[k])
        search = max(1, int(0.18 * (n / max(1, len(weights)))))
        lo = max(1, int(e) - search)
        hi = min(n - 1, int(e) + search)
        if hi > lo + 2:
            seg = energy[max(0, int(lo / hop)) : min(len(energy), int(hi / hop) + 1)]
            if seg.size:
                local = int(np.argmin(seg)) + int(lo / hop)
                e = float(local * hop)
        cuts.append(e)
    cuts.append(float(n))
    cuts = [max(0, min(n, int(round(c)))) for c in cuts]
    for i in range(1, len(cuts)):
        cuts[i] = max(cuts[i], cuts[i - 1] + 1)
    cuts[-1] = n
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def fit_pitch(
    pitch: int,
    natural_f0: float,
    *,
    max_semitones: float = 5.5,
    compression: float = 0.7,
) -> float:
    """Target frequency for a MIDI pitch, relative to the speaker's voice.

    Speech can only be shifted so far before it sounds like a chipmunk. We
    therefore fold the melody by octaves into a ±``max_semitones`` window
    around the speaker's median F0 and gently compress large intervals.
    The melodic contour (up/down) survives; the absolute register becomes
    one the synthetic voice can actually produce.
    """
    if natural_f0 <= 0:
        natural_f0 = 180.0
    f = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
    semis = 12.0 * math.log2(f / natural_f0)
    semis = ((semis + max_semitones) % 12.0) - max_semitones
    semis = float(np.clip(semis * compression, -max_semitones, max_semitones))
    return float(natural_f0 * 2.0 ** (semis / 12.0))


def render_line(
    audio: np.ndarray,
    units: Sequence[SungUnit],
    *,
    sr: int = PROCESS_RATE,
    natural_f0: float = 0.0,
    vibrato_semitones: float = 0.45,
) -> np.ndarray:
    """Turn one spoken line into a sung line placed at absolute times.

    Returns a mono buffer from t=0 to the last note's end (the caller
    offsets it to the phrase start).
    """
    x = _trim_silence(_as_mono(audio), sr)
    if not units or x.size == 0:
        return np.zeros(0, dtype=np.float32)
    if natural_f0 <= 0:
        natural_f0 = estimate_f0(x, sr) or 180.0

    weights = [_syllable_weight(u.syllable) for u in units]
    slices = _spoken_boundaries(x, weights, sr)
    base = min(u.start for u in units)
    end = max(u.start + u.duration for u in units)
    total = max(1, int(math.ceil((end - base) * sr)) + int(0.05 * sr))
    out = np.zeros(total, dtype=np.float32)

    for idx, (unit, (b0, b1)) in enumerate(zip(units, slices)):
        seg = x[b0:b1]
        if seg.size < 16:
            continue
        target_f = fit_pitch(unit.pitch, natural_f0)
        seg_f0 = estimate_f0(seg, sr) or natural_f0
        semis = 12.0 * math.log2(max(target_f, 1.0) / max(seg_f0, 1.0))
        # Clamp to keep the artefacts in check even if the F0 estimate is odd.
        semis = float(np.clip(semis, -8.0, 8.0))
        seg = pitch_shift(seg, semis, sr)
        # Closed-loop correction: estimate the shifted note and nudge it onto
        # the target. One-shot shifting inherits any octave error in the
        # per-slice F0 estimate; this catches it.
        for _ in range(2):
            out_f0 = estimate_f0(seg, sr)
            if out_f0 <= 0:
                break
            err = 12.0 * math.log2(max(target_f, 1.0) / out_f0)
            if abs(err) <= 0.7:
                break
            seg = pitch_shift(seg, float(np.clip(err, -4.0, 4.0)), sr)
        seg = fit_duration(seg, max(0.06, unit.duration), sr)
        if unit.duration >= 0.3:
            seg = vibrato(seg, sr, depth_s=0.0008 * (vibrato_semitones / 0.45))
        # Never let one syllable run far into the next: trim with a fade.
        if idx + 1 < len(units):
            limit = max(0.04, units[idx + 1].start - unit.start)
            max_len = int(round(limit * sr))
            if len(seg) > max_len:
                seg = seg[:max_len].copy()
        # Short fades keep the joins click-free.
        fade = min(len(seg) // 2, max(1, int(0.008 * sr)))
        if fade > 1:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]
        i0 = max(0, int(round((unit.start - base) * sr)))
        n = min(len(seg), len(out) - i0)
        if n > 0:
            out[i0 : i0 + n] += seg[:n]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out /= peak
    return out


# --------------------------------------------------------------------------- #
# 6. Top level
# --------------------------------------------------------------------------- #
def _read_lyrics(lyrics: str | None, lyrics_file: str | Path | None) -> list[str]:
    text = lyrics or ""
    if lyrics_file:
        text = Path(lyrics_file).read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def sing_song(
    midi_path: str | Path,
    *,
    lyrics: str | None = None,
    lyrics_file: str | Path | None = None,
    out_path: str | Path | None = None,
    instrumental_wav: str | Path | None = None,
    voice: str = DEFAULT_VOICE,
    device: str = "cpu",
    vocal_gain_db: float = 1.5,
    sample_rate: int = 44100,
    master: bool = True,
    humanize: bool = True,
    humanize_seed: int = 0,
    render_backend: str = "auto",
    stems: bool = False,
) -> SingResult:
    """Sing the lyrics on the generated melody and produce a finished mix.

    Pipeline::

        MIDI -> (humanise) -> render WAV -> extract melody -> phrase split
             -> syllabify lyrics -> syllable/note alignment -> pitch-shaped
             TTS -> master (EQ, compression, reverb, ducking, limiter) -> WAV

    If ``instrumental_wav`` is given it is used as the backing track (no
    rendering, so a soundfont is not needed); the MIDI is still used for the
    melody.
    """
    import soundfile as sf

    from .vocals import synthesize_lines

    lines = _read_lyrics(lyrics, lyrics_file)
    if not lines:
        raise ValueError("no lyrics given - pass lyrics= or lyrics_file=")
    midi_path = Path(midi_path)
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI not found: {midi_path}")

    warnings: list[str] = []
    workdir = Path(tempfile.mkdtemp(prefix="promptmusic_sing_"))

    # --- backing track ---------------------------------------------------- #
    if instrumental_wav is None:
        import pretty_midi

        from .render import render_midi

        play_midi = midi_path
        if humanize:
            from .render import humanize_midi

            play_midi = workdir / f"{midi_path.stem}_humanized.mid"
            humanize_midi(midi_path, play_midi, seed=humanize_seed)
        rendered = render_midi(play_midi, workdir / "instrumental.wav", backend=render_backend)
        instrumental_wav = rendered.wav_path
        if rendered.warnings:
            warnings.extend(rendered.warnings)
    instrumental_wav = Path(instrumental_wav)
    instrumental, inst_rate = sf.read(str(instrumental_wav), dtype="float32", always_2d=True)
    if inst_rate != sample_rate:
        from .vocals import _resample

        instrumental = np.stack(
            [_resample(instrumental[:, c], inst_rate, sample_rate) for c in range(instrumental.shape[1])],
            axis=1,
        )
    total_sec = len(instrumental) / sample_rate

    # --- melody ----------------------------------------------------------- #
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    lead = extract_lead_track(pm) or extract_melody(pm)
    bpm = _tempo_of(pm)
    melody = simplify_melody(
        quantize_melody(lead, bpm), min_duration=0.15, max_rate=4.0
    )
    phrases = segment_phrases(melody)
    if not melody or not phrases:
        raise ValueError("the MIDI has no usable melody notes")
    LOG.info(
        "melody: %d lead notes at %.0f BPM -> %d singable notes, %d phrases",
        len(lead), bpm, len(melody), len(phrases),
    )

    # --- syllabify -------------------------------------------------------- #
    syllables_per_line = [split_syllables(line) for line in lines]
    if any(not s for s in syllables_per_line):
        raise ValueError("could not split the lyrics into syllables")
    n_syllables = sum(len(s) for s in syllables_per_line)

    # One contiguous melody section per line, sized by syllable count. This
    # is what makes the words fit the music: a line cannot run out of notes
    # because its section is chosen to carry that many syllables.
    sections = allocate_line_sections(
        melody, len(lines), [len(s) for s in syllables_per_line]
    )
    if len(sections) < len(lines):
        warnings.append(
            f"only {len(sections)} melody sections for {len(lines)} lines; "
            "the last line(s) were dropped"
        )
        lines = lines[: len(sections)]
        syllables_per_line = syllables_per_line[: len(sections)]
    n_syllables = sum(len(s) for s in syllables_per_line)

    # --- speak each line once --------------------------------------------- #
    LOG.info("speaking %d line(s) with %s", len(lines), voice)
    spoken, voice_rate = synthesize_lines(lines, voice=voice, device=device)
    spoken = [resample_audio(s, voice_rate, PROCESS_RATE) for s in spoken]
    natural = [f for f in (estimate_f0(s, PROCESS_RATE) for s in spoken) if f > 0]
    natural_f0 = float(np.median(natural)) if natural else 190.0
    LOG.info("speaker F0: %.0f Hz", natural_f0)

    # --- render each line on its section ---------------------------------- #
    vocal = np.zeros(int(total_sec * PROCESS_RATE) + PROCESS_RATE, dtype=np.float32)
    placements: list[float] = []
    for line, sylls, section, audio in zip(lines, syllables_per_line, sections, spoken):
        # If the section is note-rich, sing the actual notes; if it is sparse
        # (common for a short-trained model), give each syllable its own note
        # inside the phrase and take pitches from the lead contour.
        if len(section) >= 0.8 * len(sylls):
            units = plan_line(sylls, section, articulation_gap=0.015)
        else:
            units = plan_line_grid(sylls, section)
        if not units:
            continue
        offset = min(u.start for u in units)
        rel = [SungUnit(u.syllable, u.start - offset, u.duration, u.pitch, u.n_notes) for u in units]
        seg = render_line(audio, rel, sr=PROCESS_RATE, natural_f0=natural_f0)
        i0 = max(0, int(round(offset * PROCESS_RATE)))
        n = min(len(seg), len(vocal) - i0)
        if n > 0:
            vocal[i0 : i0 + n] += seg[:n]
            placements.append(round(offset, 2))

    vocal = vocal[: int(total_sec * PROCESS_RATE)]
    if not np.any(vocal):
        raise ValueError("vocal rendering produced silence - check the lyrics/voice")

    # --- resample vocal to the output rate -------------------------------- #
    vocal_out = resample_audio(vocal, PROCESS_RATE, sample_rate)

    # --- master and write -------------------------------------------------- #
    if out_path is None:
        out_path = midi_path.with_name(f"{midi_path.stem}_sing.wav")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stem_paths: list[str] = []
    if stems:
        for name, data in (("vocals", vocal_out), ("instrumental", instrumental)):
            sp = out_path.with_name(f"{out_path.stem}_{name}.wav")
            sf.write(str(sp), data.astype(np.float32), sample_rate)
            stem_paths.append(str(sp))

    if master:
        from .master import master_mix

        mixed = master_mix(
            instrumental,
            vocal_out,
            sample_rate,
            vocal_gain_db=vocal_gain_db,
            warnings=warnings,
        )
        mixed = _fit_channels(mixed, len(instrumental))
    else:
        gain = float(10.0 ** (vocal_gain_db / 20.0))
        mixed = instrumental.copy()
        n = min(len(mixed), len(vocal_out))
        mixed[:n] += vocal_out[:n, None] * gain
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > 0.99:
            mixed *= 0.99 / peak

    sf.write(str(out_path), mixed.astype(np.float32), sample_rate)

    pitches = [n.pitch for n in melody]
    return SingResult(
        wav_path=str(out_path),
        duration_sec=len(mixed) / sample_rate,
        n_lines=len(lines),
        n_syllables=n_syllables,
        n_melody_notes=len(melody),
        n_phrases=len(phrases),
        voice=voice,
        natural_f0_hz=natural_f0,
        melody_range=(min(pitches), max(pitches)),
        placements=placements,
        instrumental_path=str(instrumental_wav),
        midi_path=str(midi_path),
        stems=stem_paths,
        warnings=warnings,
    )


def _fit_channels(audio: np.ndarray, n_samples: int) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if len(a) >= n_samples:
        return a[:n_samples]
    return np.pad(a, ((0, n_samples - len(a)), (0, 0)))
