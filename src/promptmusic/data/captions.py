"""Reading and normalising the MidiCaps caption file.

``train.json`` for MidiCaps is ~428 MB of JSON. Loading it with
``json.load`` would materialise ~1.5-2.5 GB of Python objects, which is a
bad idea on an 8 GB laptop that also has PyTorch resident. We therefore
stream it with :mod:`ijson` and immediately convert each record into a
compact dataclass, then cache the result as JSON Lines so later runs are
instant.

Field reference (from the MidiCaps paper / dataset card)
--------------------------------------------------------
``location``       relative path of the MIDI file inside the archive
``caption``        one-sentence English description of the piece
``genre``          most likely genre label
``genre_prob``     confidence of the genre label
``mood``           most likely mood label
``mood_prob``      confidence of the mood label
``key``            estimated musical key
``time_signature`` e.g. ``"4/4"``
``tempo``          estimated BPM
``tempo_word``     verbal tempo, e.g. ``"moderate"``
``duration``       duration in seconds
``duration_word``  verbal duration, e.g. ``"medium"``
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils import logger, write_jsonl

LOG = logger("promptmusic.captions")

__all__ = [
    "CaptionRecord",
    "iter_captions",
    "load_caption_index",
    "write_caption_index",
    "read_caption_index",
    "resolve_midi_path",
    "build_prompt_from_features",
]


@dataclass
class CaptionRecord:
    """One (caption, MIDI) training pair plus its audio-derived features."""

    location: str
    caption: str = ""
    genre: str = ""
    genre_prob: float = 0.0
    mood: str = ""
    mood_prob: float = 0.0
    key: str = ""
    time_signature: str = ""
    tempo: float = 0.0
    tempo_word: str = ""
    duration: float = 0.0
    duration_word: str = ""
    # Populated by the translation step (Phase 2).
    caption_ta: str = ""
    # Populated when a lyric corpus is attached (Phase 3).
    lyrics: str = ""
    lyrics_lang: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        """A stable identifier derived from the MIDI location."""
        return Path(self.location).stem

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CaptionRecord:
        known = {f for f in cls.__dataclass_fields__}
        extras = {k: v for k, v in d.items() if k not in known}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs.setdefault("extras", {})
        kwargs["extras"].update(extras)
        return cls(**kwargs)

    def style_prompt(self) -> str:
        """Build a natural-language style prompt from the feature fields.

        This is what turns a bare caption into a richer conditioning signal
        that is closer to how a human would phrase a request.
        """
        bits: list[str] = []
        if self.genre:
            bits.append(f"{self.genre} music")
        if self.mood:
            bits.append(self.mood)
        if self.tempo_word:
            bits.append(f"{self.tempo_word} tempo")
        if self.key:
            bits.append(f"in {self.key}")
        if self.time_signature and self.time_signature not in {"4/4", ""}:
            bits.append(f"in {self.time_signature}")
        if self.duration_word:
            bits.append(f"{self.duration_word} length")
        tail = ", ".join(bits)
        caption = self.caption.strip().rstrip(".")
        return f"{caption}. {tail}." if tail and caption else (caption or tail)


def _strip_mid_path(location: str) -> str:
    """Turn ``MID_PATH/a/b/c.mid`` into ``a/b/c.mid``."""
    loc = location.replace("\\", "/")
    for prefix in ("MID_PATH/", "midicaps/", "MIDI/"):
        if loc.startswith(prefix):
            return loc[len(prefix) :]
    return loc


def iter_captions(path: Path | str, limit: int | None = None) -> Iterator[CaptionRecord]:
    """Stream caption records out of a MidiCaps-style JSON file.

    Uses :mod:`ijson` when available (constant memory) and falls back to
    :func:`json.load` otherwise.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"caption file not found: {p}")

    count = 0
    try:
        import ijson

        with open(p, "rb") as fh:
            for obj in ijson.items(fh, "item", use_float=True):
                yield CaptionRecord.from_dict(obj)
                count += 1
                if limit and count >= limit:
                    return
        return
    except ImportError:
        LOG.warning("ijson not installed - falling back to json.load (uses more RAM)")

    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("data", data.get("items", []))
    for obj in data:
        yield CaptionRecord.from_dict(obj)
        count += 1
        if limit and count >= limit:
            return


def load_caption_index(
    json_path: Path | str,
    *,
    limit: int | None = None,
    cache_path: Path | str | None = None,
) -> list[CaptionRecord]:
    """Load captions into memory, using a JSONL cache when possible."""
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        LOG.info("using caption cache %s", cache)
        return read_caption_index(cache, limit=limit)

    LOG.info("streaming captions from %s ...", json_path)
    records = list(iter_captions(json_path, limit=limit))
    LOG.info("loaded %d caption records", len(records))
    if cache:
        write_caption_index(records, cache)
        LOG.info("wrote caption cache %s", cache)
    return records


def write_caption_index(records: Iterable[CaptionRecord], path: Path | str) -> Path:
    """Persist caption records as JSON Lines."""
    return write_jsonl(path, (r.to_dict() for r in records))


def read_caption_index(path: Path | str, limit: int | None = None) -> list[CaptionRecord]:
    """Read a JSONL caption cache."""
    out: list[CaptionRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(CaptionRecord.from_dict(json.loads(line)))
            if limit and len(out) >= limit:
                break
    return out


def resolve_midi_path(location: str, midi_root: Path | str, candidates: int = 4) -> Path | None:
    """Find the MIDI file for a caption record.

    The ``location`` field in MidiCaps is a virtual path such as
    ``MID_PATH/0/0/0_0_...mid``. The archive layout can differ between
    releases, so we try a handful of sensible roots and also search the
    directory tree by file name as a last resort.
    """
    root = Path(midi_root)
    rel = _strip_mid_path(location)
    name = Path(rel).name

    roots = [
        root,
        root / "midicaps",
        root / "MIDI",
        root / "midi",
        root.parent,
    ]
    for r in roots:
        cand = r / rel
        if cand.is_file():
            return cand
        cand = r / name
        if cand.is_file():
            return cand

    # Fall back to a bounded recursive search by file name.
    try:
        for idx, p in enumerate(root.rglob(name)):
            if p.is_file():
                return p
            if idx > 20000:
                break
    except OSError:
        pass
    return None


def build_prompt_from_features(record: CaptionRecord, include_caption: bool = True) -> str:
    """The conditioning text used for training.

    We concatenate the human caption with the machine-derived features so
    the model can learn to react to concrete musical attributes (tempo,
    key, mood) as well as free-form description.
    """
    parts: list[str] = []
    if include_caption and record.caption:
        parts.append(record.caption.strip())
    attrs: list[str] = []
    if record.genre:
        attrs.append(record.genre)
    if record.mood:
        attrs.append(record.mood)
    if record.key:
        attrs.append(f"key {record.key}")
    if record.tempo:
        attrs.append(f"{record.tempo:.0f} bpm")
    if record.time_signature:
        attrs.append(f"time signature {record.time_signature}")
    if attrs:
        parts.append("(" + ", ".join(attrs) + ")")
    return ". ".join(p.rstrip(". ") for p in parts if p) + "."
