"""Phase 3: lyrics - pairing words with music.

What this module does
---------------------
A *lyric* is the text sung by a vocalist. To generate sung music we need three
things:

1. **A corpus**: MIDI files that already have lyrics, so the model can learn
   the relationship between words and notes. We use
   ``asigalov61/Lyrics-MIDI-Dataset`` (~179k MIDI files with lyrics).
2. **A vocabulary**: the set of lyric tokens the model may emit. Built by
   :class:`~promptmusic.data.tokenizer.LyricVocab` from the corpus.
3. **Alignment**: a rule for *where* the words go. We interleave lyric tokens
   at bar boundaries (see ``MusicTokenizer.interleave_lyrics``), which is a
   coarse but learnable alignment - it teaches the model that a word cluster
   belongs to a bar, and the notes in that bar are what is sung.

Why a separate lyric vocabulary rather than reusing mT5's?
---------------------------------------------------------
mT5 already knows Tamil and English words. But we cannot emit mT5 ids from the
music decoder - they would collide with the MIDI tokens. A small dedicated
vocabulary (a few thousand English words plus Tamil characters) keeps the
decoder's output space manageable and makes the lyric tokens interpretable:
you can read ``lyric_w_rain`` straight out of the token stream.

Licence note
------------
``Lyrics-MIDI-Dataset`` is **CC-BY-NC-SA-4.0** - non-commercial. This phase is
therefore *opt-in* (``data.lyrics.enabled: false`` by default) and the licence
is surfaced in the manifest. See ``docs/01_DATASETS.md``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..paths import DATA_DIR, ensure_dir
from ..utils import logger
from .tokenizer import LyricVocab, contains_tamil

LOG = logger("promptmusic.lyrics")

__all__ = [
    "LyricPair",
    "build_lyric_vocab",
    "iter_lyrics_midi",
    "download_lyrics_dataset",
    "prepare_lyric_corpus",
    "strip_lyrics_from_midi",
]


# --------------------------------------------------------------------------- #
# One (lyrics, MIDI) pair
# --------------------------------------------------------------------------- #
@dataclass
class LyricPair:
    """A MIDI file and the lyrics that go with it."""

    midi_path: str
    lyrics: str = ""
    language: str = "en"
    title: str = ""
    source: str = "asigalov61/Lyrics-MIDI-Dataset"
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def is_tamil(self) -> bool:
        return contains_tamil(self.lyrics)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Reading lyrics out of MIDI files
# --------------------------------------------------------------------------- #
def strip_lyrics_from_midi(midi_path: str | Path) -> str:
    """Extract the lyric text stored inside a MIDI file.

    MIDI has a ``lyrics`` meta-event (and a ``text`` event) that many sequencers
    write when a song has words. Not every file has them - when there are none
    this returns an empty string and the file is skipped.
    """
    try:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        chunks: list[str] = []
        for inst in pm.instruments:
            for ev in getattr(inst, "lyrics", []) or []:
                text = getattr(ev, "text", "")
                if text:
                    chunks.append(text)
        # Some encoders put words on the top-level instrument list instead.
        for inst in getattr(pm, "instruments", []):
            for ev in getattr(inst, "lyrics", []) or []:
                text = getattr(ev, "text", "")
                if text and text not in chunks:
                    chunks.append(text)
        return _clean(" ".join(chunks))
    except Exception as exc:  # noqa: BLE001 - a malformed file must not stop the scan
        LOG.debug("could not read lyrics from %s: %s", midi_path, exc)
        return ""


def _clean(text: str) -> str:
    text = re.sub(r"[\r\n\t]+", " ", text or "")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Building the corpus
# --------------------------------------------------------------------------- #
def download_lyrics_dataset(
    data_root: Path | str | None = None,
    *,
    limit: int | None = None,
) -> Path:
    """Download the lyrics dataset with huggingface_hub.

    The corpus is large; pass ``limit`` to fetch only a slice of the files.
    Returns the directory the files landed in.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    root = Path(data_root or DATA_DIR) / "lyrics_midi"
    ensure_dir(root)
    LOG.info("downloading Lyrics MIDI Dataset -> %s", root)

    kwargs: dict[str, Any] = {
        "repo_id": "asigalov61/Lyrics-MIDI-Dataset",
        "repo_type": "dataset",
        "local_dir": str(root),
        "resume_download": True,
    }
    if limit:
        # The dataset is a flat tree of .mid files; cap the number fetched.
        kwargs["allow_patterns"] = ["*.mid", "*.midi", "*.txt", "*.json", "README*"]

    path = snapshot_download(**kwargs)
    LOG.info("lyrics dataset ready at %s", path)
    return Path(path)


def iter_lyrics_midi(
    root: Path | str,
    *,
    limit: int | None = None,
    min_lyric_length: int = 20,
):
    """Yield :class:`LyricPair` for every MIDI under ``root`` that has lyrics.

    Files whose lyrics are shorter than ``min_lyric_length`` characters are
    skipped - a one-word "la" is not a useful training signal.
    """
    root = Path(root)
    seen = 0
    for path in sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi")):
        if limit and seen >= limit:
            return
        lyrics = strip_lyrics_from_midi(path)
        if len(lyrics) < min_lyric_length:
            continue
        seen += 1
        yield LyricPair(
            midi_path=str(path),
            lyrics=lyrics,
            language="ta" if contains_tamil(lyrics) else "en",
            title=path.stem,
        )


def prepare_lyric_corpus(
    root: Path | str,
    *,
    limit: int | None = None,
    out_path: Path | str | None = None,
) -> tuple[list[LyricPair], LyricVocab]:
    """Scan a lyrics dataset, build the vocabulary, and cache the result.

    Returns ``(pairs, vocab)``. The corpus is written to JSONL so that the
    (slow) MIDI scan happens once.
    """
    cache = Path(out_path) if out_path else (Path(root) / "lyric_corpus.jsonl")

    pairs: list[LyricPair] = []
    if cache.exists():
        LOG.info("loading lyric corpus cache %s", cache)
        with open(cache, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    pairs.append(LyricPair(
                        midi_path=d["midi_path"], lyrics=d.get("lyrics", ""),
                        language=d.get("language", "en"), title=d.get("title", ""),
                    ))
        if limit:
            pairs = pairs[:limit]
    else:
        LOG.info("scanning %s for MIDI files with lyrics ...", root)
        for pair in iter_lyrics_midi(root, limit=limit):
            pairs.append(pair)
        LOG.info("found %d MIDI files with usable lyrics", len(pairs))
        ensure_dir(cache.parent)
        with open(cache, "w", encoding="utf-8") as fh:
            for pair in pairs:
                fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        LOG.info("cached lyric corpus to %s", cache)

    vocab = LyricVocab(mode="hybrid", max_vocab=8000)
    vocab.build([p.lyrics for p in pairs])
    return pairs, vocab


# --------------------------------------------------------------------------- #
# Attaching lyrics to caption records
# --------------------------------------------------------------------------- #
def attach_lyrics_to_records(
    records: list[Any],
    pairs: list[LyricPair],
) -> int:
    """Best-effort match of lyrics onto caption records.

    MidiCaps and the lyrics corpus describe different slices of the Lakh
    dataset, so an exact per-file match only sometimes works. When a MIDI file
    appears in both we attach its lyrics; when it does not, the record keeps an
    empty lyric field and training proceeds music-only for that example.

    Returns the number of records that received lyrics.
    """
    by_name: dict[str, LyricPair] = {}
    for pair in pairs:
        by_name[Path(pair.midi_path).stem.lower()] = pair

    matched = 0
    for record in records:
        key = Path(getattr(record, "location", "")).stem.lower()
        pair = by_name.get(key)
        if pair is not None:
            record.lyrics = pair.lyrics
            record.lyrics_lang = pair.language
            matched += 1

    LOG.info("attached lyrics to %d/%d records", matched, len(records))
    return matched
