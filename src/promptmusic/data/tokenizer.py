"""Turning MIDI files into a stream of token integers the model can learn.

The representation
------------------
We use **REMI** (Revamped MIDI-derived events), the encoding introduced with
Music Transformer and the one used by the reference *text2midi* model.

A REMI stream is a flat list of short symbol names, each describing one
event, in time order::

    Bar_None  TimeSig_4/4  Position_0  Tempo_121.29  Program_0
    Pitch_60  Velocity_87  Duration_1.0.4
    Program_25  Pitch_40  Velocity_87  Duration_1.0.4
    ...

Seven families of tokens carry the music:

``Bar``       the bar line - marks the start of a new measure
``Position``  the position inside the bar, on the beat grid
``Pitch``     the MIDI pitch number (21-108)
``Velocity``  the loudness, binned into ``num_velocities`` buckets
``Duration``  how long the note lasts, in beats
``Program``   the General MIDI instrument (0-127) of the track - this is what
              makes the model multi-instrument rather than piano-only
``Tempo`` / ``TimeSig``   the global tempo and meter

Why not raw MIDI bytes or a piano roll?
----------------------------------------
* **Short sequences.** A 30-second piece is ~1000 tokens instead of ~60 000
  piano-roll frames, so the model sees far more music per unit of compute.
* **Lossless enough.** Pitch, timing, duration, velocity and instrument all
  survive the round trip.
* **Standard.** Using REMI means our token stream is directly comparable with
  published work (text2midi, Music Transformer, MIDI-LLM).

Token ids
---------
miditok assigns every token in its vocabulary an integer id. We keep those
ids unchanged, which means the model's output can be handed straight back to
``tokenizer.decode(ids)`` to rebuild a MIDI file - no conversion layer in
between.

::

    0        PAD_None   (padding / mask)
    1        BOS_None
    2        EOS_None
    3        MASK_None
    4..V-1   the music tokens
    V        <unk>
    V+1..    lyric tokens (only when the lyrics phase is enabled)

Adding lyrics
-------------
We extend the vocabulary with a small family of *lyric tokens*::

    <lyric_start>  lyric_w_singing  lyric_c_த  <lyric_end>

Lyric tokens are interleaved into the music stream at bar boundaries, so the
model learns *when* a syllable belongs - that is the alignment between text
and music that a song generator needs. English words get a word token
(compact, and words carry meaning); Tamil text is broken into characters
because Tamil is agglutinative (a "word" can be a whole phrase) and because a
few hundred characters cover the script completely.
"""

from __future__ import annotations

import json
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils import logger

LOG = logger("promptmusic.tokenizer")

__all__ = [
    "LyricVocab",
    "MusicTokenizer",
    "build_tokenizer",
    "contains_tamil",
    "normalise_text",
    "MUSIC_FAMILIES",
]


# --------------------------------------------------------------------------- #
# Script detection
# --------------------------------------------------------------------------- #
#: Unicode ranges that make up the Tamil script (block + supplement).
TAMIL_RANGES: tuple[tuple[int, int], ...] = (
    (0x0B80, 0x0BFF),  # Tamil
    (0x200B, 0x200D),  # zero-width space / joiners
)


def is_tamil_char(ch: str) -> bool:
    """True if ``ch`` belongs to the Tamil script."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in TAMIL_RANGES)


def contains_tamil(text: str) -> bool:
    """True if the text contains at least one Tamil character."""
    return any(is_tamil_char(ch) for ch in text)


def normalise_text(text: str) -> str:
    """NFKC-normalise and collapse whitespace.

    Tamil text often uses combining marks; NFKC keeps them but makes
    compatibility characters consistent, which improves tokenizer sharing.
    """
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


#: Which token families exist in a REMI stream (used by docs and audits).
MUSIC_FAMILIES = ("Bar", "TimeSig", "Position", "Tempo", "Program", "Pitch", "Velocity", "Duration")


def _parse_beat_res(raw: Any) -> dict[tuple[int, int], int]:
    """Normalise a ``beat_res`` mapping into miditok's expected shape.

    miditok wants ``{(beat_x1, beat_x2): resolution}``. YAML cannot express a
    tuple key, so we accept several spellings:

    ``{"0,4": 4}``         the format used in this project's YAML configs
    ``{"(0, 4)": 4}``      the way Python prints a tuple
    ``{[0, 4]: 4}``        a JSON-ish list key
    ``{(0, 4): 4}``        already correct
    """
    out: dict[tuple[int, int], int] = {}
    items = raw.items() if hasattr(raw, "items") else raw
    for key, value in items:
        if isinstance(key, (tuple, list)) and len(key) == 2:
            span = (int(key[0]), int(key[1]))
        elif isinstance(key, str):
            parts = [p for p in key.strip("()[] ").replace(",", " ").split() if p]
            if len(parts) != 2:
                raise ValueError(f"cannot parse beat_res key {key!r}; expected e.g. '0,4'")
            span = (int(parts[0]), int(parts[1]))
        else:
            raise ValueError(f"cannot parse beat_res key {key!r}")
        out[span] = int(value)
    return out or {(0, 4): 4}


# --------------------------------------------------------------------------- #
# Lyric vocabulary
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[^\w']+", re.UNICODE)

UNK_TOKEN = "<unk>"
START_TOKEN = "<lyric_start>"
END_TOKEN = "<lyric_end>"


@dataclass
class LyricVocab:
    """Builds and applies the lyric token vocabulary.

    ``mode``:

    ``"hybrid"`` (default)
        English words become one token each; Tamil text is split into
        characters. This keeps sequences short for English (the bulk of the
        training data) while giving Tamil full coverage.
    ``"word"``
        Everything is split on whitespace; unknown words fall back to chars.
    ``"char"``
        Everything is split into characters. Simple and universal, but
        roughly 5x longer sequences.
    """

    mode: str = "hybrid"
    max_vocab: int = 8000
    word_tokens: list[str] = field(default_factory=list)
    char_tokens: list[str] = field(default_factory=list)

    #: Always-present Latin fallback so an unseen English word can be spelled
    #: out character-by-character. Without this, ``encode`` silently drops any
    #: word that was not in the training lyrics (char_tokens would only hold
    #: Tamil characters, which never cover English). Punctuation that appears
    #: in lyrics is included because it is part of the sung text.
    FALLBACK_CHARS: tuple[str, ...] = tuple(
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        ".,!?;:'-()"
    )

    # -- building ---------------------------------------------------------- #
    def build(self, lyrics: list[str]) -> LyricVocab:
        """Learn the vocabulary from a corpus of lyric strings."""
        word_counts: Counter[str] = Counter()
        char_counts: Counter[str] = Counter()

        for text in lyrics:
            if not text:
                continue
            for word in self._split_words(text):
                if contains_tamil(word):
                    for ch in word:
                        if not ch.isspace():
                            char_counts[ch] += 1
                else:
                    word_counts[word.lower()] += 1

        self.word_tokens = sorted(w for w, _ in word_counts.most_common(self.max_vocab))
        # Keep the Latin fallback (so unseen English words can be spelled out)
        # alongside whatever Tamil/other characters the corpus actually used.
        seen_chars = {c for c, _ in char_counts.most_common(self.max_vocab)}
        self.char_tokens = sorted(seen_chars | set(self.FALLBACK_CHARS))
        LOG.info(
            "lyric vocabulary: %d word tokens, %d char tokens",
            len(self.word_tokens),
            len(self.char_tokens),
        )
        return self

    @staticmethod
    def _split_words(text: str) -> list[str]:
        return [w for w in _WORD_RE.split(text) if w]

    # -- applying ---------------------------------------------------------- #
    def token_for_word(self, word: str) -> str | None:
        key = word.lower()
        return f"lyric_w_{key}" if key in self.word_tokens else None

    def token_for_char(self, ch: str) -> str | None:
        return f"lyric_c_{ch}" if ch in self.char_tokens else None

    def encode(self, text: str) -> list[str]:
        """Turn a lyric string into lyric token strings."""
        if not text:
            return []
        out: list[str] = []
        for word in self._split_words(text):
            if contains_tamil(word):
                for ch in word:
                    tok = self.token_for_char(ch)
                    if tok:
                        out.append(tok)
            else:
                tok = self.token_for_word(word)
                if tok:
                    out.append(tok)
                else:
                    out.extend(self._spell_out(word))
        return out

    def _spell_out(self, piece: str) -> list[str]:
        """Unknown word -> spell it out with character tokens (may be empty)."""
        return [t for ch in piece.lower() if (t := self.token_for_char(ch))]

    def decode(self, tokens: list[str]) -> str:
        """Turn lyric tokens back into readable text."""
        words = [t[len("lyric_w_") :] for t in tokens if t.startswith("lyric_w_")]
        chars = "".join(t[len("lyric_c_") :] for t in tokens if t.startswith("lyric_c_"))
        parts = [p for p in (" ".join(words), chars) if p]
        return " ".join(parts)

    def all_tokens(self) -> list[str]:
        return [START_TOKEN, END_TOKEN] + [
            *(f"lyric_w_{w}" for w in self.word_tokens),
            *(f"lyric_c_{c}" for c in self.char_tokens),
        ]

    # -- persistence ------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "max_vocab": self.max_vocab,
            "word_tokens": self.word_tokens,
            "char_tokens": self.char_tokens,
        }

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> LyricVocab:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            mode=d.get("mode", "hybrid"),
            max_vocab=d.get("max_vocab", 8000),
            word_tokens=d.get("word_tokens", []),
            char_tokens=d.get("char_tokens", []),
        )


# --------------------------------------------------------------------------- #
# The music tokenizer
# --------------------------------------------------------------------------- #
class MusicTokenizer:
    """Wraps miditok's REMI tokenizer and adds lyric tokens on top.

    The wrapper exists so the rest of the code base never has to know miditok
    API details, and so the backend could be swapped later without touching
    the model code.
    """

    def __init__(self, config: dict[str, Any] | None = None, lyrics: LyricVocab | None = None):
        self.config = config or {}
        self.lyrics = lyrics
        self._backend = None
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._base_size = 0
        self._unk_id = 0
        self._lyric_offset = 0

    # -- backend ----------------------------------------------------------- #
    def build_backend(self):
        """Create the miditok REMI tokenizer described by the config."""
        if self._backend is not None:
            return self._backend
        try:
            from miditok import REMI, TokenizerConfig
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise ImportError(
                "miditok is required for MIDI tokenization.\n"
                "Install it with:  pip install 'miditok>=3.0'"
            ) from exc

        beat_res = _parse_beat_res(self.config.get("beat_res") or {(0, 4): 4})
        tempo_range = tuple(self.config.get("tempo_range") or (40, 250))
        num_velocities = int(self.config.get("num_velocities", 32))
        try:
            cfg = TokenizerConfig(
                beat_res=beat_res,
                num_velocities=num_velocities,
                use_programs=bool(self.config.get("use_programs", True)),
                use_tempos=bool(self.config.get("use_tempos", True)),
                use_time_signatures=bool(self.config.get("use_time_signatures", True)),
                use_chords=bool(self.config.get("use_chords", False)),
                tempo_range=tempo_range,
            )
            self._backend = REMI(tokenizer_config=cfg)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("could not build a configured REMI tokenizer (%s); using defaults", exc)
            self._backend = REMI()
        LOG.info("built miditok REMI tokenizer (%d tokens)", self._backend.vocab_size)
        return self._backend

    # -- vocabulary -------------------------------------------------------- #
    def _ensure_vocab(self) -> None:
        """Build the id<->token maps on first use (they are lazy by design).

        Every public entry point that needs the vocabulary goes through this,
        so ``id_to_token`` works even if nothing ever touched the ``vocab``
        property first. (An earlier version only built the maps inside the
        properties, which made ``encode_midi`` -> ``id_to_token`` silently
        return ``<unk>`` for every id.)
        """
        if not self._token_to_id:
            self._build_vocabulary()

    @property
    def vocab(self) -> dict[str, int]:
        """Maps every token string (music + lyric + ``<unk>``) to its id."""
        self._ensure_vocab()
        return self._token_to_id

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including ``<unk>`` and lyric tokens."""
        self._ensure_vocab()
        return max(self._id_to_token) + 1

    @property
    def base_vocab_size(self) -> int:
        """Number of music tokens (the part miditok owns)."""
        self._ensure_vocab()
        return self._base_size

    @property
    def unk_id(self) -> int:
        self._ensure_vocab()
        return self._unk_id

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    def _build_vocabulary(self) -> None:
        backend = self.build_backend()
        vocab: dict[str, int] = dict(backend.vocab)

        self._base_size = backend.vocab_size
        self._unk_id = self._base_size
        vocab[UNK_TOKEN] = self._unk_id

        self._lyric_offset = self._base_size + 1
        if self.lyrics is not None:
            for i, tok in enumerate(self.lyrics.all_tokens()):
                vocab[tok] = self._lyric_offset + i

        self._token_to_id = vocab
        self._id_to_token = {i: t for t, i in vocab.items()}
        LOG.info(
            "vocabulary: %d music tokens, %d lyric tokens -> %d total",
            self._base_size,
            max(0, len(vocab) - self._base_size - 1),
            self.vocab_size,
        )

    def id_to_token(self, index: int) -> str:
        self._ensure_vocab()
        return self._id_to_token.get(index, UNK_TOKEN)

    def token_to_id(self, token: str) -> int:
        self._ensure_vocab()
        return self._token_to_id.get(token, self._unk_id)

    def decode_tokens(self, ids: list[int]) -> list[str]:
        return [self.id_to_token(i) for i in ids]

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id(t) for t in tokens]

    # -- MIDI <-> ids ------------------------------------------------------ #
    @staticmethod
    def _to_score(midi: Any):
        """Accept a path, a ``pretty_midi.PrettyMIDI`` or a ``symusic.Score``."""
        try:
            from symusic import Score
        except ImportError as exc:  # pragma: no cover
            raise ImportError("symusic is required; pip install 'symusic'") from exc

        if hasattr(midi, "tracks") and hasattr(midi, "note_num"):
            return midi
        if isinstance(midi, (str, Path)):
            return Score(str(midi), ttype="tick")
        # A pretty_midi object (or anything else with .write): dump to a temp
        # file and load it back. Cheap and robust.
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as fh:
            tmp = Path(fh.name)
        try:
            midi.write(str(tmp))
            return Score(str(tmp), ttype="tick")
        finally:
            tmp.unlink(missing_ok=True)

    def encode_midi(self, midi: Any, max_tokens: int | None = None) -> list[int]:
        """Encode a MIDI file/object into a list of token ids."""
        backend = self.build_backend()
        score = self._to_score(midi)
        try:
            score = backend.preprocess_score(score)
        except Exception:  # pragma: no cover - backend specific
            pass
        ids = list(backend.encode(score).ids)
        if max_tokens and len(ids) > max_tokens:
            ids = ids[:max_tokens]
        return ids

    def encode_midi_tokens(self, midi: Any, max_tokens: int | None = None) -> list[str]:
        """Encode a MIDI file/object into token *strings* (for inspection)."""
        return self.decode_tokens(self.encode_midi(midi, max_tokens=max_tokens))

    def decode_ids(self, ids: list[int]):
        """Rebuild a MIDI score from token ids (lyric tokens are stripped)."""
        backend = self.build_backend()
        music_ids = self.remove_lyric_ids(ids)
        music_ids = [i for i in music_ids if 0 <= i < self.base_vocab_size]
        if not music_ids:
            raise ValueError("no music tokens to decode")
        return backend.decode(music_ids)

    def save_midi(self, ids: list[int], path: Path | str) -> Path:
        """Decode ids and write a ``.mid`` file.

        Also sanitises the tempo: an undertrained model can emit an
        out-of-range ``Tempo`` token, and miditok decodes a 0 QPM event
        literally - symusic then reports ``score.end() == 0`` (no duration),
        which breaks rendering, evaluation and bar-aligned vocal mixing.
        Invalid tempos are replaced by the midpoint of the configured range.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        score = self.decode_ids(ids)
        self._sanitize_tempos(score)
        score.dump_midi(str(p))
        return p

    def _sanitize_tempos(self, score: Any) -> None:
        """Replace zero/negative/out-of-range tempo events in-place."""
        tempo_range = tuple(self.config.get("tempo_range") or (40, 250))
        lo, hi = float(tempo_range[0]), float(tempo_range[1])
        fallback = (lo + hi) / 2.0
        for tempo in getattr(score, "tempos", []):
            qpm = float(tempo.qpm)
            if not (lo <= qpm <= hi):
                tempo.qpm = fallback

    # -- lyrics ------------------------------------------------------------ #
    def _bar_positions(self, ids: list[int]) -> list[int]:
        self._ensure_vocab()
        bar_id = self._token_to_id.get("Bar_None")
        if bar_id is None:
            return []
        return [i for i, x in enumerate(ids) if x == bar_id]

    def interleave_lyrics(self, music_ids: list[int], lyric_tokens: list[str]) -> list[int]:
        """Insert lyric tokens into the music stream at bar boundaries.

        The lyric payload is split into one group per bar (as evenly as
        possible) and each group is inserted right after its bar token. Every
        group carries its own START/END markers: one pair spanning the whole
        stream would make ``remove_lyric_ids`` (and the MIDI decoder) swallow
        the music that sits *between* the first and last bar. If there are no
        bar tokens the lyrics go at the very beginning.
        """
        if not lyric_tokens or self.lyrics is None:
            return list(music_ids)

        start = self.token_to_id(START_TOKEN)
        end = self.token_to_id(END_TOKEN)
        tokens = [self.token_to_id(t) for t in lyric_tokens]

        bars = self._bar_positions(music_ids)
        if not bars:
            return [start, *tokens, end, *music_ids]

        n_bars = len(bars)
        per_bar = max(1, len(tokens) // n_bars)
        groups: list[list[int]] = [[] for _ in range(n_bars)]
        for i, tok in enumerate(tokens):
            groups[min(i // per_bar, n_bars - 1)].append(tok)

        out: list[int] = []
        group_idx = 0
        for i, tok in enumerate(music_ids):
            out.append(tok)
            if i in bars and group_idx < n_bars:
                group = groups[group_idx]
                if group:                       # empty bars stay silent
                    out.extend([start, *group, end])
                group_idx += 1
        return out

    def remove_lyric_ids(self, ids: list[int]) -> list[int]:
        """Strip lyric tokens, leaving pure music (used before MIDI decode)."""
        self._ensure_vocab()
        start = self._token_to_id.get(START_TOKEN)
        end = self._token_to_id.get(END_TOKEN)
        if start is None:
            return list(ids)
        out: list[int] = []
        inside = False
        for tok in ids:
            if tok == start:
                inside = True
                continue
            if tok == end:
                inside = False
                continue
            if not inside:
                out.append(tok)
        return out

    def extract_lyrics(self, ids: list[int]) -> str:
        """Recover the lyric text from an interleaved token stream."""
        if self.lyrics is None:
            return ""
        self._ensure_vocab()
        start = self._token_to_id.get(START_TOKEN)
        end = self._token_to_id.get(END_TOKEN)
        inside = False
        collected: list[str] = []
        for tok in ids:
            if start is not None and tok == start:
                inside = True
                continue
            if end is not None and tok == end:
                inside = False
                continue
            if inside:
                collected.append(self.id_to_token(tok))
        return self.lyrics.decode(collected)

    # -- persistence ------------------------------------------------------- #
    def save_vocabulary(self, path: Path | str) -> Path:
        """Persist the token->id map so a training run can be reproduced."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": "miditok.REMI",
            "base_vocab_size": self.base_vocab_size,
            "vocab_size": self.vocab_size,
            "unk_id": self.unk_id,
            "tokens": {t: i for t, i in self.vocab.items()},
        }
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return p

    def describe(self) -> dict[str, Any]:
        """A human-readable summary for the CLI and the docs."""
        vocab = self.vocab
        families = sorted({t.split("_")[0] for t in vocab if not t.startswith("<")})
        return {
            "backend": "miditok.REMI",
            "base_vocab_size": self.base_vocab_size,
            "vocab_size": self.vocab_size,
            "num_lyric_tokens": max(0, len(vocab) - self.base_vocab_size - 1),
            "families": families,
            "config": self.config,
        }


#: Process-wide cache keyed by the canonical JSON of the config. Building a
#: tokenizer means constructing a miditok backend, which is not free, and the
#: CLI, the dataset builder and the generator all build the *same* tokenizer
#: within one run. The key uses ``default=str`` so Path/exotic values are fine.
_TOKENIZER_CACHE: dict[str, MusicTokenizer] = {}


def build_tokenizer(config: dict[str, Any] | None = None) -> MusicTokenizer:
    """Factory used by the CLI, the dataset and the generator.

    Returns a cached instance when the same config was already used in this
    process (useful in tests and in commands that touch several stages).
    """
    cfg = dict(config or {})
    key = json.dumps(cfg, sort_keys=True, default=str)
    cached = _TOKENIZER_CACHE.get(key)
    if cached is not None:
        return cached

    lyrics_cfg = cfg.get("lyrics") or {}
    lyrics = None
    if lyrics_cfg.get("enabled"):
        lyrics = LyricVocab(
            mode=lyrics_cfg.get("mode", "hybrid"),
            max_vocab=int(lyrics_cfg.get("max_vocab", 8000)),
        )
    tok = MusicTokenizer(cfg, lyrics=lyrics)
    _TOKENIZER_CACHE[key] = tok
    return tok
