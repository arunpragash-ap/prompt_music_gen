"""Turning caption records + MIDI files into tensors the model trains on.

The pipeline, in words
----------------------
1. **Read the captions.** ``data/captions.py`` streams ``train.json`` and gives
   us one record per MIDI file: a description plus its audio-derived features.
2. **Find the MIDI file.** :func:`promptmusic.data.captions.resolve_midi_path`
   maps the record's virtual ``location`` onto a real file on disk.
3. **Tokenise the music.** The miditok REMI tokenizer turns the MIDI into a
   short list of integers (see ``data/tokenizer.py`` for why).
4. **Write down the prompt.** The caption, optionally joined with the feature
   fields, optionally translated into Tamil.
5. **Cache everything.** Steps 1-4 are expensive (minutes for 20k files), so we
   write the result to JSON Lines. Re-running the trainer then starts in
   seconds.

Why cache to JSON Lines and not to a binary format?
---------------------------------------------------
A ``.jsonl`` file is greppable, diffable and readable by any language. On a
laptop-scale dataset (20k examples) the file is ~40 MB and loads in a couple
of seconds, which is a good trade for the debuggability you get.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..paths import PROCESSED_DIR, ensure_dir
from ..utils import logger
from .captions import (
    CaptionRecord,
    build_prompt_from_features,
    load_caption_index,
    resolve_midi_path,
)

LOG = logger("promptmusic.dataset")

__all__ = [
    "TokenizedExample",
    "build_tokenized_dataset",
    "load_tokenized_dataset",
    "split_examples",
    "MusicDataset",
    "collate_fn",
    "EXAMPLE_CACHE_VERSION",
]

#: Bump this whenever the cache layout changes, so stale caches are rebuilt.
EXAMPLE_CACHE_VERSION = 2


# --------------------------------------------------------------------------- #
# One training example
# --------------------------------------------------------------------------- #
@dataclass
class TokenizedExample:
    """A single (prompt text, music token ids) pair, ready for training."""

    text: str
    ids: list[int]
    location: str = ""
    caption: str = ""
    caption_ta: str = ""
    lyrics: str = ""
    language: str = "en"
    duration: float = 0.0
    tempo: float = 0.0
    key: str = ""
    genre: str = ""
    mood: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TokenizedExample:
        known = {f for f in cls.__dataclass_fields__}
        extras = {k: v for k, v in d.items() if k not in known}
        kwargs = {k: v for k, v in d.items() if k in known}
        base = kwargs.setdefault("extras", {})
        base.update(extras)
        return cls(**kwargs)

    def __len__(self) -> int:
        return len(self.ids)


# --------------------------------------------------------------------------- #
# Building the cache
# --------------------------------------------------------------------------- #
def _pick_text(record: CaptionRecord, language: str, use_features: bool) -> tuple[str, str]:
    """Choose the conditioning text and the language tag for one record."""
    if language == "ta":
        if record.caption_ta:
            return record.caption_ta, "ta"
        LOG.debug("no Tamil caption for %s; falling back to English", record.location)
    text = build_prompt_from_features(record) if use_features else (record.caption or "")
    return text, "en"


def build_tokenized_dataset(
    config: Any,
    tokenizer: Any,
    *,
    language: str = "en",
    limit: int | None = None,
    use_features: bool = True,
    cache_path: Path | str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[TokenizedExample]:
    """Tokenise a slice of MidiCaps and return the resulting examples.

    Args:
        config: the project :class:`~promptmusic.config.Config`.
        tokenizer: a :class:`~promptmusic.data.tokenizer.MusicTokenizer`.
        language: ``"en"`` or ``"ta"`` - which caption to condition on.
        limit: cap on the number of records (``None`` = the config's value).
        use_features: fold the feature fields (genre, mood, key, tempo) into
            the prompt text as well as the free-form caption.
        cache_path: where to read/write the JSONL cache. ``None`` uses the
            default under ``data/processed/``.

    Returns:
        The examples, in the same order as the caption file.
    """
    cfg = config
    data_cfg = cfg.section("data")
    if limit is None:
        limit = data_cfg.get("limit")

    cache_path = Path(cache_path) if cache_path else _default_cache_path(cfg, language, limit)
    if cache_path.exists():
        LOG.info("loading tokenised cache %s", cache_path)
        examples = load_tokenized_dataset(cache_path)
        if limit and len(examples) > limit:
            examples = examples[:limit]
        return examples

    caption_file = _find_caption_file(cfg)
    midi_root = _find_midi_root(cfg)
    LOG.info("tokenising captions from %s (language=%s, limit=%s)", caption_file, language, limit)

    records = load_caption_index(caption_file, limit=limit)
    max_tokens = int(cfg.get("tokenizer.max_tokens", 4096))

    examples: list[TokenizedExample] = []
    skipped = 0
    total = len(records)
    for i, record in enumerate(records):
        midi_path = resolve_midi_path(record.location, midi_root)
        if midi_path is None:
            skipped += 1
            continue
        try:
            ids = tokenizer.encode_midi(midi_path, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 - one corrupt file must not stop the run
            LOG.debug("could not tokenise %s: %s", midi_path, exc)
            skipped += 1
            continue
        if len(ids) < 8:
            skipped += 1
            continue

        text, lang = _pick_text(record, language, use_features)
        if not text:
            skipped += 1
            continue

        examples.append(
            TokenizedExample(
                text=text,
                ids=ids,
                location=record.location,
                caption=record.caption,
                caption_ta=record.caption_ta,
                lyrics=record.lyrics,
                language=lang,
                duration=record.duration,
                tempo=record.tempo,
                key=record.key,
                genre=record.genre,
                mood=record.mood,
            )
        )
        if progress:
            progress(i + 1, total)
        elif (i + 1) % 500 == 0:
            LOG.info("  tokenised %d/%d (skipped %d)", i + 1, total, skipped)

    LOG.info("built %d examples (skipped %d)", len(examples), skipped)
    ensure_dir(cache_path.parent)
    _write_cache(examples, cache_path)
    LOG.info("cached to %s", cache_path)
    return examples


def _write_cache(examples: list[TokenizedExample], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def load_tokenized_dataset(path: Path | str) -> list[TokenizedExample]:
    """Read a JSONL example cache."""
    out: list[TokenizedExample] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(TokenizedExample.from_dict(json.loads(line)))
    return out


def _default_cache_path(cfg: Any, language: str, limit: int | None) -> Path:
    # An explicit cache file (set by `tokenize --cache` or in the config)
    # wins: train() calls this without a cache_path, and it must find the
    # same file tokenize wrote or it would silently re-tokenise 400 MIDI
    # files (minutes of work) for nothing.
    explicit = cfg.get("data.cache_file")
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p
    root = Path(cfg.get("paths.processed_dir", PROCESSED_DIR))
    tag = f"{language}_{limit if limit else 'all'}"
    return root / f"examples_{tag}.jsonl"


def _find_caption_file(cfg: Any) -> Path:
    data_dir = Path(cfg.get("paths.data_dir", "dataset"))
    name = cfg.get("data.caption_file", "train.json")
    candidates = [
        data_dir / name,
        data_dir / "MidiCaps" / name,
        data_dir.parent / "data" / "downloads" / name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"caption file {name!r} not found. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nRun `promptmusic download --source midicaps` first."
    )


def _find_midi_root(cfg: Any) -> Path:
    data_dir = Path(cfg.get("paths.data_dir", "dataset"))
    sub = cfg.get("data.midi_subdir", "midicaps")
    candidates = [data_dir / sub, data_dir, data_dir / "MIDI", data_dir / "midi"]
    for c in candidates:
        if c.is_dir():
            return c
    return data_dir


# --------------------------------------------------------------------------- #
# Train / validation split
# --------------------------------------------------------------------------- #
def split_examples(
    examples: list[TokenizedExample],
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[TokenizedExample], list[TokenizedExample]]:
    """Deterministic shuffle + split.

    Seeded so that two runs of the trainer see the same validation set, which
    is what makes a comparison between two checkpoints meaningful.
    """
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(examples) * val_fraction)) if val_fraction > 0 else 0
    val_idx = set(idx[:n_val])
    train = [examples[i] for i in range(len(examples)) if i not in val_idx]
    val = [examples[i] for i in range(len(examples)) if i in val_idx]
    LOG.info("split: %d train / %d val (val_fraction=%.3f, seed=%d)", len(train), len(val), val_fraction, seed)
    return train, val


# --------------------------------------------------------------------------- #
# PyTorch dataset + collation
# --------------------------------------------------------------------------- #
class MusicDataset(Dataset):
    """Wraps :class:`TokenizedExample` objects for a DataLoader.

    Returns a dict::

        {"ids": LongTensor [T], "text": str, "index": int}

    Text tokenisation happens later (in the training loop) so that the large
    mT5 tokenizer is only loaded once per process, not once per worker.
    """

    def __init__(self, examples: list[TokenizedExample], max_tokens: int | None = None):
        self.examples = examples
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ex = self.examples[index]
        ids = ex.ids
        if self.max_tokens and len(ids) > self.max_tokens:
            ids = ids[: self.max_tokens]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "text": ex.text,
            "index": index,
        }


def collate_fn(
    batch: list[dict[str, Any]],
    *,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 0,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Pad the music ids and add the boundary tokens.

    Each sequence becomes ``[BOS] + ids + [EOS]``, so the model learns both
    where a piece starts and where it ends. Everything is padded to the
    longest sequence in the batch with ``pad_id``; the padding positions are
    masked out of the loss.
    """
    seqs = []
    for item in batch:
        ids = [bos_id, *item["ids"].tolist(), eos_id]
        if max_tokens and len(ids) > max_tokens:
            ids = ids[:max_tokens]
            ids[-1] = eos_id
        seqs.append(ids)

    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = True

    return {
        "music_ids": padded,
        "music_mask": mask,
        "texts": [item["text"] for item in batch],
        "indices": [item["index"] for item in batch],
    }


def iter_batches(dataset: MusicDataset, batch_size: int, **kwargs) -> Iterator[dict[str, Any]]:
    """Yield collated batches - handy for tests and quick scripts."""
    batch: list[dict[str, Any]] = []
    for i in range(len(dataset)):
        batch.append(dataset[i])
        if len(batch) == batch_size:
            yield collate_fn(batch, **kwargs)
            batch = []
    if batch:
        yield collate_fn(batch, **kwargs)
