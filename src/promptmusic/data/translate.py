"""Phase 2: translating English captions into Tamil.

Why translate instead of finding Tamil music directly?
------------------------------------------------------
MidiCaps captions are English, and the underlying Lakh MIDI corpus is mostly
Western music. A genuinely Tamil-labelled MIDI dataset of comparable size does
not exist publicly. Translation gives us a **parallel corpus**: the same piece
of music described in both languages, which is exactly what teaches the model
that "a joyful folk song" and "மகிழ்ச்சியான நாட்டுப்புற பாடல்" mean the same thing.

That is the mechanism that makes Tamil prompting work: the music side is
identical, only the text changes, so the encoder is forced to align the two.

Which translator
----------------
``ai4bharat/indictrans2-en-indic-dist-200M`` (MIT licence). Chosen over
``facebook/nllb-200-distilled-600M`` because NLLB is **CC-BY-NC-4.0** - non
commercial - and this project aims for a permissive licence stack.

Translation quality and what to expect
--------------------------------------
IndicTrans2 produces fluent, idiomatic Tamil for short descriptive sentences
like MidiCaps captions. It is not perfect: musical terms sometimes come back
as their everyday meanings ("bright" can become "விளக்கு"-adjacent). Because we
translate the *caption* and keep the numeric features (tempo, key) in the
prompt as numerals, the conditioning signal survives even when a word is
translated loosely.

Scale strategy
--------------
Translating all 168k captions takes hours. The default is to translate a
configurable subset (``data.translation.limit``, default 20 000) and train on
that. Raise the limit and re-run ``promptmusic translate`` to scale up - the
step is incremental and caches every translation it makes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..paths import CACHE_DIR, ensure_dir
from ..utils import logger
from .captions import CaptionRecord

LOG = logger("promptmusic.translate")

__all__ = [
    "TranslationEngine",
    "translate_records",
    "load_translation_cache",
    "TARGET_LANGUAGE",
    "TARGET_LANGUAGE_CODE",
]

#: Human-readable target language and its ISO-ish code used in filenames.
TARGET_LANGUAGE = "Tamil"
TARGET_LANGUAGE_CODE = "ta"


class TranslationEngine:
    """English -> Tamil translation with IndicTrans2.

    Loading the model takes ~30 s and about 1 GB of RAM, so the engine is
    meant to be created once and reused across a whole batch.

    Args:
        model_name: Hugging Face id of an IndicTrans2 English->Indic model.
        device: ``"auto"``, ``"cpu"``, ``"mps"`` or ``"cuda"``.
        batch_size: sentences per forward pass.
        max_new_tokens: generation cap; captions are one sentence, so 128 is
            plenty.
    """

    def __init__(
        self,
        model_name: str = "ai4bharat/indictrans2-en-indic-dist-200M",
        *,
        device: str = "auto",
        batch_size: int = 16,
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ):
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)
        self.device = _resolve_device(device)
        self._tokenizer = None
        self._model = None
        self._toolkit_tokenizer = None

    # -- lazy loading ------------------------------------------------------ #
    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for translation. Install with:\n"
                "    pip install 'transformers>=4.44,<5'"
            ) from exc

        LOG.info("loading translator %s onto %s (this can take a minute) ...",
                 self.model_name, self.device)
        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(self.device)
        self._model.eval()

        # IndicTrans2 expects a small amount of language-tag preprocessing.
        # The official toolkit does it; if it is installed we use it, and
        # otherwise fall back to a manual prefix that works with the raw
        # transformers tokenizer.
        try:
            from IndicTransToolkit import IndicTransTokenizer  # type: ignore

            self._toolkit_tokenizer = IndicTransTokenizer(self.model_name)
            LOG.info("using IndicTransToolkit preprocessing")
        except Exception as exc:  # noqa: BLE001 - optional dependency
            LOG.info("IndicTransToolkit not installed (%s); using plain tokenization", exc)
            self._toolkit_tokenizer = None

        LOG.info("translator ready in %.1fs", time.time() - t0)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- translating ------------------------------------------------------- #
    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a list of English strings into Tamil."""
        if not texts:
            return []
        self.load()
        import torch

        out: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            cleaned = [t.strip() for t in chunk]

            if self._toolkit_tokenizer is not None:
                batch = self._toolkit_tokenizer(cleaned, tgt_lang=TARGET_LANGUAGE_CODE)
            else:
                # Plain-transformers fallback. IndicTrans2 was trained with the
                # source language tag prepended, so we reproduce that here.
                prefixed = [f"[Indic22] en_{TARGET_LANGUAGE_CODE} {t}" for t in cleaned]
                batch = self._tokenizer(
                    prefixed,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )

            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                generated = self._model.generate(
                    **batch,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )

            if self._toolkit_tokenizer is not None:
                decoded = self._toolkit_tokenizer.batch_decode(generated, skip_special_tokens=True)
            else:
                decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            out.extend(d.strip() for d in decoded)

            done = min(start + self.batch_size, len(texts))
            LOG.info("  translated %d/%d", done, len(texts))
        return out

    def translate(self, text: str) -> str:
        return self.translate_batch([text])[0]


def _resolve_device(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # IndicTrans2 occasionally hits ops MPS does not implement, so CPU is
        # the safe default on a Mac. Override with device="mps" to try anyway.
        return "cpu"
    return "cpu"


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: Path | str | None) -> Path:
    root = Path(cache_dir) if cache_dir else CACHE_DIR
    return ensure_dir(root) / f"captions_{TARGET_LANGUAGE_CODE}.jsonl"


def load_translation_cache(path: Path | str) -> dict[str, str]:
    """Read ``{english caption: tamil caption}`` from a JSONL cache."""
    p = Path(path)
    if not p.exists():
        return {}
    cache: dict[str, str] = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("en") and row.get("ta"):
                cache[row["en"]] = row["ta"]
    return cache


def _append_cache(path: Path, english: str, tamil: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"en": english, "ta": tamil}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Translating a caption index in place
# --------------------------------------------------------------------------- #
def translate_records(
    records: list[CaptionRecord],
    *,
    engine: TranslationEngine | None = None,
    limit: int | None = None,
    cache_dir: Path | str | None = None,
    out_path: Path | str | None = None,
) -> list[CaptionRecord]:
    """Fill in ``record.caption_ta`` for as many records as requested.

    Translation is **incremental and cached**: every caption translated is
    appended to a JSONL file, so interrupting the run and restarting it picks
    up where it left off rather than starting again.

    Args:
        records: the caption records to translate (modified in place).
        engine: an existing :class:`TranslationEngine`; one is created if
            ``None``.
        limit: translate at most this many *new* captions.
        cache_dir: where to keep the translation cache.
        out_path: optional JSONL to write the full mapping to.

    Returns:
        The same list of records, with ``caption_ta`` populated where possible.
    """
    cache_file = Path(out_path) if out_path else _cache_path(cache_dir)
    cache = load_translation_cache(cache_file)
    if cache:
        LOG.info("translation cache: %d captions already done (%s)", len(cache), cache_file)

    # 1. Apply the cache to every record we can, immediately.
    todo: list[CaptionRecord] = []
    for record in records:
        if record.caption_ta:
            continue
        if record.caption and record.caption in cache:
            record.caption_ta = cache[record.caption]
        elif record.caption:
            todo.append(record)

    if limit is not None:
        todo = todo[: int(limit)]

    LOG.info("%d records need translation (%d capped by limit=%s)",
             len(todo), len(todo), limit)
    if not todo:
        return records

    if engine is None:
        engine = TranslationEngine()
    engine.load()

    ensure_dir(cache_file.parent)
    for start in range(0, len(todo), engine.batch_size * 4):
        chunk = todo[start : start + engine.batch_size * 4]
        english = [r.caption for r in chunk]
        tamil = engine.translate_batch(english)
        for record, ta in zip(chunk, tamil):
            record.caption_ta = ta
            _append_cache(cache_file, record.caption, ta)
        LOG.info("  cached %d/%d translations", min(start + engine.batch_size * 4, len(todo)), len(todo))

    done = sum(1 for r in records if r.caption_ta)
    LOG.info("Tamil captions ready for %d/%d records", done, len(records))
    return records


def save_translations(records: list[CaptionRecord], path: Path | str) -> Path:
    """Write ``{location, caption, caption_ta}`` for every translated record."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in records:
            if not r.caption_ta:
                continue
            fh.write(json.dumps(
                {"location": r.location, "caption": r.caption, "caption_ta": r.caption_ta},
                ensure_ascii=False,
            ) + "\n")
    return p
