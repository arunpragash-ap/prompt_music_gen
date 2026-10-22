"""Generating music from a text prompt.

The generation pipeline, end to end
-----------------------------------
::

    "a cheerful tamil folk song with flute and tabla"
                    │
                    ▼
        mT5 SentencePiece tokenizer          (text -> ids)
                    │
                    ▼
        frozen mT5 encoder                   (ids -> memory [B, T, 768])
                    │
                    ▼
        trained decoder, step by step        (memory -> music token ids)
                    │
                    ▼
        miditok REMI decoder                 (ids -> MIDI file)
                    │
                    ▼
        fluidsynth + soundfont               (MIDI -> WAV file)

Sampling controls
-----------------
* ``temperature`` - how adventurous. 0 = greedy (always the most likely
  token), 1.0 = normal sampling, 1.3 = wild.
* ``top_k`` - only sample from the k most likely tokens. 50 is a good default;
  0 disables the filter.
* ``top_p`` (nucleus) - only sample from the smallest set of tokens whose
  combined probability exceeds ``p``. 0.95 keeps the "serious" options.
* ``repetition_penalty`` - discourages repeating the same token. Above 1.0
  curbs the loops a small model can fall into.
* ``max_new_tokens`` - hard cap on how long the piece is.

Two knobs are usually enough: ``temperature`` and ``top_p``. Start at
``temperature=1.0, top_p=0.95`` and move temperature if the result is too
safe (up) or too chaotic (down).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .config import Config, load_config
from .data.tokenizer import build_tokenizer
from .model.build import load_checkpoint, build_model
from .model.text_encoder import TextTokenizer
from .paths import OUTPUTS_DIR, ensure_dir
from .render import render_midi
from .utils import format_duration, logger, select_device, set_seed

LOG = logger("promptmusic.generate")

__all__ = ["GenerationResult", "generate", "generate_from_checkpoint"]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class GenerationResult:
    """Everything one call to :func:`generate` produced."""

    prompt: str
    language: str
    token_ids: list[int] = field(default_factory=list)
    lyrics: str = ""
    midi_path: str | None = None
    wav_path: str | None = None
    render_info: dict[str, Any] | None = None
    num_tokens: int = 0
    num_notes: int = 0
    duration_sec: float = 0.0
    elapsed_sec: float = 0.0
    seed: int | None = None
    checkpoint: str | None = None
    sampling: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loading a trained model for inference
# --------------------------------------------------------------------------- #
def _load_for_inference(
    checkpoint: str | Path,
    config: Config | None = None,
    device: str = "auto",
) -> tuple[Any, Any, TextTokenizer, Config, dict[str, Any]]:
    """Load a checkpoint and rebuild everything it needs.

    A checkpoint carries the config and the token vocabulary that produced it,
    so a single ``.pt`` file is enough to regenerate music - see
    ``docs/06_DEPLOYMENT.md``.
    """
    device = select_device(device)
    payload = load_checkpoint(checkpoint, map_location=device)

    if config is None:
        stored = payload.get("config")
        cfg = Config(stored) if stored else load_config("default")
    else:
        cfg = config if isinstance(config, Config) else load_config(config)

    tok_cfg = cfg.section("tokenizer")
    music_tokenizer = build_tokenizer(tok_cfg)

    # The vocab stored in the checkpoint is authoritative when its size matches.
    stored_vocab = payload.get("vocab")
    if stored_vocab and len(stored_vocab) == music_tokenizer.vocab_size:
        LOG.info("using token vocabulary from the checkpoint")
    elif stored_vocab:
        LOG.warning(
            "checkpoint vocabulary (%d) differs from the rebuilt one (%d); "
            "token ids may not line up. Re-tokenise with the same config.",
            len(stored_vocab), music_tokenizer.vocab_size,
        )

    model = build_model(
        cfg,
        vocab_size=music_tokenizer.vocab_size,
        pad_id=0, bos_id=1, eos_id=2,
        device=device,
    )
    missing, unexpected = model.decoder.load_state_dict(payload["model_state"], strict=False)
    if missing:
        LOG.warning("%d decoder tensors were missing from the checkpoint", len(missing))
    if unexpected:
        LOG.warning("%d checkpoint tensors were not used by the decoder", len(unexpected))
    model.eval()

    text_tok = TextTokenizer(
        model_name=cfg.get("model.text_encoder.name", "google/mt5-base"),
        max_length=int(cfg.get("model.text_encoder.max_length", 128)),
    )
    return model, music_tokenizer, text_tok, cfg, payload


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(
    prompt: str,
    *,
    checkpoint: str | Path | None = None,
    config: Config | str | Path | None = None,
    out_dir: str | Path | None = None,
    # Sampling ------------------------------------------------------------- #
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    num_beams: int | None = None,
    repetition_penalty: float | None = None,
    num_samples: int = 1,
    # Output --------------------------------------------------------------- #
    render_audio: bool = True,
    soundfont: str | Path | None = None,
    sample_rate: int | None = None,
    # Reference conditioning (Phase 4) ------------------------------------- #
    reference_midi: str | Path | None = None,
    reference_audio: str | Path | None = None,
    # Misc ------------------------------------------------------------------ #
    seed: int | None = None,
    device: str = "auto",
    language: str = "en",
    lyrics: str = "",
) -> list[GenerationResult]:
    """Generate music for ``prompt`` and return one result per sample.

    Args:
        prompt: the description of the music you want. English and Tamil both
            work (Tamil needs a model trained with ``data.languages: [en, ta]``).
        checkpoint: path to a trained ``.pt`` file. Required.
        config: optional config override; by default the one stored inside the
            checkpoint is used.
        out_dir: where to write MIDI/WAV files. Defaults to
            ``outputs/<project>/samples/``.
        reference_midi: (Phase 4) a MIDI file whose opening bars prime the
            generation.
        reference_audio: (Phase 4) an audio file whose tempo/key/energy are
            analysed and folded into the prompt.
        seed: set for reproducible output. ``None`` means random.
        language: ``en`` or ``ta``; names the lyrics language in the prompt.
        lyrics: sung text to condition on (English or Tamil). Folded into the
            text prompt read by mT5; token-level lyric interleaving arrives
            with Phase 3 training.

    Returns:
        A list of :class:`GenerationResult`, one per generated sample.
    """
    if checkpoint is None:
        raise ValueError("a trained checkpoint is required, e.g. checkpoint='outputs/default/checkpoints/best.pt'")

    if not isinstance(config, Config):
        cfg = load_config(config) if config else None
    else:
        cfg = config

    model, music_tokenizer, text_tok, cfg, payload = _load_for_inference(checkpoint, cfg, device)
    device = select_device(device)

    gcfg = cfg.section("generate")
    max_new_tokens = int(max_new_tokens or gcfg.get("max_new_tokens", 1024))
    temperature = float(gcfg.get("temperature", 1.0) if temperature is None else temperature)
    top_k = int(gcfg.get("top_k", 50) if top_k is None else top_k)
    top_p = float(gcfg.get("top_p", 0.95) if top_p is None else top_p)
    num_beams = int(num_beams or gcfg.get("num_beams", 1))
    repetition_penalty = float(
        repetition_penalty if repetition_penalty is not None else gcfg.get("repetition_penalty", 1.0)
    )
    sample_rate = int(sample_rate or cfg.get("render.sample_rate", 44100))
    seed = int(seed if seed is not None else gcfg.get("seed", 42))
    set_seed(seed)

    out_dir = ensure_dir(Path(out_dir) if out_dir else Path(cfg.get("paths.output_dir", OUTPUTS_DIR)) / "samples")

    # --- reference conditioning ------------------------------------------- #
    conditioning = _build_conditioning(prompt, reference_midi, reference_audio, cfg)
    final_prompt = conditioning["prompt"]

    # --- lyrics conditioning ---------------------------------------------- #
    # Lyric *tokens* interleaved at bar boundaries is a training-time feature
    # (Phase 3) and needs a model trained that way. Until then, supplied
    # lyrics influence generation the only way that works today: they become
    # part of the text the frozen mT5 encoder reads. mT5 handles Tamil and
    # English natively, so bilingual lyrics condition the decoder directly.
    lyrics = (lyrics or "").strip()
    if lyrics:
        lang_name = {"ta": "Tamil", "en": "English"}.get(language, language)
        final_prompt = f"{final_prompt}\n{lang_name} lyrics: {lyrics}"

    # --- encode the prompt -------------------------------------------------- #
    enc = text_tok(final_prompt)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    memory = model.encode_prompt(input_ids, attention_mask)

    sampling = dict(
        temperature=temperature, top_k=top_k, top_p=top_p, num_beams=num_beams,
        repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
    )

    results: list[GenerationResult] = []
    t0 = time.time()
    for i in range(num_samples):
        sample_seed = seed + i
        set_seed(sample_seed)
        ids = model.generate(
            memory,
            memory_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )[0].tolist()

        # Strip the BOS and anything from the first EOS onwards.
        ids = _strip_boundaries(ids, music_tokenizer)

        result = GenerationResult(
            prompt=prompt,
            language=language,
            token_ids=ids,
            lyrics=music_tokenizer.extract_lyrics(ids) if lyrics else "",
            num_tokens=len(ids),
            seed=sample_seed,
            checkpoint=str(checkpoint),
            sampling=sampling,
        )

        stem = _safe_stem(prompt, i)
        try:
            midi_path = music_tokenizer.save_midi(ids, out_dir / f"{stem}.mid")
            result.midi_path = str(midi_path)
            result.num_notes, result.duration_sec = _midi_stats(midi_path)
        except Exception as exc:  # noqa: BLE001 - a bad token stream must not kill the batch
            LOG.warning("could not decode sample %d into MIDI: %s", i, exc)

        if render_audio and result.midi_path:
            try:
                r = render_midi(
                    result.midi_path,
                    out_dir / f"{stem}.wav",
                    backend=cfg.get("render.backend", "auto"),
                    soundfont=soundfont,
                    sample_rate=sample_rate,
                    gain=float(cfg.get("render.gain", 0.8)),
                )
                result.wav_path = str(r.wav_path)
                result.render_info = r.to_dict()
            except Exception as exc:  # noqa: BLE001
                LOG.warning("rendering failed for sample %d: %s", i, exc)

        results.append(result)
        LOG.info(
            "sample %d/%d: %d tokens, %d notes, %.1fs, %s",
            i + 1, num_samples, result.num_tokens, result.num_notes,
            result.duration_sec, result.midi_path or "(no MIDI)",
        )

    elapsed = time.time() - t0
    LOG.info("generated %d sample(s) in %s", num_samples, format_duration(elapsed))
    for r in results:
        r.elapsed_sec = elapsed / max(num_samples, 1)
    return results


# --------------------------------------------------------------------------- #
# Reference conditioning
# --------------------------------------------------------------------------- #
def _build_conditioning(
    prompt: str,
    reference_midi: str | Path | None,
    reference_audio: str | Path | None,
    cfg: Config,
) -> dict[str, Any]:
    """Fold reference-file analysis into the conditioning prompt (Phase 4)."""
    notes: list[str] = []
    analysis: dict[str, Any] = {}

    if reference_midi:
        try:
            from .analyze import analyze_midi

            analysis = analyze_midi(reference_midi)
            notes.append(_describe_analysis(analysis, "reference piece"))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("could not analyse reference MIDI %s: %s", reference_midi, exc)

    if reference_audio and cfg.get("reference.analyze_audio", True):
        try:
            from .analyze import analyze_audio

            a = analyze_audio(reference_audio)
            analysis.update({f"audio_{k}": v for k, v in a.items()})
            notes.append(_describe_analysis(a, "reference audio"))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("could not analyse reference audio %s: %s", reference_audio, exc)

    combined = prompt.strip()
    if notes:
        combined = f"{combined} ({'; '.join(notes)})"
    return {"prompt": combined, "analysis": analysis}


def _describe_analysis(analysis: dict[str, Any], label: str) -> str:
    bits: list[str] = []
    if analysis.get("tempo"):
        bits.append(f"tempo {float(analysis['tempo']):.0f} bpm")
    if analysis.get("key"):
        bits.append(f"key {analysis['key']}")
    if analysis.get("duration_sec"):
        bits.append(f"{float(analysis['duration_sec']):.0f}s")
    if analysis.get("instruments"):
        names = ", ".join(str(x) for x in analysis["instruments"][:4])
        bits.append(f"instruments: {names}")
    return f"{label} at " + ", ".join(bits) if bits else label


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _strip_boundaries(ids: list[int], music_tokenizer) -> list[int]:
    """Remove BOS/PAD at the front and everything from the first EOS."""
    eos, bos, pad = 2, 1, 0
    out: list[int] = []
    for tok in ids:
        if tok == eos:
            break
        if tok in (bos, pad):
            continue
        out.append(tok)
    return out


def _safe_stem(prompt: str, index: int) -> str:
    """A filesystem-friendly name derived from the prompt."""
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.lower()).strip("_")[:48] or "sample"
    return f"{slug}_{index}"


def _midi_stats(path: Path | str) -> tuple[int, float]:
    """(note count, duration seconds) for a MIDI file, or (0, 0.0)."""
    try:
        from symusic import Score

        score = Score(str(path), ttype="second")
        return int(score.note_num()), float(score.end - score.start)
    except Exception:
        try:
            import pretty_midi

            pm = pretty_midi.PrettyMIDI(str(path))
            return sum(len(i.notes) for i in pm.instruments), float(pm.get_end_time())
        except Exception:
            return 0, 0.0


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #
def generate_from_checkpoint(
    checkpoint: str | Path,
    prompts: list[str],
    **kwargs: Any,
) -> list[list[GenerationResult]]:
    """Generate for several prompts with one loaded checkpoint.

    Loading the model (and the ~300 M-parameter text encoder) dominates the
    cost, so batching prompts this way is much faster than calling
    :func:`generate` repeatedly.
    """
    out: list[list[GenerationResult]] = []
    for i, prompt in enumerate(prompts):
        LOG.info("prompt %d/%d: %s", i + 1, len(prompts), prompt)
        out.append(generate(prompt, checkpoint=checkpoint, **kwargs))
    return out
