"""Assembling the full model and handling its checkpoints.

``MusicModel`` is the whole system::

    prompt text  ──►  frozen mT5 encoder  ──►  memory [B, T_text, 768]
                                                     │
                                                     ▼  (linear projection)
    music tokens ──►  embedding + position  ──►  decoder layers  ──►  logits [B, T, V]

Checkpoints carry everything needed to reproduce a run: the model weights, the
config that built them, the token vocabulary, and a short provenance record.
That is what makes the "transfer the model" story in ``docs/06_DEPLOYMENT.md``
work - a checkpoint plus this package is a complete, runnable artefact.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import logger
from .decoder import CausalCache, MusicDecoder
from .text_encoder import TEXT_ENCODER_DIM, TextEncoder

LOG = logger("promptmusic.model")

__all__ = ["MusicModel", "build_model", "save_checkpoint", "load_checkpoint", "count_parameters"]


class MusicModel(nn.Module):
    """Frozen text encoder + trainable causal music decoder.

    The text encoder is *not* part of the optimiser's parameter set when
    ``freeze=True``; it contributes only its forward output.
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        decoder: MusicDecoder,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        max_text_length: int = 128,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_text_length = max_text_length

    # -- encoding the prompt ---------------------------------------------- #
    def encode_prompt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run the frozen text encoder over tokenised prompts."""
        return self.text_encoder(input_ids, attention_mask)

    # -- training forward --------------------------------------------------- #
    def forward(
        self,
        music_ids: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass. Returns logits ``[B, T, V]``."""
        logits, _ = self.decoder(music_ids, memory, memory_mask=memory_mask)
        return logits

    def loss(
        self,
        music_ids: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        ignore_index: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Next-token cross-entropy.

        ``music_ids`` is the full sequence (``[BOS, t1, ..., tN, EOS]``). The
        decoder sees everything except the last token; the target is
        everything except the first. Returns ``(loss, logits)``.
        """
        ignore_index = self.pad_id if ignore_index is None else ignore_index
        dec_in = music_ids[:, :-1]
        targets = music_ids[:, 1:]
        logits = self.forward(dec_in, memory, memory_mask=memory_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )
        return loss, logits

    # -- generation --------------------------------------------------------- #
    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample token ids conditioned on ``memory``.

        Supports plain sampling, top-k, nucleus (top-p), a repetition penalty
        and simple beam search. KV caching is used throughout.
        """
        eos_id = self.eos_id if eos_id is None else eos_id
        batch = memory.size(0)
        device = memory.device

        # Never run past the decoder's positional table: the positional slice
        # pe[offset:offset+n] would come back short and the add would fail
        # with an obscure shape error deep inside a long generation.
        max_len = self.decoder.max_len
        if max_new_tokens > max_len:
            LOG.warning(
                "max_new_tokens=%d exceeds decoder max_len=%d; clamping "
                "(raise model.decoder.max_len for longer pieces)",
                max_new_tokens, max_len,
            )
        max_new_tokens = max(2, min(max_new_tokens, max_len))

        generated = torch.full((batch, 1), self.bos_id, dtype=torch.long, device=device)
        cache: list | None = None
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        # First step consumes only the BOS token.
        logits, cache = self.decoder(generated, memory, memory_mask=memory_mask, need_cache=True)

        for _ in range(max_new_tokens - 1):
            step_logits = logits[:, -1, :].float()

            if repetition_penalty > 1.0:
                step_logits = _apply_repetition_penalty(step_logits, generated, repetition_penalty)

            next_tokens = _sample(
                step_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_beams=num_beams,
            )

            if batch == 1 and finished[0]:
                break
            next_tokens = torch.where(finished.unsqueeze(1), torch.full_like(next_tokens, self.pad_id), next_tokens)
            generated = torch.cat([generated, next_tokens], dim=1)

            finished = finished | (next_tokens.squeeze(1) == eos_id)
            if bool(finished.all()):
                break

            logits, cache = self.decoder(
                next_tokens, memory, memory_mask=memory_mask, cache=cache, need_cache=True
            )

        return generated

    # -- utilities ---------------------------------------------------------- #
    def trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param

    def count_parameters(self, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)

    def extra_repr(self) -> str:
        return f"text_encoder={self.text_encoder.model_name!r}, decoder={self.decoder!r}"


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #
def _apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Down-weight tokens that already appeared (CTRL-style)."""
    for row in range(generated.size(0)):
        seen = torch.unique(generated[row])
        seen = seen[seen != 0]  # never penalise padding
        if seen.numel():
            vals = logits[row, seen]
            logits[row, seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def _sample(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    num_beams: int = 1,
) -> torch.Tensor:
    """Sample (or argmax for beams) the next token from raw logits."""
    if num_beams > 1:
        # Cheap beam search: pick the top-``num_beams`` candidates greedily
        # per row and keep the highest-scoring one. Full beam search over a
        # whole sequence needs a score accumulator; for a single step this is
        # the greedy beam head.
        k = min(num_beams, logits.size(-1))
        return torch.topk(logits, k, dim=-1).indices[:, :1]

    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-6)

    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        probs = F.softmax(sorted_logits, dim=-1)
        choice = torch.multinomial(probs, num=1)
        return sorted_idx.gather(-1, choice)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num=1)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(
    config: Any,
    vocab_size: int,
    *,
    pad_id: int = 0,
    bos_id: int = 1,
    eos_id: int = 2,
    device: str | None = None,
) -> MusicModel:
    """Build a :class:`MusicModel` from a :class:`~promptmusic.config.Config`."""
    enc_cfg = config.section("model.text_encoder")
    dec_cfg = config.section("model.decoder")

    text_encoder = TextEncoder(
        model_name=enc_cfg.get("name", "google/mt5-base"),
        freeze=bool(enc_cfg.get("freeze", True)),
        max_length=int(enc_cfg.get("max_length", 128)),
        dtype=enc_cfg.get("dtype", "float32"),
        device=device,
    )

    decoder = MusicDecoder(
        vocab_size=vocab_size,
        d_model=int(dec_cfg.get("d_model", 512)),
        nhead=int(dec_cfg.get("nhead", 8)),
        num_layers=int(dec_cfg.get("num_layers", 12)),
        dim_feedforward=int(dec_cfg.get("dim_feedforward", 2048)),
        dropout=float(dec_cfg.get("dropout", 0.1)),
        max_len=int(dec_cfg.get("max_len", 4096)),
        text_dim=text_encoder.output_dim,
        norm_first=bool(dec_cfg.get("norm_first", True)),
        activation=dec_cfg.get("activation", "gelu"),
        pad_id=pad_id,
        tie_embeddings=bool(dec_cfg.get("tied_embeddings", False)),
    )

    model = MusicModel(
        text_encoder,
        decoder,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        max_text_length=int(enc_cfg.get("max_length", 128)),
    )
    if device:
        model.to(device)
    LOG.info("built model: %s", model)
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: Path | str,
    model: MusicModel,
    *,
    optimizer: Any = None,
    scheduler: Any = None,
    epoch: int = 0,
    step: int = 0,
    metrics: dict | None = None,
    config: Any = None,
    vocab: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a self-contained checkpoint.

    The file contains the weights *and* everything needed to reload them, so a
    single ``.pt`` file is a portable artefact (see ``docs/06_DEPLOYMENT.md``).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_state": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "num_parameters": count_parameters(model),
        "model_config": {
            "text_encoder": model.text_encoder.model_name,
            "text_encoder_dim": model.text_encoder.output_dim,
            "decoder": {
                "vocab_size": model.decoder.vocab_size,
                "d_model": model.decoder.d_model,
                "nhead": model.decoder.nhead,
                "num_layers": model.decoder.num_layers,
                "max_len": model.decoder.max_len,
            },
        },
        "special_ids": {
            "pad": model.pad_id,
            "bos": model.bos_id,
            "eos": model.eos_id,
        },
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if config is not None:
        payload["config"] = config.to_dict() if hasattr(config, "to_dict") else config
    if vocab is not None:
        payload["vocab"] = vocab
    if extra:
        payload.update(extra)

    torch.save(payload, p)
    LOG.info("saved checkpoint %s (%s)", p, _human(p.stat().st_size))
    return p


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def load_checkpoint(path: Path | str, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict (weights + metadata)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    try:
        payload = torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:  # older torch without weights_only
        payload = torch.load(p, map_location=map_location)
    LOG.info("loaded checkpoint %s (epoch %s, step %s)", p, payload.get("epoch"), payload.get("step"))
    return payload
