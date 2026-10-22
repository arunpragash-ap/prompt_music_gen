"""The frozen multilingual text encoder.

Why a *separate*, frozen encoder?
---------------------------------
The job of the text side is to turn "a cheerful Tamil folk song with flute
and tabla" into a sequence of vectors that mean the same thing as "an upbeat
Tamil folk tune with flute and tabla". We do not need to train that ability
from scratch - large text models already have it. Freezing the encoder:

* keeps ~220 M parameters out of the optimiser (huge memory saving on an
  8 GB laptop),
* preserves the multilingual competence we specifically chose the model for,
* means the checkpoint we ship is small (only the decoder changes).

Choice of encoder: ``google/mt5-base``
--------------------------------------
Verified facts that drove the choice:

* ``google/flan-t5-base`` (what the reference text2midi uses) contains
  **1 of 15** sampled Tamil characters in its SentencePiece vocabulary.
  Tamil prompts would be shredded into unknown tokens - no Tamil support.
* ``google/mt5-base`` contains **15 of 15**. It was pretrained on 101
  languages, Tamil included.
* Both models are architecturally identical: ``d_model=768``,
  12 encoder layers, 12 heads, ``d_ff=2048``. Swapping one for the other
  keeps the decoder's cross-attention input dimension unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..paths import HF_CACHE_DIR

__all__ = ["TextEncoder", "TextTokenizer", "TEXT_ENCODER_DIM"]


#: Hidden size of ``google/mt5-base`` (and ``google/flan-t5-base``).
TEXT_ENCODER_DIM = 768


class TextEncoder(nn.Module):
    """Wraps a T5-family encoder and returns its last hidden state.

    Args:
        model_name: Hugging Face id of a T5-family *encoder*. Both
            ``google/mt5-base`` and ``google/flan-t5-base`` work.
        freeze: If ``True`` (the default) the encoder is set to eval mode and
            its parameters get ``requires_grad=False``.
        max_length: Truncation length for the token ids (tokens, not bytes).

    Forward signature::

        hidden = encoder(input_ids, attention_mask)   # [B, T_text, 768]
    """

    def __init__(
        self,
        model_name: str = "google/mt5-base",
        *,
        freeze: bool = True,
        max_length: int = 128,
        dtype: str = "float32",
        device: str | None = None,
    ):
        super().__init__()
        try:
            from transformers import T5EncoderModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for the text encoder.\n"
                "Install it with: pip install 'transformers>=4.44,<5'"
            ) from exc

        self.model_name = model_name
        self.freeze = freeze
        self.max_length = max_length
        self.dtype = _parse_dtype(dtype)

        # ``torch_dtype`` is deprecated in favour of ``dtype`` from
        # transformers 4.56, but the project pins ``>=4.44,<5`` and the old
        # name still works (with a warning) across that whole range, whereas
        # ``dtype`` would crash on 4.44-4.55.
        self.encoder = T5EncoderModel.from_pretrained(
            model_name, torch_dtype=self.dtype, cache_dir=str(HF_CACHE_DIR / "hub")
        )
        self.hidden_size = int(getattr(self.encoder.config, "d_model", TEXT_ENCODER_DIM))

        if freeze:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        if device:
            self.encoder.to(device)

    # -- properties -------------------------------------------------------- #
    @property
    def output_dim(self) -> int:
        return self.hidden_size

    def train(self, mode: bool = True):  # noqa: D401 - nn.Module override
        """Never put a frozen encoder into training mode."""
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    # -- forward ----------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``last_hidden_state`` of shape ``[B, T_text, hidden]``."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.freeze:
            with torch.no_grad():
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def encode_texts(self, texts: list[str], tokenizer: TextTokenizer) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: tokenise then encode a batch of strings."""
        enc = tokenizer(texts, max_length=self.max_length)
        device = next(self.parameters()).device
        return self(enc["input_ids"].to(device), enc["attention_mask"].to(device))

    # -- persistence ------------------------------------------------------- #
    def extra_repr(self) -> str:
        return f"model_name={self.model_name!r}, freeze={self.freeze}, dim={self.hidden_size}"


def _parse_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(str(name).lower(), torch.float32)


class TextTokenizer:
    """The SentencePiece tokenizer that belongs with the encoder.

    Kept separate from :class:`TextEncoder` so that generation can tokenise a
    prompt without holding the (large) encoder in memory, and so tests can
    exercise tokenisation offline.
    """

    def __init__(self, model_name: str = "google/mt5-base", max_length: int = 128):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers is required; pip install 'transformers>=4.44,<5'") from exc

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=str(HF_CACHE_DIR / "hub")
        )
        # mT5 does not have a pad token by default; it uses <extra_id_*>.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"

    def __call__(self, texts: str | list[str], max_length: int | None = None) -> dict[str, torch.Tensor]:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        enc = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length or self.max_length,
        )
        if single:
            enc = {k: v[0] for k, v in enc.items()}
        return enc

    def __len__(self) -> int:
        return len(self.tokenizer)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "vocab_size": len(self.tokenizer),
            "max_length": self.max_length,
        }
