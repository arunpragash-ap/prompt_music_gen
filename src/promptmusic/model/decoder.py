"""The autoregressive music decoder.

Shape of the problem
--------------------
The decoder is a *causal* Transformer: it reads a sequence of already-emitted
MIDI tokens and predicts the next one, while also looking at the text encoder
output (the "memory") so that the music it produces is conditioned on the
prompt.

::

    token ids ──► embedding ──► + positional ──► [ layer ] ──► [ layer ] ─► ... ──► logits
                                   encoding        │              │
                                       ▲        self-attn      self-attn
                                       │        (causal)       (causal)
                                       │              │              │
                                       │              ▼              ▼
                                       │        cross-attn      cross-attn  ◄── text memory
                                       │              │              │
                                       ▼              ▼              ▼
                                    position       feed-forward   feed-forward

Why implement attention by hand instead of using ``nn.TransformerDecoder``?
------------------------------------------------------------------------
Two reasons, both practical:

1. **Key/value caching.** PyTorch's built-in decoder recomputes attention
   over the whole prefix at every generation step, making decoding quadratic.
   Hand-rolled attention lets us cache the keys and values of past positions,
   which turns generation from *O(T^2)* into *O(T)*. On an M2 laptop that is
   the difference between a minute and an hour for a 1000-token piece.
2. **Dimension flexibility.** The text encoder emits 768-dim vectors while our
   decoder is 512-dim, so the cross-attention input must be projected. That
   is a single learned linear layer, and writing it here keeps the contract
   explicit.

The weight layout of the attention modules deliberately mirrors
``nn.MultiheadAttention`` (``in_proj_weight`` of shape ``[3d, d]``, plus an
``out_proj`` linear layer), so weights released by other projects can be
mapped onto it mechanically.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PositionalEncoding", "MultiHeadAttention", "DecoderLayer", "MusicDecoder", "CausalCache"]


# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    ``PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))``
    ``PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))``

    Sinusoidal rather than learned because it extrapolates: a model trained on
    512-token pieces can still be asked for 1000-token pieces without
    inventing new parameters.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Add positional encodings. ``offset`` shifts the window (for caching)."""
        seq_len = x.size(1)
        pos = self.pe[offset : offset + seq_len].to(dtype=x.dtype)
        return self.dropout(x + pos.unsqueeze(0))


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class MultiHeadAttention(nn.Module):
    """Multi-head attention with an optional key/value cache.

    Weight layout mirrors ``nn.MultiheadAttention`` with ``batch_first=True``
    so that foreign checkpoints can be loaded directly.

    Args:
        d_model: model width.
        nhead: number of attention heads (must divide ``d_model``).
        dropout: dropout applied to the attention probabilities.
        bias: whether the projection layers use a bias term.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout

        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model)) if bias else None
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight, math.sqrt(self.nhead))
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, hd = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * hd)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        need_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Attend and return ``(output, new_cache_or_None)``.

        ``attn_mask`` is an additive float mask broadcastable to
        ``[B, H, Tq, Tk]`` (use ``-inf`` to block a position). When
        ``is_causal`` is True a triangular mask is built automatically - only
        valid when ``Tq == Tk``.
        """
        q = F.linear(query, self.in_proj_weight[: self.d_model], None if self.in_proj_bias is None else self.in_proj_bias[: self.d_model])
        k = F.linear(key, self.in_proj_weight[self.d_model : 2 * self.d_model], None if self.in_proj_bias is None else self.in_proj_bias[self.d_model : 2 * self.d_model])
        v = F.linear(value, self.in_proj_weight[2 * self.d_model :], None if self.in_proj_bias is None else self.in_proj_bias[2 * self.d_model :])

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        new_cache = (k, v) if need_cache else None

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = self.out_proj(self._merge_heads(attn))
        return out, new_cache


# --------------------------------------------------------------------------- #
# Decoder layer
# --------------------------------------------------------------------------- #
# NOTE: these must be nn.Module instances, not the F.* functional forms.
# nn.Sequential only accepts Modules, so F.gelu would raise
# "torch._C._nn.gelu is not a Module subclass" the first time a decoder was built.
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class DecoderLayer(nn.Module):
    """Pre-norm Transformer decoder layer: self-attn -> cross-attn -> FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _ACTIVATIONS[activation](),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        self_attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        need_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        h = self.norm1(x) if self.norm_first else x
        attn_out, new_cache = self.self_attn(
            h, h, h, attn_mask=self_attn_mask, is_causal=is_causal, cache=cache, need_cache=need_cache
        )
        x = x + self.dropout(attn_out)

        h = self.norm2(x) if self.norm_first else x
        cross_out, _ = self.cross_attn(h, memory, memory, attn_mask=None, is_causal=False)
        x = x + self.dropout(cross_out)

        h = self.norm3(x) if self.norm_first else x
        x = x + self.dropout(self.ffn(h))
        return x, new_cache


# --------------------------------------------------------------------------- #
# Cache container
# --------------------------------------------------------------------------- #
class CausalCache(list):
    """One entry per decoder layer: ``(key_cache, value_cache)``.

    A plain ``list`` of tuples so it can be passed through PyTorch modules
    without graph tracking.
    """


# --------------------------------------------------------------------------- #
# The decoder
# --------------------------------------------------------------------------- #
class MusicDecoder(nn.Module):
    """Causal Transformer decoder over the music token vocabulary.

    Args:
        vocab_size: size of the vocabulary (music + lyric + ``<unk>``).
        d_model: width of every hidden vector.
        nhead: attention heads.
        num_layers: number of :class:`DecoderLayer` blocks.
        dim_feedforward: width of the feed-forward inner layer.
        dropout: dropout rate.
        max_len: maximum sequence length the positional encoding supports.
        text_dim: width of the text encoder output (768 for mT5-base). A
            learned projection maps it to ``d_model``.
        norm_first: ``True`` = pre-norm (more stable), ``False`` = post-norm.
        activation: ``"gelu"`` | ``"relu"`` | ``"silu"``.
        pad_id: id of the padding token (used to build attention masks).
        tie_embeddings: share the input embedding with the output head, which
            saves parameters and often helps small models.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 4096,
        text_dim: int = 768,
        norm_first: bool = True,
        activation: str = "gelu",
        pad_id: int = 0,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)
        self.emb_dropout = nn.Dropout(dropout)

        self.memory_proj = nn.Linear(text_dim, d_model) if text_dim != d_model else nn.Identity()
        self.memory_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, norm_first)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.output_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    # -- forward ----------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        cache: list | None = None,
        need_cache: bool = False,
    ) -> tuple[torch.Tensor, list | None]:
        """Return ``(logits [B, T, V], new_cache_or_None)``.

        ``input_ids`` are the token ids (``[B, T]``). ``memory`` is the text
        encoder output (``[B, T_text, text_dim]``), optionally masked with
        ``memory_mask`` (``[B, T_text]``, ``True`` = keep).
        """
        cache = cache if cache is not None else [None] * self.num_layers
        offset = cache[0][0].shape[2] if cache[0] is not None else 0

        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x, offset=offset)
        x = self.emb_dropout(x)

        mem = self.memory_proj(memory)
        if hasattr(self.memory_norm, "weight"):
            mem = self.memory_norm(mem)

        # A key-padding mask for the text side, if one was supplied.
        cross_mask = None
        if memory_mask is not None:
            # memory_mask: [B, T_text] with True = keep -> additive mask [B, 1, 1, T_text]
            cross_mask = torch.zeros_like(memory_mask, dtype=x.dtype)
            cross_mask = cross_mask.masked_fill(~memory_mask.bool(), float("-inf"))
            cross_mask = cross_mask[:, None, None, :]

        is_causal = (offset == 0) and input_ids.size(1) > 1

        new_cache: list = []
        for i, layer in enumerate(self.layers):
            x, layer_cache = layer(
                x, mem,
                self_attn_mask=None,
                is_causal=is_causal,
                cache=cache[i],
                need_cache=need_cache,
            )
            new_cache.append(layer_cache)

        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits, (new_cache if need_cache else None)

    # -- helpers ----------------------------------------------------------- #
    def num_parameters(self, trainable_only: bool = False) -> int:
        params = (p for p in self.parameters() if p.requires_grad) if trainable_only else self.parameters()
        return sum(p.numel() for p in params)

    def extra_repr(self) -> str:
        return (
            f"vocab={self.vocab_size}, d_model={self.d_model}, nhead={self.nhead}, "
            f"layers={self.num_layers}"
        )

    # -- text2midi weight import ------------------------------------------ #
    def load_released_text2midi(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Best-effort import of the released text2midi decoder weights.

        The released checkpoint is an 18-layer, 768-wide model built with
        ``nn.TransformerDecoder``. We map every tensor whose shape matches and
        report the rest, so a user can see exactly how much of the released
        model actually landed in ours.

        Returns a report: ``{"loaded": [...], "shape_mismatch": [...],
        "missing": [...], "unexpected": [...]}``.
        """
        our = self.state_dict()
        loaded: list[str] = []
        mismatched: list[str] = []
        skipped: list[str] = []

        for key, tensor in state_dict.items():
            if key not in our:
                skipped.append(key)
                continue
            if tuple(our[key].shape) == tuple(tensor.shape):
                our[key] = tensor
                loaded.append(key)
            else:
                mismatched.append(f"{key}: released {tuple(tensor.shape)} vs ours {tuple(our[key].shape)}")

        self.load_state_dict(our, strict=False)
        return {
            "loaded": loaded,
            "shape_mismatch": mismatched,
            "skipped_unknown_keys": skipped,
            "loaded_count": len(loaded),
            "mismatched_count": len(mismatched),
        }
