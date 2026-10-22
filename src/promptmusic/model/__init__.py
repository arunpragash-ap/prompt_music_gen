"""Model components: the frozen text encoder, the causal decoder, and assembly."""

from __future__ import annotations

from .build import MusicModel, build_model, count_parameters, load_checkpoint, save_checkpoint
from .decoder import MusicDecoder, MultiHeadAttention, PositionalEncoding
from .text_encoder import TEXT_ENCODER_DIM, TextEncoder, TextTokenizer

__all__ = [
    "MusicModel",
    "build_model",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "MusicDecoder",
    "MultiHeadAttention",
    "PositionalEncoding",
    "TextEncoder",
    "TextTokenizer",
    "TEXT_ENCODER_DIM",
]
