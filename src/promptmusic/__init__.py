"""promptmusic - text-prompt conditioned music generation.

A small, fully-documented project that turns a written prompt (in English or
Tamil) into music: first a multi-track MIDI file, then rendered audio.

The public surface is deliberately tiny so beginners can find their way::

    from promptmusic import load_config, build_tokenizer

    cfg = load_config("smoke")
    tok = build_tokenizer(cfg.section("tokenizer"))

Everything else lives in submodules and is documented in ``docs/``.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "load_config",
    "Config",
    "build_tokenizer",
    "PROJECT_ROOT",
]

from .config import Config, load_config
from .paths import PROJECT_ROOT


def build_tokenizer(config=None):
    """Lazily import the tokenizer so importing ``promptmusic`` stays cheap."""
    from .data.tokenizer import build_tokenizer as _build

    return _build(config)
