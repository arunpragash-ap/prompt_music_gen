"""Configuration handling.

A config is a plain nested ``dict`` loaded from YAML. It is wrapped in
:class:`Config`, which adds dotted-path access (``cfg.train.epochs``) and
allows command-line overrides (``--set train.epochs=1 model.decoder.d_model=256``).

Why a dict and not dataclasses?
-------------------------------
Because every section must be able to grow without a schema migration, and
because users will want to drop in their own YAML file without importing
Python. The typed defaults below document the expected shape; unknown keys
are preserved so a user config can carry extra fields.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS_DIR, DATA_DIR, OUTPUTS_DIR

__all__ = ["Config", "DEFAULTS", "load_config", "deep_merge", "parse_override", "available_configs"]


DEFAULTS: dict[str, Any] = {
    "project": {
        "name": "prompt_music_gen",
        "seed": 42,
        "deterministic": False,
    },
    "paths": {
        # Relative paths are resolved against the project root.
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUTS_DIR),
        "cache_dir": "data/cache",
        "processed_dir": "data/processed",
    },
    "data": {
        "dataset": "midicaps",
        "hf_repo_id": "amaai-lab/MidiCaps",
        "caption_file": "train.json",
        "midi_subdir": "midicaps",
        # ``limit`` caps how many samples are indexed (null = all).
        "limit": 20000,
        "val_fraction": 0.05,
        "languages": ["en", "ta"],
        "translation": {
            "model": "ai4bharat/indictrans2-en-indic-dist-200M",
            "batch_size": 16,
            "max_input_length": 192,
            "max_new_tokens": 256,
            "num_beams": 4,
            "device": "auto",
        },
        "lyrics": {
            "enabled": False,
            "hf_repo_id": "asigalov61/Lyrics-MIDI-Dataset",
            "limit": 5000,
        },
    },
    "tokenizer": {
        "name": "remi",
        "use_programs": True,
        # miditok wants {(beat_start, beat_end): resolution}; YAML cannot
        # express tuple keys so we use "start,end" strings.
        "beat_res": {"0,4": 4},
        "num_velocities": 32,
        "tempo_range": [40, 250],
        "max_tokens": 4096,
        "special_tokens": {
            "pad": "<pad>",
            "bos": "<bos>",
            "eos": "<eos>",
            "unk": "<unk>",
        },
        "lyrics": {
            "enabled": False,
            "max_vocab": 8000,
            "mode": "hybrid",  # "hybrid" = word tokens for English, char tokens for Tamil
            "max_lyric_tokens": 256,
        },
    },
    "model": {
        "text_encoder": {
            "name": "google/mt5-base",
            "freeze": True,
            "max_length": 128,
            "dtype": "float32",
        },
        "decoder": {
            "d_model": 512,
            "nhead": 8,
            "num_layers": 12,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "max_len": 4096,
            "norm_first": True,
            "activation": "gelu",
            "tied_embeddings": False,
        },
    },
    "train": {
        "epochs": 3,
        "batch_size": 8,
        "grad_accum": 1,
        "lr": 3.0e-4,
        "encoder_lr": 0.0,  # 0 = the frozen encoder is not optimised
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.05,
        "max_grad_norm": 1.0,
        "num_workers": 2,
        "log_every": 50,
        "val_every": 500,
        "save_every": 500,
        "max_steps": None,
        "resume": None,
        "device": "auto",
        "compile": False,
        "amp": False,
    },
    "generate": {
        "max_new_tokens": 1024,
        # Measured REMI token density on the raga corpus: median ~12 tokens
        # per second of music (sparse classical material; dense pop can reach
        # ~20). Used by ``--duration`` to size max_new_tokens; the actual
        # duration is measured from the MIDI afterwards, so this only has to
        # be in the right ballpark.
        "tokens_per_second": 12.0,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        "num_samples": 1,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "seed": 42,
    },
    "render": {
        "sample_rate": 44100,
        "backend": "auto",  # auto | fluidsynth | pyfluidsynth | sf2utils | numpy
        "soundfont": None,  # path to a .sf2; None = auto-download FluidR3 GM
        "gain": 0.8,
    },
    "evaluate": {
        "metrics": ["compression_ratio", "key_match", "tempo_match", "note_density"],
    },
    "export": {
        "formats": ["pt", "torchscript"],
        "opset": 17,
        "half": False,
    },
    "reference": {
        "prime_bars": 2,
        "prime_max_tokens": 256,
        "analyze_audio": True,
    },
    "raga": {
        "enabled": False,
        "table": "data/ragas/raga_table.yaml",
        "variations_per_raga": 24,
        "seed": 7,
    },
}


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``.

    Lists are replaced wholesale (not concatenated) so a user can shorten a
    list such as ``data.languages``.
    """
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(raw: str) -> Any:
    """Turn a command-line string into a Python value."""
    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null", "~"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith(("[", "{")):
        try:
            return yaml.safe_load(raw)
        except yaml.YAMLError:
            return raw
    return raw


def parse_override(item: str) -> tuple[list[str], Any]:
    """Parse ``a.b.c=value`` into ``(["a","b","c"], value)``."""
    if "=" not in item:
        raise ValueError(f"override must look like key.path=value, got: {item!r}")
    key, _, raw = item.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"override has an empty key: {item!r}")
    return key.split("."), _coerce(raw)


def available_configs() -> list[Path]:
    """List the YAML configs shipped in ``configs/``."""
    if not CONFIGS_DIR.is_dir():
        return []
    return sorted(p for p in CONFIGS_DIR.glob("*.yaml"))


#: Name of the config used when none is specified: ``configs/default.yaml``,
#: falling back to the built-in ``DEFAULTS`` dict if that file is absent.
DEFAULT_CONFIG = "default"


# --------------------------------------------------------------------------- #
# The config object
# --------------------------------------------------------------------------- #
class Config:
    """Nested configuration with dotted access.

    >>> cfg = load_config("smoke")
    >>> cfg.train.epochs
    1
    >>> cfg.get("model.decoder.d_model")
    512
    """

    def __init__(self, data: dict | None = None, source: str | Path | None = None):
        self._data = deep_merge(DEFAULTS, data or {})
        self.source = str(source) if source else "defaults"

    # -- construction ------------------------------------------------------ #
    @classmethod
    def load(cls, source: str | Path | None = None) -> Config:
        """Load a config from a YAML path, a named config, or the defaults.

        ``source`` may be:
        * ``None``                     -> built-in defaults
        * ``"smoke"``                  -> ``configs/smoke.yaml``
        * ``"configs/smoke.yaml"``     -> that exact file
        * a path to any YAML file      -> that file
        """
        if source is None:
            return cls(DEFAULTS, source="defaults")

        path = Path(source)
        if not path.is_absolute() and not path.exists():
            candidate = CONFIGS_DIR / f"{source}.yaml"
            if candidate.exists():
                path = candidate
            elif (CONFIGS_DIR / source).exists():
                path = CONFIGS_DIR / source
        if not path.exists():
            raise FileNotFoundError(f"config not found: {source}")

        with open(path, encoding="utf-8") as fh:
            user_data = yaml.safe_load(fh) or {}
        if not isinstance(user_data, dict):
            raise ValueError(f"config file must contain a mapping: {path}")
        return cls(deep_merge(DEFAULTS, user_data), source=path)

    # -- access ------------------------------------------------------------ #
    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a value by dotted path, e.g. ``"train.lr"``."""
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, dotted: str, value: Any) -> None:
        """Set a value by dotted path, creating intermediate dicts."""
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def apply_overrides(self, overrides: list[str] | None) -> Config:
        """Apply ``key.path=value`` strings (in place) and return self."""
        for item in overrides or []:
            parts, value = parse_override(item)
            self.set(".".join(parts), value)
        return self

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self._data, fh, sort_keys=False, allow_unicode=True)
        return p

    # -- section helpers --------------------------------------------------- #
    def section(self, name: str) -> dict:
        value = self.get(name)
        return value if isinstance(value, dict) else {}

    def path(self, key: str, default: Path) -> Path:
        """Resolve a configured path relative to the project root."""
        raw = self.get(f"paths.{key}", default)
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            from .paths import PROJECT_ROOT

            p = PROJECT_ROOT / p
        return p.resolve()

    # -- dunder ------------------------------------------------------------ #
    def __contains__(self, dotted: str) -> bool:
        sentinel = object()
        return self.get(dotted, sentinel) is not sentinel

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails.
        if name.startswith("_"):
            raise AttributeError(name)
        data = self.__dict__.get("_data", {})
        if name in data:
            return data[name]
        raise AttributeError(f"config has no key {name!r}")

    def __repr__(self) -> str:
        return f"Config(source={self.source!r}, keys={sorted(self._data)})"

    def __iter__(self):
        return iter(self._data)


def load_config(source: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Convenience wrapper: load a config and apply CLI overrides."""
    cfg = Config.load(source)
    cfg.apply_overrides(overrides)
    return cfg


def _env_overrides() -> dict:
    """Allow ``PROMPTMUSIC_SEED`` etc. to influence the config."""
    out: dict = {}
    if (v := os.environ.get("PROMPTMUSIC_SEED")) is not None:
        out["project"] = {"seed": int(v)}
    return out


# Merge environment overrides into the defaults at import time so that every
# entry point (CLI, scripts, notebooks) sees the same values.
DEFAULTS = deep_merge(DEFAULTS, _env_overrides())
