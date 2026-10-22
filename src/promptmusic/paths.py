"""Central place for every filesystem path the project uses.

Keeping paths in one module means:

* Nothing in the code base hard-codes an absolute path.
* Every large, regenerable artifact lives under a known directory, so the
  disk-cleanup guide (``docs/10_DISK_CLEANUP.md``) can be kept accurate.
* A user can relocate the whole project by setting ``PROMPTMUSIC_HOME``.

Environment variables
---------------------
``PROMPTMUSIC_HOME``   Overrides the project root. Default: the directory
                       that contains ``src/`` and ``pyproject.toml``.
``PROMPTMUSIC_DATA``   Overrides the dataset root (downloaded MIDI + captions).
``PROMPTMUSIC_OUTPUT`` Overrides the output root (checkpoints, samples).
``HF_HOME``            Used by ``huggingface_hub`` for the model cache.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "SRC_DIR",
    "CONFIGS_DIR",
    "DOCS_DIR",
    "RAGAS_DIR",
    "SCRIPTS_DIR",
    "TESTS_DIR",
    "DATA_DIR",
    "DOWNLOADS_DIR",
    "CACHE_DIR",
    "PROCESSED_DIR",
    "OUTPUTS_DIR",
    "HF_CACHE_DIR",
    "ensure_dir",
    "rel_to_root",
]


def _find_project_root() -> Path:
    """Locate the project root.

    Order of preference:
    1. ``PROMPTMUSIC_HOME`` environment variable.
    2. Walk up from this file until a directory containing ``pyproject.toml``
       is found (works for an editable install and for a plain checkout).
    3. Fall back to the current working directory.
    """
    env = os.environ.get("PROMPTMUSIC_HOME")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT: Path = _find_project_root()
SRC_DIR: Path = PROJECT_ROOT / "src"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
RAGAS_DIR: Path = PROJECT_ROOT / "data" / "ragas"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

# --- Large, regenerable data -------------------------------------------------
DATA_DIR: Path = Path(os.environ.get("PROMPTMUSIC_DATA", PROJECT_ROOT / "dataset")).resolve()
DOWNLOADS_DIR: Path = PROJECT_ROOT / "data" / "downloads"
CACHE_DIR: Path = PROJECT_ROOT / "data" / "cache"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

# --- Model outputs -----------------------------------------------------------
OUTPUTS_DIR: Path = Path(os.environ.get("PROMPTMUSIC_OUTPUT", PROJECT_ROOT / "outputs")).resolve()

# --- Hugging Face model cache ------------------------------------------------
HF_CACHE_DIR: Path = Path(
    os.environ.get("PROMPTMUSIC_HF_CACHE", PROJECT_ROOT / ".hf_cache")
).resolve()


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def rel_to_root(path: Path | str) -> str:
    """Render ``path`` relative to the project root when possible.

    Used by the CLI so messages show short, readable paths.
    """
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)
