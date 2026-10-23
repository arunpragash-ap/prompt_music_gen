"""Small, dependency-free helpers used across the project.

Nothing in here needs PyTorch, so it stays importable in a bare environment
(useful for the disk-cleanup and repo-tooling scripts).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

__all__ = [
    "set_seed",
    "sha256_file",
    "human_bytes",
    "read_json",
    "write_json",
    "read_jsonl",
    "write_jsonl",
    "append_jsonl",
    "chunked",
    "timed",
    "format_duration",
    "device_info",
    "env_report",
    "disk_usage_report",
    "select_device",
    "dir_size",
    "which",
    "run_cmd",
    "logger",
]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def logger(name: str = "promptmusic"):
    """Return a configured stdlib logger.

    Uses a single stream handler so CLI output stays readable; library code
    should use ``logging.getLogger(__name__)`` instead of this helper.
    """
    import logging

    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
                                               datefmt="%H:%M:%S"))
        log.addHandler(handler)
        log.setLevel(os.environ.get("PROMPTMUSIC_LOGLEVEL", "INFO").upper())
        log.propagate = False
    return log


LOG = logger()


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed every random source we use.

    ``torch`` is optional here so this function still works in a
    download-only environment.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Hashing / files
# --------------------------------------------------------------------------- #
def sha256_file(path: Path | str, chunk_size: int = 1 << 20, progress: bool = False) -> str:
    """SHA-256 of a file, streamed so multi-GB files do not land in RAM."""
    h = hashlib.sha256()
    total = Path(path).stat().st_size if Path(path).is_file() else 0
    done = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
            done += len(chunk)
            if progress and total:
                pct = 100.0 * done / total
                print(f"\r  hashing {pct:5.1f}%  ({done / 1e6:.0f}/{total / 1e6:.0f} MB)",
                      end="", flush=True)
    if progress:
        print()
    return h.hexdigest()


def human_bytes(n: float) -> str:
    """Human readable byte count, e.g. ``1.62 GB``."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(n)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.2f} PB"


def dir_size(path: Path | str) -> int:
    """Total size in bytes of a directory tree (follows symlinks=False)."""
    total = 0
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    for entry in p.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def which(cmd: str) -> str | None:
    """``shutil.which`` that returns ``None`` instead of raising."""
    return shutil.which(cmd)


def run_cmd(cmd: list[str] | str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the completed process."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


# --------------------------------------------------------------------------- #
# JSON / JSONL
# --------------------------------------------------------------------------- #
def read_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path | str, obj: Any, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)
    return p


def read_jsonl(path: Path | str) -> Iterator[dict]:
    """Read a JSON-lines file, skipping blank and malformed lines."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def append_jsonl(path: Path | str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def chunked(items: Iterable, size: int) -> Iterator[list]:
    """Yield lists of at most ``size`` items."""
    buf: list = []
    for item in items:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def timed(label: str = "task"):
    """Context manager that prints the wall time of a block."""

    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            self.elapsed = time.perf_counter() - self.t0
            print(f"{label}: {format_duration(self.elapsed)}")
            return False

    return _Timer()


def format_duration(seconds: float) -> str:
    """``3661.5`` -> ``1h 01m 01s``."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def device_info() -> dict:
    """Describe the machine and the accelerators available."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["mps_available"] = torch.backends.mps.is_available()
        info["mps_built"] = torch.backends.mps.is_built()
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    return info


def select_device(requested: str = "auto") -> str:
    """Pick a torch device string.

    ``auto`` prefers CUDA, then Apple MPS, then CPU.
    """
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# Environment report (`promptmusic info`)
# --------------------------------------------------------------------------- #
def env_report() -> dict[str, Any]:
    """Describe this machine's ability to run the project.

    Grouped into four sections so the output of ``promptmusic info`` reads as
    a checklist: hardware, Python packages, audio tooling, and data on disk.
    """
    from .paths import DATA_DIR, DOWNLOADS_DIR, HF_CACHE_DIR, OUTPUTS_DIR

    hardware = device_info()
    hardware["ram_total_gb"] = _total_ram_gb()
    hardware["free_disk_gb"] = round(
        shutil.disk_usage(str(Path.cwd())).free / (1024**3), 1
    )
    hardware["selected_device"] = select_device("auto")

    packages: dict[str, Any] = {}
    for mod, attr in (
        ("torch", "__version__"),
        ("transformers", "__version__"),
        ("miditok", "__version__"),
        ("symusic", "__version__"),
        ("pretty_midi", "__version__"),
        ("numpy", "__version__"),
        ("yaml", "__version__"),
        ("librosa", "__version__"),
    ):
        try:
            m = __import__(mod)
            packages[mod] = getattr(m, attr, "?")
        except ImportError:
            packages[mod] = "not installed"

    audio: dict[str, Any] = {
        "fluidsynth_cli": which("fluidsynth") or which("fluid-synth") or "not found",
        "soundfont": None,
    }
    try:
        from .render import find_soundfont

        sf = find_soundfont()
        audio["soundfont"] = (
            f"{sf} ({human_bytes(sf.stat().st_size)})" if sf else "not found"
        )
    except Exception as exc:  # noqa: BLE001
        audio["soundfont"] = f"lookup failed: {exc}"

    def _presence(p: Path) -> str:
        if not p.exists():
            return "missing"
        if p.is_dir():
            size = dir_size(p)
            return f"present ({human_bytes(size)})"
        return f"present ({human_bytes(p.stat().st_size)})"

    data = {
        "dataset_dir": f"{DATA_DIR}: {_presence(DATA_DIR)}",
        "downloads_dir": f"{DOWNLOADS_DIR}: {_presence(DOWNLOADS_DIR)}",
        "outputs_dir": f"{OUTPUTS_DIR}: {_presence(OUTPUTS_DIR)}",
        "hf_cache": f"{HF_CACHE_DIR}: {_presence(HF_CACHE_DIR)}",
        "midi_files": _count_midi(DATA_DIR),
    }
    return {
        "hardware": hardware,
        "packages": packages,
        "audio": audio,
        "data": data,
    }


def _total_ram_gb() -> float | None:
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
        )
        if out.returncode == 0 and out.stdout.strip():
            return round(int(out.stdout.strip()) / (1024**3), 1)
    except Exception:  # noqa: BLE001 - non-macOS or sysctl absent
        pass
    return None


def _count_midi(root: Path, cap: int = 200_000) -> int:
    if not root.is_dir():
        return 0
    n = 0
    try:
        for p in root.rglob("*"):
            if p.suffix.lower() in {".mid", ".midi"}:
                n += 1
                if n >= cap:
                    break
    except OSError:
        pass
    return n


# --------------------------------------------------------------------------- #
# Disk report (`promptmusic disk-report`)
# --------------------------------------------------------------------------- #
def disk_usage_report() -> dict[str, Any]:
    """Measure the project's directories and mark which are safe to delete.

    The ``deletable`` flag is the whole point: every artifact here can be
    regenerated by re-running a documented command, so none of it belongs in
    version control or needs to survive a cleanup. ``docs/10_DISK_CLEANUP.md``
    gives the step-by-step version of this same table.
    """
    from .paths import (
        CACHE_DIR,
        DATA_DIR,
        DOWNLOADS_DIR,
        HF_CACHE_DIR,
        OUTPUTS_DIR,
        PROCESSED_DIR,
        PROJECT_ROOT,
    )

    entries: list[dict[str, Any]] = []
    total = 0

    def add(path: Path, deletable: bool, reason: str, *, label: str | None = None) -> None:
        nonlocal total
        if not path.exists():
            return
        size = dir_size(path)
        total += size
        entries.append({
            "path": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
            "label": label or path.name,
            "size_bytes": size,
            "size_human": human_bytes(size),
            "deletable": deletable,
            "reason": reason,
        })

    add(HF_CACHE_DIR, True, "Hugging Face downloads; re-fetched automatically on demand")
    add(CACHE_DIR, True, "derived caches (caption index, translation cache)")
    add(PROCESSED_DIR, True, "token caches; rebuild with `promptmusic tokenize`")
    add(OUTPUTS_DIR, True, "checkpoints, samples and logs; rebuild with train/generate")
    add(DATA_DIR, True, "datasets; re-download with `promptmusic download`")
    add(DOWNLOADS_DIR, True, "raw archives; re-download with `promptmusic download`")
    seen: set[Path] = set()
    for p in PROJECT_ROOT.rglob("__pycache__"):
        if p in seen:
            continue
        seen.add(p)
        add(p, True, "Python bytecode cache")
    add(PROJECT_ROOT / ".pytest_cache", True, "pytest cache")
    add(PROJECT_ROOT / ".venv", False,
        "the virtual environment - delete only if you plan to rebuild it with "
        "`uv venv` + `uv pip install -e .` (saves a lot, costs a few minutes)")
    # The source part of data/ is small and must survive; the large artifact
    # subdirectories (downloads, cache, processed) are counted above, so do
    # not add data/ as a whole here - that double-counted ~2 GB and marked
    # deletable content as "keep".
    for src in (PROJECT_ROOT / "data" / "ragas", PROJECT_ROOT / "data" / "lyrics"):
        add(src, False, "hand-written source (raga table / your lyrics) - KEEP")

    entries.sort(key=lambda e: -e["size_bytes"])
    reclaimable = sum(e["size_bytes"] for e in entries if e["deletable"])
    return {
        "entries": entries,
        "total_bytes": total,
        "total_human": human_bytes(total),
        "reclaimable_bytes": reclaimable,
        "reclaimable_human": human_bytes(reclaimable),
        "doc": "docs/10_DISK_CLEANUP.md",
    }
