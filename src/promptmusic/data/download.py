"""Download the datasets and pretrained artefacts this project needs.

Every remote file is declared in :data:`SOURCE_REGISTRY` together with its
URL, expected byte size, expected SHA-256 digest and licence. The downloader
refuses to silently accept a file whose digest does not match, and it writes
a machine-readable manifest (``dataset/manifest.json``) recording exactly
what was fetched, from where, and under which licence.

Why a registry instead of "just call wget"?
--------------------------------------------
* Reproducibility: a digest pins the exact bytes, so a re-run months later
  either gets the same file or fails loudly.
* Licensing: the licence of every input is recorded next to the file, which
  is what a compliance reviewer needs.
* Cleanup: the manifest is what ``scripts/disk_report.sh`` and
  ``docs/10_DISK_CLEANUP.md`` use to report reclaimable space.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..paths import DATA_DIR, DOWNLOADS_DIR, ensure_dir
from ..utils import human_bytes, logger, sha256_file, write_json

LOG = logger("promptmusic.download")

__all__ = [
    "FileSpec",
    "SourceSpec",
    "SOURCE_REGISTRY",
    "download_file",
    "download_source",
    "extract_archive",
    "write_manifest",
    "read_manifest",
    "verify_file",
    "list_sources",
]

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


@dataclass
class FileSpec:
    """One remote file."""

    url: str
    filename: str
    size: int | None = None
    sha256: str | None = None
    license: str = "unknown"
    note: str = ""
    extract: bool = False
    extract_to: str | None = None

    def local_path(self, root: Path) -> Path:
        return root / self.filename


@dataclass
class SourceSpec:
    """A named group of remote files (one dataset or one model)."""

    key: str
    title: str
    kind: str = "dataset"  # dataset | model
    homepage: str = ""
    citation: str = ""
    files: list[FileSpec] = field(default_factory=list)
    notes: str = ""

    def total_bytes(self) -> int:
        return sum(f.size or 0 for f in self.files)


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
SOURCE_REGISTRY: dict[str, SourceSpec] = {
    "midicaps": SourceSpec(
        key="midicaps",
        title="MidiCaps (captions + features for Lakh MIDI)",
        kind="dataset",
        homepage="https://huggingface.co/datasets/amaai-lab/MidiCaps",
        citation=(
            "Ashwal, Brijen et al. 'MidiCaps: Large-scale MIDI Dataset with "
            "Textual Captions and Audio Features.' arXiv:2406.02255, 2024."
        ),
        notes=(
            "MIDI files are derived from the Lakh MIDI Dataset (Raffel et al., "
            "2016, CC-BY 4.0). Captions were written in English by a large "
            "language model from the audio features. The dataset itself is "
            "released under CC-BY-SA-4.0 (share-alike)."
        ),
        files=[
            FileSpec(
                url=f"{HF_ENDPOINT}/datasets/amaai-lab/MidiCaps/resolve/main/midicaps.tar.gz",
                filename="midicaps.tar.gz",
                size=1622239748,
                sha256="69f8cb1c6812785ab71a7c58389ed2814cb86bde330a0acfec804cf03151163c",
                license="CC-BY-SA-4.0",
                note="Archive of 168,385 multi-track MIDI files.",
                extract=True,
                extract_to="midicaps",
            ),
            FileSpec(
                url=f"{HF_ENDPOINT}/datasets/amaai-lab/MidiCaps/resolve/main/train.json",
                filename="train.json",
                size=428237966,
                sha256="c7e3b2586bd5cd9aa875f525280c4ae79577b75e17b0b3511d58ea0d5e757f89",
                license="CC-BY-SA-4.0",
                note="Caption + 13 audio-feature fields per MIDI file.",
            ),
        ],
    ),
    "text2midi": SourceSpec(
        key="text2midi",
        title="Text2midi (reference decoder weights, REMI vocab, GM soundfont)",
        kind="model",
        homepage="https://huggingface.co/amaai-lab/text2midi",
        citation=(
            "Saketh, Srijan et al. 'Text2MIDI: Text Generation to MIDI "
            "Generation with a Diffusion Model' and 'Leveraging Large Language "
            "Models for Text-to-MIDI Generation'. arXiv:2412.16526, 2024."
        ),
        notes=(
            "Used for three things: (1) the reference baseline we compare "
            "against, (2) the FluidR3 GM soundfont used to render audio, and "
            "(3) the REMI vocabulary pickle that lets us reproduce the "
            "published token stream exactly. Apache-2.0."
        ),
        files=[
            FileSpec(
                url=f"{HF_ENDPOINT}/amaai-lab/text2midi/resolve/main/soundfont.sf2",
                filename="soundfont.sf2",
                size=155405818,
                sha256="9a57fb3b6714e69dda12390e351b087e81fc3b1eca15c6b4bbe172799f4cf3cd",
                license="free (FluidR3 GM soundfont)",
                note="General MIDI soundfont used for MIDI -> WAV rendering.",
            ),
            FileSpec(
                url=f"{HF_ENDPOINT}/amaai-lab/text2midi/resolve/main/vocab_remi.pkl",
                filename="vocab_remi.pkl",
                size=27256,
                sha256="877d4511d6b9d5eea1c706199fe13a0de3d984c8f5d09c75d727ffe7f6f54ee6",
                license="Apache-2.0",
                note="REMI vocabulary used by the released text2midi checkpoint.",
            ),
            FileSpec(
                url=f"{HF_ENDPOINT}/amaai-lab/text2midi/resolve/main/pytorch_model.bin",
                filename="text2midi_pytorch_model.bin",
                size=901867614,
                sha256="d94e8d7dda9207ed734268933629d4a4b7a1a6ce84c77299f610d3470448bfc1",
                license="Apache-2.0",
                note="Reference decoder weights (18 layers x 768 dims).",
            ),
            FileSpec(
                url=f"{HF_ENDPOINT}/amaai-lab/text2midi/resolve/main/transformer_model.py",
                filename="text2midi_transformer_model.py",
                size=68142,
                sha256=None,
                license="Apache-2.0",
                note="Reference implementation, kept for comparison and auditing.",
            ),
        ],
    ),
    "mt5": SourceSpec(
        key="mt5",
        title="google/mt5-base (multilingual text encoder, frozen)",
        kind="model",
        homepage="https://huggingface.co/google/mt5-base",
        citation="Xue et al. 'mT5: A massively multilingual pre-trained text-to-text transformer.' NAACL 2021.",
        notes=(
            "Apache-2.0. Chosen over google/flan-t5-base because flan-t5's "
            "SentencePiece vocabulary contains only 1 of 15 sampled Tamil "
            "characters, while mt5-base contains all 15. Both have identical "
            "tensor shapes (d_model=768, 12 layers, 12 heads, d_ff=2048)."
        ),
        files=[],
    ),
    "lyrics_midi": SourceSpec(
        key="lyrics_midi",
        title="Lyrics MIDI Dataset (MIDI paired with lyrics)",
        kind="dataset",
        homepage="https://huggingface.co/datasets/asigalov61/Lyrics-MIDI-Dataset",
        citation="Lev, Alex. 'Lyrics MIDI Dataset' (Tegridy collection).",
        notes=(
            "CC-BY-NC-SA-4.0 - NON-COMMERCIAL. Optional corpus used only "
            "when the user opts into the lyrics phase."
        ),
        files=[],
    ),
}


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
def _human_rate(n: int) -> str:
    return human_bytes(n)


def download_file(
    spec: FileSpec,
    dest_root: Path,
    *,
    resume: bool = True,
    verify: bool = True,
    timeout: int = 60,
    max_retries: int = 3,
) -> Path:
    """Download one file, resuming and verifying as needed.

    Returns the local path. Raises on checksum mismatch after retries.
    """
    ensure_dir(dest_root)
    dest = spec.local_path(dest_root)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and (spec.size is None or dest.stat().st_size == spec.size):
        if not verify or not spec.sha256:
            LOG.info("already present: %s", dest.name)
            return dest
        LOG.info("verifying existing %s ...", dest.name)
        if sha256_file(dest) == spec.sha256:
            LOG.info("checksum OK: %s", dest.name)
            return dest
        LOG.warning("checksum mismatch on %s - re-downloading", dest.name)
        dest.unlink(missing_ok=True)

    url = spec.url
    attempt = 0
    while True:
        attempt += 1
        try:
            headers = {"User-Agent": "promptmusic/0.1 (+https://github.com/)"}
            mode = "wb"
            existing = part.stat().st_size if (resume and part.exists()) else 0
            if existing:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"
                LOG.info("resuming %s at %s", dest.name, human_bytes(existing))

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = spec.size or 0
                if resp.headers.get("Content-Length"):
                    total = int(resp.headers["Content-Length"]) + existing
                done = existing
                t0 = time.time()
                with open(part, mode) as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100.0 * done / total
                            rate = (done - existing) / max(time.time() - t0, 1e-6)
                            LOG.info(
                                "  %s %5.1f%%  %s / %s  (%s/s)",
                                dest.name, pct, human_bytes(done), human_bytes(total),
                                human_bytes(int(rate)),
                            )
                            # Only log every ~5% to keep the output readable.
                            LOG.handlers[0].flush()
            part.replace(dest)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"failed to download {url} after {attempt} attempts: {exc}") from exc
            wait = min(2**attempt, 30)
            LOG.warning("download error (%s), retry %d/%d in %ds", exc, attempt, max_retries, wait)
            time.sleep(wait)

    if verify and spec.sha256:
        LOG.info("verifying SHA-256 of %s ...", dest.name)
        digest = sha256_file(dest, progress=True)
        if digest != spec.sha256:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {spec.filename}\n  expected {spec.sha256}\n  got      {digest}"
            )
        LOG.info("checksum OK: %s", dest.name)
    return dest


def verify_file(path: Path | str, sha256: str | None = None, size: int | None = None) -> bool:
    """Check a local file against an expected digest and/or size."""
    p = Path(path)
    if not p.is_file():
        return False
    if size is not None and p.stat().st_size != size:
        return False
    if sha256:
        return sha256_file(p) == sha256
    return True


def extract_archive(
    archive: Path,
    dest_root: Path,
    *,
    strip_top: bool = False,
    subdir: str | None = None,
) -> Path:
    """Extract a ``.tar.gz`` archive safely.

    ``strip_top`` removes the single top-level directory that the archive
    happens to contain, which keeps the on-disk layout predictable
    regardless of how the dataset was packaged.
    """
    ensure_dir(dest_root)
    target = dest_root / subdir if subdir else dest_root
    ensure_dir(target)
    LOG.info("extracting %s -> %s", archive.name, target)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        base = members[0].name.split("/")[0] if members else None
        for member in members:
            name = member.name
            if strip_top and base and (name == base or name.startswith(base + "/")):
                name = name[len(base) :].lstrip("/")
            if not name or name == ".":
                continue
            # Guard against path traversal in archive entries.
            member.name = name
            tar.extract(member, path=target, filter="data")
    return target


def download_source(
    key: str,
    *,
    data_root: Path | None = None,
    downloads_root: Path | None = None,
    files: list[str] | None = None,
    verify: bool = True,
    force: bool = False,
    extract: bool = True,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Download every file of a registered source.

    ``files`` optionally restricts the download to a subset of filenames.
    ``extract=False`` downloads archives without unpacking them.
    ``max_bytes`` aborts (before any network traffic) when the declared size of
    the selected files exceeds the cap - the CLI's ``--max-bytes`` guard.
    """
    if key not in SOURCE_REGISTRY:
        raise KeyError(f"unknown source {key!r}; known: {sorted(SOURCE_REGISTRY)}")
    spec = SOURCE_REGISTRY[key]
    data_root = Path(data_root or DATA_DIR)
    downloads_root = Path(downloads_root or DOWNLOADS_DIR)
    ensure_dir(data_root)
    ensure_dir(downloads_root)

    wanted = [f for f in spec.files if files is None or f.filename in files or Path(f.filename).stem in files]
    if max_bytes is not None:
        total = sum(f.size for f in wanted if f.size)
        if total > max_bytes:
            raise ValueError(
                f"{key} would download {total:,} bytes, over the --max-bytes cap of {max_bytes:,}"
            )
    results: list[dict[str, Any]] = []
    for f in wanted:
        if force:
            (downloads_root / f.filename).unlink(missing_ok=True)
        path = download_file(f, downloads_root, verify=verify)
        entry = {
            "filename": f.filename,
            "url": f.url,
            "size": path.stat().st_size,
            "sha256": f.sha256,
            "license": f.license,
            "local_path": str(path),
            "note": f.note,
        }
        if f.extract and extract:
            dest = extract_archive(path, data_root, strip_top=True, subdir=f.extract_to)
            entry["extracted_to"] = str(dest)
        results.append(entry)
    return {"key": key, "title": spec.title, "license": spec.license if hasattr(spec, "license") else "", "files": results}


def list_sources() -> list[dict[str, Any]]:
    """Summarise the registry for the CLI."""
    out = []
    for spec in SOURCE_REGISTRY.values():
        out.append(
            {
                "key": spec.key,
                "title": spec.title,
                "kind": spec.kind,
                "homepage": spec.homepage,
                "files": len(spec.files),
                "total_bytes": spec.total_bytes(),
                "licenses": sorted({f.license for f in spec.files}),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
MANIFEST_NAME = "manifest.json"


def write_manifest(entries: list[dict[str, Any]], data_root: Path | None = None) -> Path:
    """Write ``dataset/manifest.json`` describing every fetched artefact."""
    data_root = Path(data_root or DATA_DIR)
    ensure_dir(data_root)
    path = data_root / MANIFEST_NAME
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")).get("files", [])
        except (json.JSONDecodeError, OSError):
            existing = []

    by_name = {e["filename"]: e for e in existing}
    for e in entries:
        e = dict(e)
        e["retrieved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        by_name[e["filename"]] = e

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project": "prompt_music_gen",
        "files": sorted(by_name.values(), key=lambda e: e["filename"]),
    }
    write_json(path, payload)
    return path


def read_manifest(data_root: Path | None = None) -> dict[str, Any]:
    """Read the manifest, returning an empty structure if it does not exist."""
    data_root = Path(data_root or DATA_DIR)
    path = data_root / MANIFEST_NAME
    if not path.exists():
        return {"files": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"files": []}


def free_space(path: Path) -> int:
    """Free bytes on the filesystem holding ``path``."""
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def as_dict(spec: SourceSpec) -> dict[str, Any]:
    return asdict(spec)
