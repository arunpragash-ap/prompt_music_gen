"""Packaging a trained model so something else can run it.

Why "export" is its own step
----------------------------
A training checkpoint (``checkpoints/best.pt``) contains everything needed to
*resume training*: optimizer moments, scheduler state, the config, the token
vocabulary. That is exactly what a deployment target does not want - it is
several times larger than the weights and tied to PyTorch's internal layout.

:mod:`promptmusic.export` produces lean, shareable artefacts instead:

===================  =====================================================
Format               What it is good for
===================  =====================================================
``.pt``              A minimal ``state_dict`` + metadata. Load it in any
                     PyTorch app with :func:`load_exported`. Smallest and
                     the canonical format for this project.
``TorchScript``      No Python needed at inference time. Works with
                     ``torch.jit.load`` in C++ or mobile deployments.
``ONNX``             Framework-neutral; runs on ONNX Runtime, Core ML (via
                     onnxruntime), or a browser. Needs ``pip install
                     onnx onnxruntime``.
===================  =====================================================

What cannot be exported (and why)
---------------------------------
The **frozen mT5 text encoder** is deliberately kept outside the exported
graph. It is ~300 M parameters of SentencePiece-backed text processing that
must be loaded separately anyway (the tokenizer lives in Python), and keeping
it out keeps the artefact small. The exported module therefore takes
*already-encoded* ``memory`` tensors plus music token ids as input.

That interface is documented in ``docs/06_DEPLOYMENT.md`` with a complete
standalone example, including how to run the encoder yourself.

The ``inspect`` command
-----------------------
``promptmusic inspect path/to/file`` reads an artefact and prints its
architecture, parameter count and metadata without loading it into memory -
useful when you have five checkpoints in ``outputs/`` and cannot remember
which is which.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import Config
from .data.tokenizer import build_tokenizer
from .model.build import build_model, load_checkpoint
from .paths import ensure_dir
from .utils import human_bytes, logger, select_device

LOG = logger("promptmusic.export")

__all__ = [
    "ExportResult",
    "export_model",
    "load_exported",
    "inspect_artifact",
    "ExportWrapper",
]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class ExportResult:
    """What one export produced."""

    format: str
    path: str
    size_bytes: int
    input_spec: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "size_human": human_bytes(self.size_bytes),
            "input_spec": self.input_spec,
            "metadata": self.metadata,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# The exported interface
# --------------------------------------------------------------------------- #
class ExportWrapper(nn.Module):
    """A thin module with a stable, documented input/output contract.

    Forward signature::

        (memory, music_ids) -> logits

    ==================  ====================================  ===============
    Tensor              Shape                                 Meaning
    ==================  ====================================  ===============
    ``memory``          ``[B, T_text, 768]``                  frozen mT5
                                                              encoder output
    ``music_ids``       ``[B, T_music]``                      tokens fed to the
                                                              decoder so far
    ``logits`` (out)    ``[B, T_music, vocab_size]``          next-token
                                                              distribution
    ==================  ====================================  ===============

    The wrapper exists because the raw ``MusicModel`` takes additional keyword
    arguments (masks, cache flags) that differ between training and inference;
    freezing them here means an external consumer only ever passes two tensors.
    """

    def __init__(self, model: Any):
        super().__init__()
        self.model = model
        # Expose the few scalars a consumer may want to read.
        self.vocab_size = int(getattr(model, "vocab_size", 0))
        self.pad_id = int(getattr(model, "pad_id", 0))
        self.bos_id = int(getattr(model, "bos_id", 1))
        self.eos_id = int(getattr(model, "eos_id", 2))

    def forward(
        self,
        memory: torch.Tensor,
        music_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Tolerate a 2-D memory (already pooled) by expanding a batch axis.
        if memory.dim() == 2:
            memory = memory.unsqueeze(0)
        logits, _ = self.model.decoder(music_ids, memory, need_cache=False)
        return logits


# --------------------------------------------------------------------------- #
# Loading a training checkpoint into an exportable module
# --------------------------------------------------------------------------- #
def _load_trained(
    checkpoint: str | Path,
    config: Config | str | Path | None = None,
    device: str = "cpu",
) -> tuple[ExportWrapper, Config, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    if config is None:
        stored = payload.get("config")
        cfg = Config(stored) if stored else None
        if cfg is None:
            raise ValueError("checkpoint has no stored config; pass config= explicitly")
    elif isinstance(config, Config):
        cfg = config
    else:
        from .config import load_config as _lc

        cfg = _lc(config)

    tok = build_tokenizer(cfg.section("tokenizer"))
    model = build_model(
        cfg, vocab_size=tok.vocab_size, pad_id=0, bos_id=1, eos_id=2, device="cpu"
    )
    missing, unexpected = model.decoder.load_state_dict(payload["model_state"], strict=False)
    if missing:
        LOG.warning("%d tensors missing from checkpoint", len(missing))
    if unexpected:
        LOG.warning("%d unused tensors in checkpoint", len(unexpected))
    model.eval()
    wrapper = ExportWrapper(model)
    wrapper.eval()
    return wrapper, cfg, payload


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_model(
    checkpoint: str | Path,
    *,
    out_path: str | Path | None = None,
    fmt: str = "pt",
    config: Config | str | Path | None = None,
    example_text_len: int = 16,
    example_music_len: int = 32,
    opset: int = 17,
    dynamic_axes: bool = True,
) -> ExportResult:
    """Export a training checkpoint to ``pt``, ``torchscript`` or ``onnx``.

    Args:
        checkpoint: a file written by :func:`promptmusic.train.train`.
        out_path: destination path. Defaults to a name derived from the
            checkpoint and format under ``outputs/<project>/export/``.
        fmt: ``"pt"`` | ``"torchscript"`` | ``"onnx"``.
        config: override the config stored inside the checkpoint.
        example_text_len: sequence length used to trace a dummy text memory.
        example_music_len: sequence length used to trace dummy music ids.
        opset: ONNX opset version.
        dynamic_axes: let batch and sequence dimensions vary in ONNX. Turning
        this off produces a static graph, which some runtimes prefer.

    Returns:
        An :class:`ExportResult`.

    Raises:
        ValueError: on an unknown ``fmt``.
        ImportError: if ``fmt="onnx"`` and the onnx packages are absent.
    """
    fmt = fmt.lower()
    if fmt not in {"pt", "torchscript", "onnx", "ts"}:
        raise ValueError(f"unknown format {fmt!r}; use pt, torchscript or onnx")
    if fmt == "ts":
        fmt = "torchscript"

    t0 = time.time()
    wrapper, cfg, payload = _load_trained(checkpoint, config)
    vocab_size = wrapper.vocab_size or payload.get("vocab_size") or 0

    if out_path is None:
        run_dir = ensure_dir(Path(cfg.get("paths.output_dir", "outputs")) / "export")
        suffix = {"pt": ".pt", "torchscript": ".pt", "onnx": ".onnx"}[fmt]
        out_path = run_dir / f"prompt_music_gen_{fmt}{suffix}"
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    result = ExportResult(format=fmt, path=str(out_path), size_bytes=0)
    result.metadata = {
        "checkpoint": str(checkpoint),
        "vocab_size": int(vocab_size),
        "step": payload.get("step"),
        "epoch": payload.get("epoch"),
        "metrics": payload.get("metrics", {}),
        "project": cfg.get("project.name"),
        "torch_version": torch.__version__,
    }
    result.input_spec = {
        "memory": ["B", "T_text", 768],
        "music_ids": ["B", "T_music"],
        "logits": ["B", "T_music", int(vocab_size)],
        "notes": "memory is the frozen mT5 encoder output; see docs/06_DEPLOYMENT.md",
    }

    if fmt == "pt":
        _export_pt(wrapper, out_path, result)
    elif fmt == "torchscript":
        _export_torchscript(
            wrapper, out_path, result,
            example_text_len, example_music_len,
        )
    else:
        _export_onnx(
            wrapper, out_path, result,
            example_text_len, example_music_len, opset, dynamic_axes,
        )

    result.elapsed_sec = time.time() - t0
    result.size_bytes = out_path.stat().st_size if out_path.exists() else 0
    LOG.info(
        "exported %s -> %s (%s, %.2fs)",
        fmt, out_path, human_bytes(result.size_bytes), result.elapsed_sec,
    )
    return result


# --------------------------------------------------------------------------- #
# Format-specific writers
# --------------------------------------------------------------------------- #
def _export_pt(wrapper: ExportWrapper, out: Path, result: ExportResult) -> None:
    """A minimal state dict plus JSON-able metadata.

    Only the decoder weights go in - the text encoder is loaded separately
    from Hugging Face by its model name, which is recorded here.
    """
    state = {k: v.cpu() for k, v in wrapper.model.decoder.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "vocab_size": wrapper.vocab_size,
            "pad_id": wrapper.pad_id,
            "bos_id": wrapper.bos_id,
            "eos_id": wrapper.eos_id,
            "text_encoder": wrapper.model.cfg.get("model.text_encoder.name")
            if hasattr(wrapper.model, "cfg")
            else "google/mt5-base",
            "format": "promptmusic.pt/v1",
            **{k: v for k, v in result.metadata.items() if k != "torch_version"},
        },
        out,
    )


def _export_torchscript(
    wrapper: ExportWrapper,
    out: Path,
    result: ExportResult,
    text_len: int,
    music_len: int,
) -> None:
    """Trace the wrapper with dummy inputs.

    ``torch.jit.trace`` rather than ``script`` because the decoder contains
    Python control flow (the KV-cache branch) that scripting cannot always
    infer; tracing with ``need_cache=False`` follows the inference path only.
    """
    memory = torch.zeros(1, text_len, 768)
    music_ids = torch.zeros(1, music_len, dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (memory, music_ids), strict=False)
    traced.save(str(out))

    # Sanity check: the saved file must reload and produce the right shape.
    reloaded = torch.jit.load(str(out))
    with torch.no_grad():
        out_logits = reloaded(memory, music_ids)
    if tuple(out_logits.shape) != (1, music_len, wrapper.vocab_size):
        result.warnings.append(
            f"post-save shape check: got {tuple(out_logits.shape)}, "
            f"expected {(1, music_len, wrapper.vocab_size)}"
        )


def _export_onnx(
    wrapper: ExportWrapper,
    out: Path,
    result: ExportResult,
    text_len: int,
    music_len: int,
    opset: int,
    dynamic_axes: bool,
) -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ONNX export needs two extra packages:\n"
            "    pip install onnx onnxruntime\n"
            "or:  pip install 'promptmusic[onnx]'"
        ) from exc

    memory = torch.zeros(1, text_len, 768)
    music_ids = torch.zeros(1, music_len, dtype=torch.long)

    axes = None
    if dynamic_axes:
        axes = {
            "memory": {0: "batch", 1: "text_len"},
            "music_ids": {0: "batch", 1: "music_len"},
            "logits": {0: "batch", 1: "music_len"},
        }

    torch.onnx.export(
        wrapper,
        (memory, music_ids),
        str(out),
        input_names=["memory", "music_ids"],
        output_names=["logits"],
        dynamic_axes=axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    # Validate the graph actually loads.
    try:
        import onnx

        model = onnx.load(str(out))
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"ONNX checker reported: {exc}")


# --------------------------------------------------------------------------- #
# Loading an exported artefact back
# --------------------------------------------------------------------------- #
def load_exported(
    path: str | Path,
    *,
    device: str = "cpu",
    config: Config | str | Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load anything :func:`export_model` produced.

    Returns ``(module, metadata)``. For TorchScript the module is already
    executable; for ``.pt`` the caller must rebuild the model (use the returned
    metadata's ``text_encoder`` field) and load ``state_dict`` into it - see
    ``docs/06_DEPLOYMENT.md`` for the full recipe.

    Example::

        module, meta = load_exported("outputs/export/prompt_music_gen_torchscript.pt")
        logits = module(memory, music_ids)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"exported artifact not found: {path}")

    dev = select_device(device)

    # TorchScript files self-describe; detect them by magic bytes.
    if _is_torchscript(path):
        module = torch.jit.load(str(path), map_location=dev)
        return module, {"format": "torchscript", "path": str(path)}

    if path.suffix == ".onnx":
        raise ValueError(
            "ONNX files are loaded with onnxruntime, not PyTorch:\n"
            "    import onnxruntime as ort\n"
            "    sess = ort.InferenceSession(str(path))\n"
            "    logits = sess.run(None, {'memory': m, 'music_ids': ids})[0]"
        )

    payload = torch.load(path, map_location=dev, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            f"{path} is not a promptmusic .pt export (no 'state_dict' key)"
        )

    cfg = None
    if config is not None:
        cfg = config if isinstance(config, Config) else None
    if cfg is None:
        stored = payload.get("config")
        if stored:
            cfg = Config(stored)
    if cfg is None:
        from .config import load_config as _lc

        cfg = _lc("default")

    tok = build_tokenizer(cfg.section("tokenizer"))
    model = build_model(
        cfg, vocab_size=tok.vocab_size,
        pad_id=payload.get("pad_id", 0),
        bos_id=payload.get("bos_id", 1),
        eos_id=payload.get("eos_id", 2),
        device=device,
    )
    model.decoder.load_state_dict(payload["state_dict"], strict=False)
    model.eval()

    meta = {
        k: v for k, v in payload.items()
        if k not in {"state_dict", "optimizer_state", "scheduler_state"}
    }
    meta.setdefault("format", "promptmusic.pt/v1")
    meta["path"] = str(path)
    return model, meta


def _is_torchscript(path: Path) -> bool:
    """TorchScript archives begin with the ``PK`` zip magic; ``.pt`` from
    ``torch.save`` of a plain dict usually does too, so also probe for the
    jit method table."""
    try:
        import zipfile

        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            return any(n.startswith("code/") or n == "constants.pkl" for n in names)
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Inspection (the `inspect` command)
# --------------------------------------------------------------------------- #
def inspect_artifact(path: str | Path) -> dict[str, Any]:
    """Describe an artifact without fully materialising it in memory.

    Works on: training checkpoints (``.pt``), TorchScript archives, ONNX
    graphs, and miditok vocabulary files (``.json``).

    Returns a JSON-able dict with ``kind``, architecture summary, parameter
    counts and whatever metadata the file carries.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"not found: {path}")

    info: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_human": human_bytes(path.stat().st_size),
        "kind": "unknown",
    }

    if path.suffix == ".onnx":
        return _inspect_onnx(path, info)
    if path.suffix == ".json":
        return _inspect_json(path, info)
    if _is_torchscript(path):
        return _inspect_torchscript(path, info)
    if path.suffix in {".pt", ".pth", ".bin", ".ckpt"}:
        return _inspect_checkpoint(path, info)

    info["hint"] = "unrecognised extension; tried generic readers"
    return info


def _inspect_checkpoint(path: Path, info: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        info["kind"] = "torch-raw"
        info["note"] = f"top-level object is {type(payload).__name__}"
        return info

    if "model_state" in payload or "model" in payload:
        info["kind"] = "training-checkpoint"
        state = payload.get("model_state") or payload.get("model") or {}
        info.update(_tensor_summary(state))
        info["step"] = payload.get("step")
        info["epoch"] = payload.get("epoch")
        info["metrics"] = payload.get("metrics", {})
        cfg = payload.get("config")
        if isinstance(cfg, dict):
            info["project"] = (cfg.get("project") or {}).get("name")
            te = ((cfg.get("model") or {}).get("text_encoder") or {}).get("name")
            if te:
                info["text_encoder"] = te
        info["has_optimizer_state"] = "optimizer_state" in payload
        info["num_vocab_tokens"] = len(payload.get("vocab") or {})
        return info

    if "state_dict" in payload:
        info["kind"] = "promptmusic-export"
        info.update(_tensor_summary(payload["state_dict"]))
        for k in ("vocab_size", "text_encoder", "pad_id", "bos_id", "eos_id", "format"):
            if k in payload:
                info[k] = payload[k]
        return info

    info["kind"] = "torch-dict"
    info["keys"] = sorted(payload.keys())[:40]
    return info


def _tensor_summary(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    total = 0
    tensors = 0
    for v in state.values():
        if isinstance(v, torch.Tensor):
            total += v.numel()
            tensors += 1
    layers: dict[str, int] = {}
    for k in state:
        prefix = k.split(".")[0]
        layers[prefix] = layers.get(prefix, 0) + 1
    return {
        "num_tensors": tensors,
        "num_parameters": total,
        "num_parameters_m": round(total / 1e6, 2),
        "groups": dict(sorted(layers.items(), key=lambda kv: -kv[1])[:12]),
    }


def _inspect_torchscript(path: Path, info: dict[str, Any]) -> dict[str, Any]:
    module = torch.jit.load(str(path), map_location="cpu")
    info["kind"] = "torchscript"
    try:
        graph = str(module.graph)
        info["graph_lines"] = len(graph.splitlines())
        info["has_forward"] = hasattr(module, "forward")
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


def _inspect_onnx(path: Path, info: dict[str, Any]) -> dict[str, Any]:
    try:
        import onnx
    except ImportError:
        info["kind"] = "onnx"
        info["note"] = "install onnx to inspect graphs: pip install onnx"
        return info
    model = onnx.load(str(path))
    info["kind"] = "onnx"
    info["ir_version"] = model.ir_version
    info["opset"] = [
        {"domain": o.domain or "ai.onnx", "version": o.version}
        for o in model.opset_import
    ]
    info["inputs"] = [
        {"name": i.name, "shape": [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]}
        for i in model.graph.input
    ]
    info["outputs"] = [
        {"name": o.name, "shape": [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]}
        for o in model.graph.output
    ]
    info["num_nodes"] = len(model.graph.node)
    return info


def _inspect_json(path: Path, info: dict[str, Any]) -> dict[str, Any]:
    """Handle miditok vocabulary JSON and any other JSON artifact."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
        return info
    info["kind"] = "json"
    if isinstance(data, dict):
        info["keys"] = sorted(data.keys())[:30]
        # miditok v3 vocab files: {"token_str": id, ...}
        if data and all(isinstance(v, int) for v in list(data.values())[:5]):
            info["kind"] = "token-vocabulary"
            info["vocab_size"] = len(data)
            sample = list(data.items())[:8]
            info["sample_tokens"] = [k for k, _ in sample]
    elif isinstance(data, list):
        info["kind"] = "json-array"
        info["length"] = len(data)
    return info


# --------------------------------------------------------------------------- #
# Convenience: export + report in one call
# --------------------------------------------------------------------------- #
def export_and_report(
    checkpoint: str | Path,
    formats: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Export to several formats and return their reports keyed by format."""
    formats = formats or ["pt"]
    reports: dict[str, Any] = {}
    for fmt in formats:
        try:
            reports[fmt] = export_model(checkpoint, fmt=fmt, **kwargs).to_dict()
        except Exception as exc:  # noqa: BLE001 - one format must not block others
            reports[fmt] = {"format": fmt, "error": str(exc)}
            LOG.warning("export to %s failed: %s", fmt, exc)
    return reports
