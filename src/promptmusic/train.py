"""Training the model.

What actually happens in one training step
------------------------------------------
1. **Tokenise the prompts.** A batch of caption strings goes through the mT5
   SentencePiece tokenizer, producing ``input_ids`` of shape ``[B, T_text]``.
2. **Encode the prompts.** The *frozen* mT5 encoder turns those ids into a
   matrix of vectors, ``memory`` of shape ``[B, T_text, 768]``. This happens
   under ``torch.no_grad()``, so it costs memory but no gradients.
3. **Teacher forcing.** The music token sequence ``[BOS, t1, ..., tN, EOS]``
   is fed to the decoder with its last token removed; the decoder predicts
   every *next* token in one parallel pass.
4. **Cross-entropy loss.** Compare the predicted distribution at each position
   with the token that actually follows, ignoring padded positions.
5. **Backpropagate** through the decoder only, clip the gradient, and take an
   AdamW step with a warmup-then-cosine learning rate.

Memory notes for an 8 GB Apple M2
---------------------------------
The frozen encoder is ~295 M parameters (~1.2 GB in float32) and is the single
biggest consumer. The decoder is ~60 M parameters (~240 MB) plus its AdamW
moments (~480 MB). Everything else is activations. The defaults here
(``batch_size=8``, ``max_tokens=1024``, ``num_workers=0``) fit with headroom.
If you run out of memory, lower ``train.batch_size`` and raise
``train.grad_accum`` by the same factor to keep the effective batch size.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import Config, load_config
from .data.dataset import (
    MusicDataset,
    build_tokenized_dataset,
    collate_fn,
    split_examples,
)
from .data.tokenizer import build_tokenizer
from .model.build import build_model, count_parameters, load_checkpoint, save_checkpoint
from .model.text_encoder import TextTokenizer
from .paths import OUTPUTS_DIR, ensure_dir
from .utils import format_duration, logger, select_device, set_seed

LOG = logger("promptmusic.train")

__all__ = ["train", "TrainLogger", "make_scheduler"]


# --------------------------------------------------------------------------- #
# CSV logging
# --------------------------------------------------------------------------- #
class TrainLogger:
    """Append-only CSV logger.

    A CSV file is deliberately boring: it opens in a spreadsheet, it plots in
    three lines of matplotlib, and it survives a crash because every row is
    flushed as soon as it is written.
    """

    FIELDS = [
        "step", "epoch", "split", "loss", "perplexity", "lr",
        "tokens_per_sec", "elapsed_sec", "grad_norm",
    ]

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS, extrasaction="ignore")
        if write_header:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, **row: Any) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # pragma: no cover - already closed
            pass


# --------------------------------------------------------------------------- #
# Optimiser / scheduler
# --------------------------------------------------------------------------- #
def _split_params(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Separate weights from biases/norms so weight decay is applied fairly.

    Standard practice: decay the matrices, leave the biases and the LayerNorm
    gains alone. Decaying a normalisation gain pushes it towards zero, which
    quietly damages the network.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = _split_params(model)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup: int,
    num_training: int,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by a cosine decay to ``min_lr_ratio * lr``.

    Warmup matters because at step 0 the decoder's weights are random and a
    full-sized update would destroy whatever structure the initialisation
    gave it. Cosine decay then lets the model settle into a minimum.
    """

    def lr_lambda(step: int) -> float:
        if num_warmup > 0 and step < num_warmup:
            return (step + 1) / num_warmup
        if num_training <= num_warmup:
            return 1.0
        progress = (step - num_warmup) / max(1, num_training - num_warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #
def train(
    config: Config | str | Path | None = None,
    *,
    overrides: list[str] | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train the model and return a summary dict.

    Args:
        config: a :class:`~promptmusic.config.Config`, a config name
            (``"smoke"``, ``"default"``) or a path to a YAML file.
        overrides: ``key.path=value`` strings applied on top of the config.
        run_name: subdirectory of ``outputs/`` for this run's artefacts.
    """
    if not isinstance(config, Config):
        cfg = load_config(config, overrides)
    else:
        cfg = config
        cfg.apply_overrides(overrides)

    tcfg = cfg.section("train")
    run_name = run_name or cfg.get("project.name", "prompt_music_gen")
    run_dir = ensure_dir(Path(cfg.get("paths.output_dir", OUTPUTS_DIR)) / run_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    log_path = run_dir / "train_log.csv"

    set_seed(int(cfg.get("project.seed", 42)), bool(cfg.get("project.deterministic", False)))

    device = select_device(tcfg.get("device", "auto"))
    LOG.info("device: %s", device)
    if device == "mps":
        LOG.info("Apple MPS selected - float64 and some ops are unavailable on this device")

    # --- data ------------------------------------------------------------- #
    tok_cfg = cfg.section("tokenizer")
    languages = cfg.get("data.languages", ["en"]) or ["en"]
    language = "ta" if "ta" in languages and cfg.get("data.use_tamil_for_training") else "en"

    music_tokenizer = build_tokenizer(tok_cfg)
    vocab_size = music_tokenizer.vocab_size
    LOG.info("music vocabulary: %d tokens", vocab_size)

    examples = build_tokenized_dataset(
        cfg, music_tokenizer, language=language, limit=cfg.get("data.limit")
    )
    if not examples:
        raise RuntimeError(
            "no training examples were built. Did the MIDI files download and extract?\n"
            "Try:  promptmusic download --source midicaps"
        )
    train_examples, val_examples = split_examples(
        examples,
        val_fraction=float(cfg.get("data.val_fraction", 0.05)),
        seed=int(cfg.get("project.seed", 42)),
    )

    max_tokens = int(tok_cfg.get("max_tokens", 4096))
    pad_id, bos_id, eos_id = 0, 1, 2

    def _make_loader(rows, shuffle: bool) -> DataLoader:
        ds = MusicDataset(rows, max_tokens=max_tokens)
        return DataLoader(
            ds,
            batch_size=int(tcfg.get("batch_size", 8)),
            shuffle=shuffle,
            num_workers=int(tcfg.get("num_workers", 0)),
            collate_fn=lambda b: collate_fn(
                b, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id, max_tokens=max_tokens
            ),
            drop_last=False,
        )

    train_loader = _make_loader(train_examples, shuffle=True)
    val_loader = _make_loader(val_examples, shuffle=False) if val_examples else None

    # --- model ------------------------------------------------------------ #
    model = build_model(
        cfg,
        vocab_size=vocab_size,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        device=device,
    )
    n_total = count_parameters(model)
    n_train = count_parameters(model, trainable_only=True)
    LOG.info("parameters: %.1fM total, %.1fM trainable", n_total / 1e6, n_train / 1e6)

    text_tok = TextTokenizer(
        model_name=cfg.get("model.text_encoder.name", "google/mt5-base"),
        max_length=int(cfg.get("model.text_encoder.max_length", 128)),
    )

    # --- optimiser --------------------------------------------------------- #
    epochs = int(tcfg.get("epochs", 3))
    grad_accum = max(1, int(tcfg.get("grad_accum", 1)))
    max_steps = tcfg.get("max_steps")
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * epochs
    if max_steps:
        total_steps = min(total_steps, int(max_steps))

    optimizer = make_optimizer(
        model, lr=float(tcfg.get("lr", 3e-4)), weight_decay=float(tcfg.get("weight_decay", 0.01))
    )
    warmup = int(total_steps * float(tcfg.get("warmup_ratio", 0.05)))
    scheduler = make_scheduler(
        optimizer, warmup, total_steps, float(tcfg.get("min_lr_ratio", 0.05))
    )

    # --- resume ------------------------------------------------------------ #
    start_epoch, global_step = 0, 0
    best_val = float("inf")
    resume = tcfg.get("resume")
    if resume:
        payload = load_checkpoint(resume, map_location=device)
        model.decoder.load_state_dict(payload["model_state"], strict=False)
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload.get("epoch", 0))
        global_step = int(payload.get("step", 0))
        best_val = float(payload.get("metrics", {}).get("best_val_loss", best_val))
        LOG.info("resumed from %s at epoch %d, step %d", resume, start_epoch, global_step)

    logger_csv = TrainLogger(log_path)
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
    log_every = int(tcfg.get("log_every", 50))
    val_every = int(tcfg.get("val_every", 500))
    save_every = int(tcfg.get("save_every", 500))

    model.train()
    t_start = time.time()
    history: list[dict[str, Any]] = []
    stopped_early = False

    LOG.info(
        "training: %d epochs, %d steps total, batch=%d x grad_accum=%d (effective %d)",
        epochs, total_steps, tcfg.get("batch_size", 8), grad_accum,
        int(tcfg.get("batch_size", 8)) * grad_accum,
    )

    for epoch in range(start_epoch, epochs):
        optimizer.zero_grad(set_to_none=True)
        running_loss, running_tokens, epoch_start = 0.0, 0, time.time()

        for batch_idx, batch in enumerate(train_loader):
            texts = batch["texts"]
            enc = text_tok(texts)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                memory = model.encode_prompt(input_ids, attention_mask)

            music_ids = batch["music_ids"].to(device)
            loss, _ = model.loss(music_ids, memory, memory_mask=attention_mask)
            (loss / grad_accum).backward()

            running_loss += float(loss.detach()) * music_ids.size(0)
            running_tokens += int(batch["music_mask"].sum())

            stepped = False
            if (batch_idx + 1) % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                stepped = True

                if global_step % log_every == 0:
                    n = max(1, batch_idx + 1)
                    avg = running_loss / n
                    elapsed = time.time() - t_start
                    row = {
                        "step": global_step,
                        "epoch": epoch,
                        "split": "train",
                        "loss": round(avg, 5),
                        "perplexity": round(math.exp(min(avg, 20)), 3),
                        "lr": f"{scheduler.get_last_lr()[0]:.3e}",
                        "tokens_per_sec": round(running_tokens / max(elapsed, 1e-6), 1),
                        "elapsed_sec": round(elapsed, 1),
                        "grad_norm": round(float(grad_norm), 4),
                    }
                    logger_csv.log(**row)
                    history.append(row)
                    LOG.info(
                        "epoch %d step %d/%d  loss %.4f  ppl %.1f  lr %.2e  %.0f tok/s",
                        epoch, global_step, total_steps, avg, math.exp(min(avg, 20)),
                        scheduler.get_last_lr()[0], running_tokens / max(elapsed, 1e-6),
                    )
                    running_loss, running_tokens, t_start = 0.0, 0, time.time()

                if val_loader is not None and global_step % val_every == 0:
                    val_loss = _validate(model, val_loader, text_tok, device)
                    model.train()
                    row = {"step": global_step, "epoch": epoch, "split": "val",
                           "loss": round(val_loss, 5),
                           "perplexity": round(math.exp(min(val_loss, 20)), 3)}
                    logger_csv.log(**row)
                    history.append(row)
                    LOG.info("  validation loss %.4f (ppl %.1f)", val_loss, math.exp(min(val_loss, 20)))
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(
                            ckpt_dir / "best.pt", model, optimizer=optimizer, scheduler=scheduler,
                            epoch=epoch, step=global_step,
                            metrics={"val_loss": val_loss, "best_val_loss": best_val},
                            config=cfg, vocab=music_tokenizer.vocab,
                        )

                if global_step % save_every == 0:
                    save_checkpoint(
                        ckpt_dir / "last.pt", model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch, step=global_step, metrics={"best_val_loss": best_val},
                        config=cfg, vocab=music_tokenizer.vocab,
                    )

                if max_steps and global_step >= int(max_steps):
                    stopped_early = True
                    break

        epoch_time = time.time() - epoch_start
        LOG.info("epoch %d finished in %s", epoch, format_duration(epoch_time))

        if val_loader is not None:
            val_loss = _validate(model, val_loader, text_tok, device)
            model.train()
            row = {"step": global_step, "epoch": epoch, "split": "val",
                   "loss": round(val_loss, 5), "perplexity": round(math.exp(min(val_loss, 20)), 3)}
            logger_csv.log(**row)
            history.append(row)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    ckpt_dir / "best.pt", model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, step=global_step,
                    metrics={"val_loss": val_loss, "best_val_loss": best_val},
                    config=cfg, vocab=music_tokenizer.vocab,
                )

        save_checkpoint(
            ckpt_dir / "last.pt", model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch + 1, step=global_step, metrics={"best_val_loss": best_val},
            config=cfg, vocab=music_tokenizer.vocab,
        )

        if stopped_early:
            LOG.info("reached max_steps=%d; stopping", max_steps)
            break

    logger_csv.close()

    final_ckpt = ckpt_dir / ("best.pt" if (ckpt_dir / "best.pt").exists() else "last.pt")
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(final_ckpt),
        "log": str(log_path),
        "steps": global_step,
        "epochs_run": epoch + 1 if "epoch" in dir() else 0,
        "best_val_loss": best_val if best_val != float("inf") else None,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "vocab_size": vocab_size,
        "params_total": n_total,
        "params_trainable": n_train,
        "device": device,
        "wall_time_sec": round(time.time() - (t_start - epoch_time) if "epoch_time" in dir() else 0, 1),
    }
    LOG.info("training complete: %s", summary)
    return summary


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _validate(model, loader, text_tok, device: str) -> float:
    """Mean cross-entropy over the validation set."""
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        enc = text_tok(batch["texts"])
        memory = model.encode_prompt(
            enc["input_ids"].to(device), enc["attention_mask"].to(device)
        )
        music_ids = batch["music_ids"].to(device)
        loss, _ = model.loss(music_ids, memory, memory_mask=enc["attention_mask"].to(device))
        total += float(loss) * music_ids.size(0)
        count += music_ids.size(0)
    return total / max(count, 1)
