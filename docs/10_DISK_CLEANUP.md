# 10 — Disk cleanup: what is big, what is safe to delete

This project downloads and generates a lot of regenerable data. This page tells
you exactly what everything is, what it costs, and how to reclaim it safely.

Run the built-in report first:

```bash
promptmusic disk-report            # aligned table of sizes + deletability
promptmusic disk-report --json     # machine-readable
./scripts/disk_report.sh           # same information, plain shell
```

---

## 1. Measured footprint (reference machine)

| Path | Size | What it is | Safe to delete? |
|---|---|---|---|
| `dataset/midicaps/` | **5.5 GB** | 168 385 extracted MIDI files | **yes** — `promptmusic download` |
| `data/downloads/midicaps.tar.gz` | **1.5 GB** | the archive you already extracted | **yes** — never needed again |
| `data/downloads/train.json` | **408 MB** | caption index (JSON Lines) | **yes** — re-download |
| `.hf_cache/` | **4.5 GB** | mT5-base (4.3 GB) + Tamil TTS (139 MB) | **yes** — auto re-download |
| `outputs/` | **3.2 GB** | checkpoints (2 × 1.7 GB), samples (27 MB), raga corpus (2.6 MB) | **yes** if you accept retraining |
| `.venv/` | **905 MB** | the Python environment | yes — recreate with `uv venv` |
| `data/processed/` | **5 MB** | token caches + subsets | yes — re-tokenise |
| `data/ragas/` | < 1 MB | the raga table (source, not data) | **no** — hand-written source |

Total regenerable: roughly **15 GB**.

## 2. Safe cleanup recipes

### Reclaim ~7 GB, keep the ability to retrain fast

```bash
rm -f data/downloads/midicaps.tar.gz        # 1.5 GB, extracted already
rm -rf outputs/prompt_music_gen/checkpoints/last.pt   # 1.7 GB older snapshot
rm -rf .hf_cache/hub/models--facebook--mms-tts-tam    # 139 MB, re-downloaded on demand
```

### Reclaim ~13 GB, keep the model but not the data

```bash
rm -rf dataset/ data/downloads/ data/cache/
rm -f data/processed/*.jsonl
```

Keep `outputs/.../best.pt` — it contains everything inference needs.

### Reclaim ~16 GB, keep nothing but the code

```bash
rm -rf dataset/ data/downloads/ data/cache/ data/processed/
rm -rf outputs/ .hf_cache/
# or:
make clean-all
```

Everything comes back with `promptmusic download` / `tokenize` / `train`.

## 3. What you must not delete

| Path | Why |
|---|---|
| `src/`, `tests/`, `configs/`, `docs/`, `scripts/` | the project itself |
| `data/ragas/raga_table.yaml` | hand-written raga knowledge |
| `data/lyrics/*.txt` | your lyric files |
| `outputs/.../best.pt` | your trained model (unless you plan to retrain) |
| `.git/` | history |

## 4. Makefile shortcuts

```bash
make disk-report     # show sizes and what is reclaimable
make clean-outputs   # outputs/, runs/, logs, stray audio files
make clean-data      # dataset/, downloads, cache, processed
make clean-all       # everything regenerable (asks nothing — be sure)
```

`clean-all` does **not** touch `src/`, `configs/`, `docs/`, `data/ragas/` or
`.git/`.

## 5. Why the caches exist

* **`.hf_cache`** — model downloads are expensive (4.3 GB mT5). `paths.py` sets
  `HF_HOME` once so every component shares one cache. Deleting it is safe but
  the next run re-downloads.
* **`data/processed`** — tokenising 168 k MIDI files takes hours; the JSONL
  cache makes retraining start in seconds. Keep it if you retrain often.
* **`outputs/.../last.pt`** — written every `save_every` steps; `best.pt` is
  kept on validation improvement. Deleting `last.pt` only loses resumability.

## 6. Checking before you delete

```bash
du -sh dataset/* .hf_cache/hub/models--* outputs/* 2>/dev/null
promptmusic disk-report
```

If a path is not listed in `paths.py`, it is not part of the project's managed
storage — inspect it manually before removing.

## 7. A note on swap

The reference machine (8 GB M2) showed 28–31 GB of swap in use even when idle,
which made long training runs slow. If `vm_stat` shows heavy swap during
training, the fix is less data in memory (shorter `tokenizer.max_tokens`),
not more disk cleanup. See [03_TRAINING.md](03_TRAINING.md) §5.
