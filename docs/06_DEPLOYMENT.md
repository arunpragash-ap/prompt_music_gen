# 06 — Deployment: running this on another machine

This project has no server component. "Deployment" means: get it running
somewhere else — a second laptop, a Linux box, or a shared machine — and
package the trained model so the receiver does not need the training data.

---

## 1. Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | macOS 13+ / Linux | Windows works but is untested |
| Python | 3.10+ | 3.12 is what CI/dev used |
| RAM | 8 GB | 6 GB works for generation only |
| Disk | 3 GB (inference) / 12 GB (full training data) | see §4 |
| Accelerator | optional | Apple MPS, CUDA, or CPU |
| `fluidsynth` | recommended | `brew install fluid-synth` (macOS) / `apt install fluidsynth` |

Generation on CPU works; MPS (Apple silicon) is roughly 3–5× faster for the
decoder.

## 2. Fresh-machine setup

```bash
git clone <your-local-repo> prompt_music_gen     # or copy the folder
cd prompt_music_gen

uv venv --python 3.12 .venv
uv pip install -e ".[mac]"                        # macOS extras
# Linux: uv pip install -e ".[analysis,render,fluidsynth,logging]"

.venv/bin/python -m pytest -q -m "not slow"       # 200 tests, no data needed
.venv/bin/promptmusic info                        # hardware + data report
```

`promptmusic info` is the fastest way to confirm that torch sees MPS/CUDA, that
the soundfont is found, and that the dataset is present.

## 3. Moving a trained model (without the data)

The checkpoint contains the weights, decoder config, vocabulary and
`special_ids`. That is everything inference needs — **no MIDI files** required.

```bash
# on the source machine: produce a lean artifact
promptmusic export outputs/prompt_music_gen/checkpoints/best.pt -f pt --out model.pt
promptmusic export outputs/prompt_music_gen/checkpoints/best.pt -f torchscript
promptmusic inspect model.pt                       # sanity-check metadata

# copy just this file (tens of MB, not 1.7 GB)
scp model.pt other-host:/srv/promptmusic/
```

| Format | Size | Needs Python? | Use case |
|---|---|---|---|
| `pt` | smallest | yes (`torch`) | canonical; load with `load_exported` |
| `torchscript` | medium | no (LibTorch) | embed in C++/mobile |
| `onnx` | medium | no (ONNX Runtime) | cross-language serving |

The mT5 encoder is **not** baked into the export — the target machine downloads
it once (4.3 GB) or receives a copy of `.hf_cache/hub/models--google--mt5-base`.
Plan for that transfer explicitly; it is the largest single item.

## 4. What to copy vs. what to regenerate

| Directory | Copy? | Why |
|---|---|---|
| `src/`, `configs/`, `docs/`, `tests/`, `Makefile` | yes | the project |
| `outputs/.../best.pt` or exported model | yes | the trained weights |
| `.hf_cache/` | optional | 4.5 GB; regenerates on demand |
| `dataset/` | no | 5.5 GB; `promptmusic download` |
| `data/downloads/`, `data/cache/` | no | regenerable |
| `data/processed/*.jsonl` | optional | small; speeds up a retrain |

A typical transfer is therefore **< 200 MB** plus the mT5 cache if the target
is offline.

## 5. Configuration overrides for a different machine

```bash
# CPU-only box
promptmusic generate -o train.device=cpu --prompt "..."

# less RAM
promptmusic train -c ta_love_test -o tokenizer.max_tokens=768 -o train.batch_size=1

# relocate data/outputs
export PROMPTMUSIC_DATA=/mnt/music/midicaps
export PROMPTMUSIC_OUTPUT=/mnt/music/outputs
export PROMPTMUSIC_HOME=/srv/prompt_music_gen
```

Path precedence and every environment variable are documented at the top of
`src/promptmusic/paths.py`.

## 6. Reproducibility

* Seeds: `generate.seed`, `project.seed`, or `PROMPTMUSIC_SEED` (wins over all).
* Determinism: `-o project.deterministic=true` (slower; not guaranteed on MPS).
* Every checkpoint records `model_config`, `special_ids`, `metrics` and step
  count — `promptmusic inspect` prints them.
* Configs are plain YAML and are copied into checkpoints, so an old model can
  always be regenerated with its original settings.

## 7. Deployment gotchas

1. **Licences travel with the model.** MidiCaps + MMS-TTS are non-commercial;
   see [05_MODELS_AND_LICENSES.md](05_MODELS_AND_LICENSES.md).
2. **Soundfonts differ per machine.** Pin one with `render.soundfont` so audio
   is reproducible.
3. **mT5 cache mismatch** is the classic failure: `cache_dir=.../hub` points at
   the directory *containing* the model dirs, whereas `HF_HOME=...` stores
   under `.../hub`. `paths.py` sets `HF_HOME` once, so leave it alone.
4. **`max_len` is baked into the checkpoint.** Generating longer than the
   trained length still works (sinusoidal extrapolation) but quality degrades.
5. **fluidsynth missing** silently falls back to a sine-wave renderer — check
   the `backend` field in the generate summary.
