# 03 — Training: from data to a checkpoint

Training is the expensive phase: the model looks at many (caption, music)
pairs and slowly adjusts its weights. This page covers the whole loop, the
memory maths for an 8 GB Apple M2, and the problems you are likely to hit.

---

## 1. Prerequisites

```bash
cd prompt_music_gen
uv venv --python 3.12 .venv
.venv/bin/python -m pip install -e ".[mac]"      # or: uv pip install -e ".[mac]"
.venv/bin/python -m pytest -q -m "not slow"      # should pass: 200 tests
```

You need the dataset (see [01_DATASETS.md](01_DATASETS.md)):

```bash
promptmusic download --source midicaps   # 5.5 GB, resumable
```

## 2. Step 1 — tokenise

```bash
promptmusic tokenize -c ta_love_test     # the 400-piece subset, seconds
# or, the whole corpus (hours, ~5-10 GB of cache):
promptmusic tokenize
```

The output is JSON Lines, one record per piece:
`{"caption": "...", "tokens": [...], "midi": "lmd_full/0/....mid"}`.
The `data.cache_file` config key lets a later `train` run reuse this file
instead of tokenising again.

## 3. Step 2 — choose a config

| Config | Purpose | Model | Data | Speed |
|---|---|---|---|---|
| `smoke` | prove the pipeline works | tiny | ~100 pieces | minutes |
| `ta_love_test` | end-to-end Tamil demo | 12 L / 512 | 400 love songs | ~2.5 s/step |
| `default` | normal training | 12 L / 512 | 20 k pieces | hours |
| `from_scratch` | full MidiCaps run | 12 L / 512 | 168 k pieces | many hours |
| `paper_size` | larger decoder | 18 L / 768 | 20 k pieces | needs ≥16 GB RAM |

Configs are plain YAML. Values can be overridden on the command line:

```bash
promptmusic train -c ta_love_test -o train.max_steps=500 -o train.batch_size=1
```

Precedence: CLI `-o key=value` > YAML file > built-in `DEFAULTS`.
`PROMPTMUSIC_SEED` overrides the seed everywhere.

## 4. Step 3 — train

```bash
promptmusic train -c smoke               # first: sanity check
promptmusic train -c ta_love_test        # the demo checkpoint
```

What you see: a loss line every `train.log_every` steps, validation loss every
`train.val_every` steps, and checkpoints under
`outputs/<project>/checkpoints/` (`last.pt` always, `best.pt` on improvement).
Resume with `-o train.resume=outputs/.../best.pt`.

### Reading the loss

* ~5.5 at step 0 — the model is guessing uniformly.
* ~3.4 — early structure: bars, common instruments.
* < 1.0 — it reproduces the training distribution well.
* Falling train loss with rising validation loss — overfitting; stop earlier or
  add data.

## 5. Memory maths for an 8 GB M2

The limiting factor is not the weights (328 M ≈ 1.3 GB) but the **activations**
kept for backpropagation, which grow with sequence length.

| Tokens per example | Approx. peak | Verdict on 8 GB |
|---|---|---|
| 512 | ~3 GB | comfortable |
| 1 024 | ~5–6 GB | works (demo setting) |
| 2 048 | > 8 GB | swaps heavily — avoid |

Additional settings that matter:

* `train.batch_size: 1` + `train.grad_accum: 2` — keeps the effective batch at
  2 while only holding one example's activations at a time.
* `tokenizer.max_tokens: 1024` — truncates long pieces at tokenise time.
* `train.amp` is not used on MPS; keep it `false`.
* Free RAM before a long run. Close browsers/IDEs; check `vm_stat` and
  Activity Monitor. The demo machine ran with ~28–31 GB of swap in use even
  when idle (an unrelated process), which is why 2 048-token training
  thrashed.

## 6. What the demo checkpoint actually is

`outputs/prompt_music_gen/checkpoints/best.pt` was trained with
`configs/ta_love_test.yaml`:

* 400 love-song pieces from MidiCaps (300 romantic + 100 general)
* 1 024-token examples, batch 1, gradient accumulation 2
* 100 steps (killed at 120 when swap pressure made it slow)
* validation loss **3.39**
* 1.7 GB file (weights + optimizer state + config + vocabulary)

It is a *demonstration*: enough to prove every stage of the pipeline on real
data, not enough to produce polished music.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Step time 10× slower than expected | swap / another heavy process | free RAM; reduce `max_tokens`; restart |
| `mT5` "downloading" for minutes | HF cache mismatch | the project sets `HF_HOME` automatically; check `.hf_cache/hub` exists |
| Loss stuck at ~5.5 | learning rate too low or data broken | inspect a tokenised record; try `-o train.lr=1e-3` for a smoke run |
| `MPS backend out of memory` | sequence too long | `-o tokenizer.max_tokens=768`, batch 1 |
| Checkpoint won't load | saved with a different vocab | checkpoints store `special_ids` and `model_config`; use `promptmusic inspect` |

## 8. Beyond the demo

To get qualitatively better music you need **more steps at the length you
will generate**, not just more parameters. Practical upgrade path:

1. Train the full 400-piece subset for 2 000–5 000 steps at 1 024 tokens.
2. Train at 2 560 tokens (the generation length) on a machine with ≥16 GB.
3. Scale to the full 168 k corpus with `from_scratch.yaml`.

The model's positional encoding extrapolates beyond its training length — see
[02_ARCHITECTURE.md](02_ARCHITECTURE.md) §3 — so matching train and generate
lengths matters more than raw dataset size.
