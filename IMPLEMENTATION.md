# IMPLEMENTATION — how this project was built

A build log: what was built, in what order, what broke, and the measured
numbers behind the decisions. This is the document to read if you want to know
*why* something is the way it is. It is deliberately a working notebook, not
marketing.

---

## 1. Build order

The project was built in phases. Each phase was committed separately; the
history reads as a tutorial.

| # | Phase | What it added |
|---|---|---|
| 1 | Data + tokens | MidiCaps download, REMI tokenizer, caption index, token cache |
| 2 | Translation | English captions → Tamil with IndicTrans2 (bilingual prompts) |
| 3 | Lyrics | Lyric corpus, vocabulary, prompt fold-in (token-level interleaving scaffolded) |
| 4 | Reference | `analyze` a MIDI/audio file, condition or prime generation on it |
| 5 | Ragas | Hand-written Hindustani raga table → synthetic MIDI corpus |
| 6 | Audit | Instrument distribution report + rebalancing weights |
| 7 | Model | Frozen mT5 encoder → projection → 12-layer decoder |
| 8 | Train | Loop, scheduler, validation, checkpoints, memory tuning |
| 9 | Generate/Render | Sampling → MIDI → fluidsynth WAV |
| 10 | Vocals | Lyric vocal layer (added after the first end-to-end demo) |
| 11 | Docs | This documentation set |

## 2. Git history (local only, never pushed)

```
3c5484b Initial commit: prompt_music_gen text-to-music pipeline
24d37bc Fix HF cache wiring, CLI download kwargs, tokenizer lazy-vocab + lyric interleave
bd7116b Fix MidiCaps caption loading: JSONL format, lmd_full/ paths, list-typed genre/mood
bc9bbdf Tamil love-song 60s test: config, love-subset captions, cache reuse, density calibration
addf355 CLI: render/evaluate summaries print aligned key: value instead of raw dicts
a00b248 Right-size ta_love_test for 8 GB: train at 1024 tokens (2048 thrashed swap)
4331ec2 Fix two generation-path bugs: 1-D tokenizer output, checkpoint loaded into wrong module
4df1936 Fix multinomial kwarg typo in sampler (num= -> num_samples=); cover all sampling paths
3c6c2f5 vocals: speak lyrics over the instrumental + fix 0-BPM tempo bug
```

`scripts/init_repo.sh --branches` creates the `phase-*` branch layout for
anyone who wants branch-per-phase development.

## 3. Decisions worth recording

**Frozen mT5-base as the text encoder.** A trained text tower would need ~4 GB
of optimizer state — impossible on an 8 GB machine used for other things. The
cost is that language is not adapted to music.

**12 layers × 512 dim (51 M trainable).** `configs/paper_size.yaml` (18 × 768)
exists for larger machines. The small decoder is the difference between
"trains on a laptop" and "does not".

**REMI tokens, not audio tokens.** MIDI is compact, debuggable and editable;
the price is that timbre comes from a soundfont.

**`tokens_per_second` default changed 12 → 40.** Measured densities: raga
corpus ~12 tok/s, MidiCaps love subset ~40 median, MidiCaps overall ~60.
`--duration` sizes the request from this number.

**Train at 1 024 tokens, generate at 2 560.** Training at 2 048 thrashed swap
(~60 s/step, measured) and was killed. Positions above the training length are
extrapolated by the sinusoidal encoding — acceptable for a pipeline test,
documented as a caveat.

**Lyrics fold into the prompt.** `"{prompt}\n{Tamil|English} lyrics: {lyrics}"`
gives the mT5 encoder the words. Token-level interleaving (Phase 3) is
scaffolded but not enabled.

## 4. Measured numbers

| Measurement | Value |
|---|---|
| Decoder parameters | 51.3 M |
| Total model | 328.3 M |
| Music vocabulary | 423 tokens (`pad=0, bos=1, eos=2`) |
| Special ids in checkpoint | confirmed |
| Training, 1 024 tokens, batch 1 × accum 2 | ~2.5–3 s/step |
| Training, 2 048 tokens | thrashed swap; killed |
| Loss at step 0 → step 120 | 5.49 → 0.27 |
| Validation loss at step 100 | 3.39 |
| Checkpoint size | 1.7 GB (with optimizer state) |
| Generation speed (MPS) | ~64 tokens/s (2 559 tokens ≈ 40 s) |
| Tamil love song | 78.0 s MIDI, 516 notes, 18 instruments, 81.6 s WAV |
| Evaluation of that song | validity 100 %, 6.61 notes/s, 12 pitch classes, entropy 3.43 bits, repetition 1.8 %, CR 0.96 |
| Full MidiCaps | 168 385 MIDI files, 5.5 GB; captions 428 MB JSON Lines |
| mT5 cache | 4.3 GB |
| Vocal layer | 8 Tamil lines, placement every 10 s, mix RMS 0.060 → 0.079 |

## 5. Bugs found, and what they taught

The most important lesson: **167 green unit tests did not catch three real
bugs.** Unit tests verified components; only an end-to-end run on real data
exposed the wiring. All three were on the generation path.

### 5.1 `TextTokenizer.__call__` squeezed the batch dimension
A single prompt string produced a 1-D tensor, so the T5 encoder raised
`ValueError: not enough values to unpack (expected 2, got 1)`.
Fix: always return `[B, T]`. (commit `4331ec2`)

### 5.2 The checkpoint was loaded into the wrong module
`_load_for_inference` loaded `model_state` into `model.decoder`, but
checkpoints store the **full** `MusicModel` (`decoder.*` + `text_encoder.*`
keys). The result: the decoder ran on *random weights* and silently produced
noise, with only a `224 missing / 336 unused keys` warning as a clue. Fix: load
into the whole model and assert `missing == 0` — which is why the demo song is
coherent. (commit `4331ec2`)

### 5.3 `torch.multinomial(probs, num=1)` used a wrong keyword
`num` is not a parameter (it is `num_samples`). Depending on the torch version
this either raised or silently sampled wrong. Fixed, and all six sampling paths
now have parametrised tests. (commit `4df1936`)

### 5.4 MidiCaps captions were not what the loader expected
`train.json` is **JSON Lines**, not a JSON array; locations carry an
`lmd_full/` prefix; `genre`/`mood` can be lists. The loader now sniffs the
first character, strips the prefix, and normalises list values to their top
label. (commit `bd7116b`)

### 5.5 The Hugging Face cache was wired two different ways
`HF_HOME=X` stores models under `X/hub`, while `cache_dir=X/hub` points at the
directory *containing* the model dirs. Mixing them made mT5 re-download 2.3 GB
and look like a hang. Fix: `paths.py` sets `HF_HOME` once via `setdefault`;
loaders pass the matching `cache_dir`. (commit `24d37bc`)

### 5.6 A generated MIDI carried a 0 BPM tempo event
An undertrained model emitted an invalid `Tempo` token; `symusic` then reported
a zero-length score, which broke duration maths and the vocal bar alignment.
`save_midi` now replaces tempos outside 40–250 BPM with the range midpoint.
Found only because the vocal stage needed bar times. (commit `3c6c2f5`)

### 5.7 (docs phase) `disk-report` mislabelled and double-counted `data/`
The report added both `data/` and its artifact subdirectories, marking 1.9 GB
of deletable downloads as "keep" and inflating the total. Now only the
hand-written parts (`data/ragas`, `data/lyrics`) are marked keep.

## 6. Testing philosophy

* 200 tests, all fast and offline (`-m "not slow"`); no network, no model
  downloads, no dataset.
* Pure functions are tested exhaustively (placement, mixing, resampling,
  tokenizer round-trips, sampler paths).
* Anything that would need a download or a GPU is marked `slow`.
* **Every bug that reached a real run got a regression test.** That is the
  mechanism that turns "unit tests passed" into "the pipeline works".

## 7. What is intentionally unfinished

| Item | State | Where it would go |
|---|---|---|
| Token-level lyric interleaving | scaffolded, disabled | Phase 3: `tokenizer.lyrics.enabled` |
| Tamil caption translation | implemented, optional | Phase 2: `promptmusic translate` |
| True singing voice synthesis | not attempted | see `docs/07_VOCALS.md` §6 |
| CLAP / TB / CK metrics | deliberately omitted | `docs/09_EVALUATION.md` §3 |
| Full 168 k training run | config ready | `configs/from_scratch.yaml` |
| GitHub publication | not done (local only, by request) | `git remote add origin ...` |

## 8. Reproduce the demo from scratch

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[mac]"
.venv/bin/promptmusic download --source midicaps
# build data/processed/love_subset.jsonl (see docs/08_QUICKSTART.md step 5)
.venv/bin/promptmusic tokenize -c ta_love_test --cache data/processed/love_en_400.jsonl
.venv/bin/promptmusic train -c ta_love_test
.venv/bin/promptmusic generate -c ta_love_test \
  --prompt "a romantic tamil love song, gentle and emotional" \
  --lyrics-file data/lyrics/ta_love_song.txt --language ta --duration 60
.venv/bin/promptmusic vocals --midi outputs/samples/*.mid --wav outputs/samples/*.wav \
  --lyrics-file data/lyrics/ta_love_song.txt
.venv/bin/promptmusic evaluate outputs/samples/*.mid --expect-duration 60
```

Total wall-clock on the reference machine: roughly 45 minutes, dominated by the
download and the 100 training steps.

---

## 9. Acknowledgement of the work's shape

This was built as a *learning* artifact: the point was to walk the whole path —
scrape, tokenise, translate, train, sample, render, sing, evaluate, document —
and to record the mistakes honestly. The model is small on purpose; the
documentation is long on purpose. If you extend it, keep the habit: measure,
write the number down, and add a regression test for every bug that escaped the
unit tests.
