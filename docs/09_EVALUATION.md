# 09 — Evaluation: the numbers behind "is this any good?"

No single number measures musical quality. This project computes a panel of
cheap, deterministic metrics so you can see *where* a model is weak.

```bash
promptmusic evaluate outputs/samples/song.mid
promptmusic evaluate outputs/samples/ --per-file
promptmusic evaluate song.mid --reference dataset/midicaps --expect-tempo 92 \
  --expect-key "C major" --expect-duration 60 --expect-instruments 0 40 33
```

Results print as a table (or JSON with `--json`) and can be saved with `--out`.

---

## 1. Implemented metrics

| Metric | Question it answers | Good direction |
|---|---|---|
| `validity_rate` | Does the decoder emit well-formed music at all? | 1.0 |
| `compression_ratio` (CR) | Bits per note vs. a uniform baseline — did the model find structure? | lower |
| `tempo_match` / `tempo_MAE` | Did it obey the requested tempo? | match / low MAE |
| `key_match` | Did it obey the requested key or scale? | 1.0 |
| `instrument_coverage` | How many requested General MIDI programs appeared? | 1.0 |
| `unique_pitch_classes` | Is it exploring a scale or hammering one note? | 5–12 |
| `repetition_rate` | How often does an exact phrase repeat? | low, but not 0 |
| `duration_error` | Did it stop where asked? | low |
| `notes_per_second` | Plausibility of density | ~0.5–8 |
| `dhat` | Distributional similarity of pitch histograms (reference mode) | higher |

`--reference` compares the target against a directory of real MIDI (pitch
histogram divergence, etc.). `--per-file` keeps per-file rows instead of
aggregating.

## 2. What the demo song scored (measured)

| Metric | Value |
|---|---|
| validity | 100 % |
| notes/second | 6.61 (516 notes / 78 s) |
| unique pitch classes | 12 |
| pitch entropy | 3.43 bits |
| repetition rate | 1.8 % |
| compression ratio | 0.96 |
| duration | 78.0 s (+18 s vs the 60 s request) |

Reading: the output is *well-formed* (validity 100 %, low repetition) and
uses all twelve pitch classes with a healthy entropy. The length overshoot
comes from 40 tokens/s × 60 s = 2 400 tokens producing 78 s of slow music;
`generate` reported 18 distinct instruments. Tune `generate.tokens_per_second`
or `--duration` if length matters.

## 3. Deliberately not implemented

These appear in the research literature but need multi-GB pre-trained models
or human listeners:

| Metric | Why omitted | How to add |
|---|---|---|
| **CLAP score** (text↔audio similarity) | needs a CLAP checkpoint (~1–2 GB) and audio embeddings | install `laion-clap`, embed prompt + rendered WAV, cosine similarity |
| **TB / TBT** (token/beat distance) | needs a reference distribution per genre | collect real MIDI, compute token n-gram statistics |
| **CK / CKD** (key consistency) | heavier music-theory model | use `music21` key analysis on windows |
| **Human listening test** | the only ground truth | blind A/B with 5+ listeners |

The principle: a metric you cannot explain is worse than no metric. Everything
implemented here is a few lines of arithmetic.

## 4. Comparing runs

The output dict is stable, so diff two runs:

```bash
promptmusic evaluate outputs/samples/ --json > run_a.json
promptmusic evaluate other_samples/ --json > run_b.json
```

Watch `validity_rate` first (did generation break?), then `repetition_rate`
and `unique_pitch_classes` (did it collapse?), then the prompt-adherence
metrics.

## 5. A sensible evaluation loop

1. Fix a prompt and seed set (e.g. 5 prompts × 3 seeds).
2. Generate all 15 songs at a fixed `--duration`.
3. Score the folder; save JSON.
4. Change one training variable; repeat.
5. Listen to the best 3 yourself — metrics cannot hear phrasing.

See also: [04_GENERATION.md](04_GENERATION.md) for sampling, and
[03_TRAINING.md](03_TRAINING.md) for the training-side knobs worth testing.
