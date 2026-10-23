# 00 — Overview: what this project is (read this first)

`prompt_music_gen` (package name **`promptmusic`**) turns a written description
into music. You type a sentence — in English or Tamil — and the project
generates a multi-track **MIDI** file, then renders it to an audible **WAV**.
If you supply lyrics, a final optional stage speaks/sings them over the
instrumental (see [07_VOCALS.md](07_VOCALS.md)).

It is a *learning project*: small enough to read end-to-end, honest about its
limits, and fully documented so someone with no AI background can follow every
step.

---

## 1. The one-paragraph explanation

A neural network called a **transformer decoder** is trained on pairs of
(text description, music token sequence). Music is first converted to a
sequence of tokens — the **REMI** representation — so that generating music
becomes the same problem as generating text. A frozen multilingual language
model (**mT5-base**) reads the description, and the decoder predicts music
tokens one at a time, conditioned on that reading. The token sequence is
converted back to MIDI, and a soundfont synthesiser (fluidsynth) turns MIDI
into audio.

## 2. What it can do

| Capability | Command | Status |
|---|---|---|
| Prompt (English/Tamil) → MIDI | `promptmusic generate --prompt "..."` | works |
| MIDI → WAV | `promptmusic render file.mid` | works (fluidsynth) |
| Lyrics spoken over the music | `promptmusic vocals ...` | works (spoken vocal layer) |
| Train on MidiCaps | `promptmusic tokenize && promptmusic train` | works |
| Reference MIDI/audio conditioning | `promptmusic analyze`, `generate --ref-midi` | works |
| Hindustani raga corpus | `promptmusic raga-build` | works |
| Evaluation metrics | `promptmusic evaluate` | works |
| Export (pt/TorchScript/ONNX) | `promptmusic export` | works |

## 3. What it cannot do (read before judging output)

* **No true singing voice.** MIDI has no vocal track. The `vocals` stage uses a
  speech model and places lines on bar boundaries — the words are audible but
  they do **not** follow the generated melody. Open Tamil singing-voice
  synthesis models do not exist. See [07_VOCALS.md](07_VOCALS.md).
* **Short training ≠ good music.** The shipped checkpoint was trained for 100
  steps on 400 pieces (a laptop-memory constraint). It produces plausible
  structure and instrumentation but the musical quality is that of a demo, not
  a product. [03_TRAINING.md](03_TRAINING.md) explains how to train longer.
* **Audio is not generated directly.** The model generates notation; the
  sound comes from a soundfont. Timbre quality depends on the soundfont.
* **Tamil is conditioning, not native generation.** mT5 is multilingual, so
  Tamil prompts work, but the music vocabulary is language-agnostic.

## 4. The pipeline at a glance

```
                     TRAINING (done once)                     GENERATION (each run)
                     ─────────────────────                    ─────────────────────
  MIDI files ──► REMI tokens ──┐                    "a romantic tamil
  captions  ──► mT5 ids ───────┤                     love song, gentle"
                               ▼                              │
                        (text, music) pairs                   ▼
                               │                     mT5 tokenizer → frozen
                               ▼                     mT5 encoder → memory
                     train decoder (51 M)                     │
                               │                              ▼
                               ▼                     decoder samples music
                     checkpoints/best.pt  ──────────► token ids
                                                              │
                                                              ▼
                                                       decode → .mid
                                                              │
                                                              ▼
                                              fluidsynth + soundfont → .wav
                                                              │
                                                    (optional) vocals stage
                                                              ▼
                                                       .wav with lyrics
```

## 5. Repository layout

```
prompt_music_gen/
├── configs/              YAML configs (default, smoke, ta_love_test, ...)
├── data/
│   ├── lyrics/           lyric text files (UTF-8)
│   ├── processed/        token caches + filtered caption subsets
│   ├── ragas/            raga table (Hindustani scales)
│   └── downloads/        raw caption files (gitignored)
├── dataset/              MIDI corpora (gitignored, regenerable)
├── docs/                 this documentation
├── outputs/              checkpoints, samples (gitignored)
├── scripts/              repo + disk helper scripts
├── src/promptmusic/      the Python package
├── tests/                pytest suite (200 tests)
├── Makefile              shortcuts for every command
└── pyproject.toml        dependencies and packaging
```

## 6. Key numbers (measured on an Apple M2, 8 GB)

| Quantity | Value |
|---|---|
| Decoder parameters (trained) | 51.3 M |
| Text encoder parameters (frozen) | ~277 M (mT5-base encoder) |
| Total model | 328.3 M |
| Music vocabulary | 423 tokens |
| Token density (love-song subset) | ~40 tokens/second |
| Training speed (1 024-token examples) | ~2.5–3 s/step |
| Generation speed | ~64 tokens/s |
| Checkpoint size | 1.6 GB (includes optimizer state) |
| Full MidiCaps download | 5.5 GB (168 385 MIDI files) |
| mT5-base download | 4.3 GB |

## 7. Glossary for non-AI readers

| Term | Plain meaning |
|---|---|
| **Token** | A small unit of information; here one musical event (note-on, bar, tempo…) or one piece of text. |
| **REMI** | A standard way to write music as tokens (`Bar`, `Position`, `Pitch`, `Velocity`, `Instrument`…). |
| **Transformer** | The neural-network architecture behind modern language models; it learns which parts of the input matter for each prediction. |
| **Encoder / decoder** | The encoder *reads* the prompt; the decoder *writes* the music, one token at a time. |
| **Frozen** | Weights that are not updated during training (saves memory and keeps language understanding intact). |
| **Checkpoint** | A saved snapshot of everything needed to resume training or generate. |
| **MIDI** | A file of instructions ("play note 60 at volume 87 for one beat"). It contains no sound. |
| **Soundfont (.sf2)** | A library of recorded instrument samples; the synthesiser uses it to turn MIDI into audio. |
| **Teacher forcing** | Training trick: show the model the true previous tokens and ask it to predict the next one. |
| **Sampling** | Choosing the next token from the model's probability distribution (temperature/top-k/top-p control the choice). |

## 8. Where to go next

* Brand new? → [08_QUICKSTART.md](08_QUICKSTART.md), then run the commands.
* Understand the model → [02_ARCHITECTURE.md](02_ARCHITECTURE.md).
* Train it → [03_TRAINING.md](03_TRAINING.md).
* Generate music → [04_GENERATION.md](04_GENERATION.md).
* Add vocals → [07_VOCALS.md](07_VOCALS.md).
* Free disk space → [10_DISK_CLEANUP.md](10_DISK_CLEANUP.md).
* Build log, decisions and bugs → [../IMPLEMENTATION.md](../IMPLEMENTATION.md).
