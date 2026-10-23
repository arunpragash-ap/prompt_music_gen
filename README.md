# prompt_music_gen

**Turn a written description into music.** Give it a prompt in English or
Tamil, optionally with lyrics, and it generates a multi-track MIDI file and
renders it to audio. An optional stage then **sings the lyrics on the generated
melody** (pitch-shaped TTS — not a human voice; see the limits below).

```bash
promptmusic generate --prompt "a romantic tamil love song, gentle and emotional" \
  --lyrics-file data/lyrics/ta_love_song.txt --language ta --duration 60 \
  --checkpoint outputs/prompt_music_gen/checkpoints/best.pt
```

This is a **learning project**: small enough to read end-to-end, honest about
its limits, and documented for people with no AI background.

---

## What is inside

* A frozen multilingual text encoder (`google/mt5-base`) reads the prompt.
* A 12-layer, 512-dim transformer decoder (**51 M trainable** parameters)
  generates **REMI** music tokens.
* Tokens → MIDI (via `miditok`) → WAV (via `fluidsynth` + a soundfont).
* Optional: reference-file conditioning, a synthetic Hindustani raga corpus,
  an instrument-balance auditor, evaluation metrics, and model export.

Total model: 328 M parameters, of which 51 M are trained. It runs on an
8 GB Apple M2 with no GPU.

## What it produces

| Stage | Output |
|---|---|
| `generate` | `.mid` (score) + `.wav` (audio) |
| `arrange` | a cleaned `.mid` (key, roles, ranges, sustained notes) |
| `sing` | `.wav` with the lyrics sung on the melody, mastered (+ stems) |
| `vocals` | `.wav` with the lyrics merely spoken over the music |
| `train` | `checkpoints/best.pt` |
| `export` | lean `.pt` / TorchScript / ONNX artifacts |

## Honest limitations

1. **No human singing.** MIDI has no vocal track and no open Tamil
   singing-voice model exists. `sing` pitch-shifts a Tamil TTS voice onto the
   generated notes (with vibrato and mastering), so the words follow the tune,
   but it is still a synthetic voice — not a singer. The older `vocals` stage is
   only a spoken layer. Details and limits: `docs/07_VOCALS.md`.
2. **The shipped checkpoint is a demo.** 100 training steps on 400 pieces, on a
   memory-constrained laptop. It proves the pipeline; it is not release-quality
   music. Training longer is documented.
3. **Audio comes from a soundfont**, not from the model.
4. **Licences:** the training data (MidiCaps) and the Tamil voice model are
   **non-commercial**. See `docs/05_MODELS_AND_LICENSES.md`.

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[mac]"
.venv/bin/python -m pytest -q -m "not slow"      # 265 tests
.venv/bin/promptmusic info
.venv/bin/promptmusic download --source midicaps # 5.5 GB, resumable
.venv/bin/promptmusic tokenize -c ta_love_test
.venv/bin/promptmusic train -c ta_love_test
.venv/bin/promptmusic generate -c ta_love_test \
  --prompt "a romantic tamil love song, gentle and emotional" \
  --lyrics-file data/lyrics/ta_love_song.txt --duration 60 --no-render
.venv/bin/promptmusic arrange --midi outputs/samples/<take>.mid --out /tmp/song.mid
.venv/bin/promptmusic sing --midi /tmp/song.mid \
  --lyrics-file data/lyrics/ta_love_song.txt --out outputs/samples/song.wav --stems
```

Full walkthrough, including a **no-download** first run, in
[`docs/08_QUICKSTART.md`](docs/08_QUICKSTART.md).

## Documentation

| Doc | Contents |
|---|---|
| [00_OVERVIEW](docs/00_OVERVIEW.md) | what this is, glossary, numbers |
| [01_DATASETS](docs/01_DATASETS.md) | MidiCaps, lyric corpus, ragas, licences |
| [02_ARCHITECTURE](docs/02_ARCHITECTURE.md) | how the model works |
| [03_TRAINING](docs/03_TRAINING.md) | training, memory maths, troubleshooting |
| [04_GENERATION](docs/04_GENERATION.md) | prompts → MIDI → audio |
| [05_MODELS_AND_LICENSES](docs/05_MODELS_AND_LICENSES.md) | every component + licence |
| [06_DEPLOYMENT](docs/06_DEPLOYMENT.md) | run it on another machine |
| [07_VOCALS](docs/07_VOCALS.md) | the lyric vocal layer and its limits |
| [08_QUICKSTART](docs/08_QUICKSTART.md) | copy-paste first run |
| [09_EVALUATION](docs/09_EVALUATION.md) | metrics and how to read them |
| [10_DISK_CLEANUP](docs/10_DISK_CLEANUP.md) | what is big, what is safe to delete |
| [IMPLEMENTATION](IMPLEMENTATION.md) | build log, decisions and bugs |

## Commands

```
info          hardware, packages, data availability
download      fetch MidiCaps (resumable)
tokenize      MIDI + captions -> token cache
translate     Phase 2: English captions -> Tamil (IndicTrans2)
lyrics-build  Phase 3: lyric corpus + vocabulary
raga-build    Phase 5: synthesise Hindustani raga MIDI
audit         Phase 6: instrument balance audit
train         train the model (MPS/CPU aware)
generate      prompt -> MIDI (+ WAV)
arrange       clean a generated MIDI (key, roles, sustain)
render        MIDI -> WAV
sing          sing lyrics on the melody, then master
vocals        speak lyrics over an instrumental
analyze       describe a reference MIDI/audio file
evaluate      score generated MIDI
export        package a checkpoint
inspect       describe a checkpoint/export
disk-report   what occupies disk, what is safe to delete
```

Every command has `--help`. `make help` lists the equivalent shortcuts.

## Layout

```
configs/     YAML configs (smoke, default, ta_love_test, from_scratch, paper_size)
src/         the Python package
tests/       265 tests (fast, no network)
docs/        the documentation above
scripts/     repo setup + disk report
data/        raga table, lyrics, processed caches (large parts gitignored)
outputs/     checkpoints and samples (gitignored)
```

## Licence

Code: **Apache-2.0**. Data and the speech model are non-commercial — see
[docs/05_MODELS_AND_LICENSES.md](docs/05_MODELS_AND_LICENSES.md) before any
commercial use.

## Attribution

```
Music generated with prompt_music_gen (Apache-2.0).
Trained on MidiCaps (CC BY-NC 4.0), derived from the Lakh MIDI Dataset.
Text encoder: google/mt5-base (Apache-2.0).
Vocal layer: facebook/mms-tts-tam (CC BY-NC 4.0), pitch-shaped onto the melody.
Rendered with fluidsynth + a General MIDI soundfont.
```
