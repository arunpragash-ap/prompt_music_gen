# 04 — Generation: prompt → MIDI → WAV

This is the part you will use most: typing a sentence and getting music back.
Every option below is also visible via `promptmusic generate --help`.

---

## 1. The shortest possible run

```bash
promptmusic generate --prompt "a bright piano study"
```

Output (relative to the project root):

```
outputs/samples/a_bright_piano_study__0.mid
outputs/samples/a_bright_piano_study__0.wav     # rendered automatically
```

The filename is a slug of the prompt plus the sample index. `--json` prints a
machine-readable result instead of the aligned table.

## 2. What the flags do

| Flag | Meaning |
|---|---|
| `--prompt "..."` | Text description (English or Tamil) |
| `--lyrics "..."` / `--lyrics-file f` | Fold lyric text into the prompt |
| `--language en\|ta` | How the prompt is labelled to the encoder |
| `--max-tokens N` | Hard cap on generated music tokens |
| `--duration SEC` | Target length; converts using `generate.tokens_per_second` (default 40) |
| `--temperature`, `--top-k`, `--top-p` | Sampling controls |
| `--beams N` | Beam search width (1 = sampling) |
| `--seed N` | Reproducibility |
| `--checkpoint PATH` | Which checkpoint to load |
| `--no-render` | MIDI only (faster) |
| `--ref-midi PATH` | Phase 4: prime the decoder with the opening bars of a MIDI |
| `--ref-audio PATH` | Phase 4: analyse a recording and append its character to the prompt |
| `-o key=value` | Any config override, e.g. `-o generate.temperature=0.9` |

## 3. Length: tokens vs seconds

The model counts tokens, not seconds. The conversion depends on how dense the
music is:

| Music | Tokens/second |
|---|---|
| Sparse raga | ~12 |
| Slow love song | ~40 (default) |
| Dense pop | ~60 |

So `--duration 60` requests `60 × 40 = 2 400` tokens (plus `BOS`), clamped by
`--max-tokens` / `decoder.max_len`. The actual duration is measured from the
MIDI after generation and reported in the summary. For a sparse piece, use
`-o generate.tokens_per_second=15` to avoid generating far more tokens than
you need.

The demo config caps generation at `decoder.max_len: 2560`; asking for 60 s
produced 2 559 tokens and a **78 s** MIDI (the clamp, plus token/note density).
That is expected behaviour, not a bug.

## 4. Sampling: how to steer character

* `temperature` (default 1.0) — lower (0.7) is safer and more repetitive;
  higher (1.2) is more surprising and more error-prone.
* `top-k` (50) — only the 50 likeliest tokens are considered.
* `top-p` (0.95) — dynamic alternative to top-k.
* `repetition_penalty` (1.0) — above 1.0 discourages repeating phrases.

Two songs from the same prompt with different seeds differ. The project ships
no "good seed" list; explore.

## 5. Rendering

`generate` renders automatically unless `--no-render`. Rendering can also be a
separate step:

```bash
promptmusic render outputs/samples/foo.mid
promptmusic render outputs/samples/ --backend fluidsynth --out out.wav
promptmusic render --backends          # list what this machine supports
```

Backend preference: `fluidsynth` CLI → `pyfluidsynth` → pure-Python
`sf2utils` → `numpy` (a sine-wave fallback that always works). The soundfont
is auto-located: config → `data/downloads/soundfont.sf2` → MuseScore General
in `~/Library/Application Support/soundfont/` → Homebrew FluidR3 →
`pretty_midi`'s bundled `TimGM6mb.sf2`.

## 6. The Tamil love-song demo, end to end

```bash
promptmusic generate \
  --prompt "a romantic tamil love song, gentle and emotional" \
  --lyrics-file data/lyrics/ta_love_song.txt \
  --language ta \
  --duration 60 \
  --checkpoint outputs/prompt_music_gen/checkpoints/best.pt
```

Measured result: 2 559 tokens → 78.0 s MIDI, 516 notes, 18 instruments
(piano 243 notes, drums 122, guitar 40, strings 13, electric piano 19,
bass 11, …) → 81.6 s WAV. Generation took ~40 s; rendering a few seconds.

## 7. Adding vocals

```bash
promptmusic vocals \
  --midi   outputs/samples/<song>.mid \
  --wav    outputs/samples/<song>.wav \
  --lyrics-file data/lyrics/ta_love_song.txt
# -> outputs/samples/<song>_vocal.wav
```

This is a **spoken vocal layer**, not melody-matched singing. Read
[07_VOCALS.md](07_VOCALS.md) before expecting a singer.

## 8. Reference conditioning (Phase 4)

```bash
promptmusic analyze my_song.mid                 # describe a MIDI
promptmusic analyze my_song.mp3 --kind audio    # describe a recording (needs librosa)
promptmusic generate --ref-midi my_song.mid --prompt "continue in this style"
promptmusic generate --ref-audio my_song.mp3 --prompt "something like this"
```

MIDI analysis reads tempo, key, time signature, instruments, note density and
pitch range, and can prime the decoder with the opening bars. Audio analysis
extracts statistical descriptors (tempo, key estimate, brightness, energy)
because automatic transcription is out of scope. Without `librosa`, audio
references degrade gracefully.

## 9. Output locations

| What | Where |
|---|---|
| MIDI + WAV | `outputs/samples/` |
| checkpoint | `outputs/prompt_music_gen/checkpoints/best.pt` |
| raga corpus | `outputs/raga_corpus/` |

`paths.py` owns these locations; `PROMPTMUSIC_OUTPUT` relocates all outputs.
