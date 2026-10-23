# 07 — Vocals: putting the lyrics into the audio

This page explains a limitation that surprises most people, and the workaround
this project implements.

---

## 1. Why the generated song has no singing (by default)

The model generates **MIDI**. MIDI is sheet music for a computer:

```
Bar_None Position_0 Instrument_0 Pitch_60 Velocity_80 Duration_1.0.0
```

There is no audio and no voice track. No soundfont can articulate words — the
closest General MIDI offers is "Choir Aahs", which is a vowel pad, not Tamil.

So when you pass `--lyrics`, the lyrics are folded into the *prompt text*:

```
"{prompt}\n{Tamil|English} lyrics: {lyrics}"
```

The frozen mT5 encoder reads that and steers the **mood and structure** of the
music. The words influence the music; they are never sung.

## 2. What the `vocals` stage does

`src/promptmusic/vocals.py` adds a second pass:

1. Split the lyric file into lines.
2. Speak each line with a local Tamil text-to-speech model
   (`facebook/mms-tts-tam`, VITS, 36 M params, 16 kHz).
3. Read the MIDI's tempo and time signature to compute bar length.
4. Place each line on a bar boundary, spread across the song. Lines that would
   collide are pushed forward instead of overlapping.
5. Resample speech to the instrumental's sample rate (44.1 kHz) with a
   polyphase filter, fade the edges to avoid clicks, and mix it under the music
   at `--gain` dB (default −4 dB).
6. Rescale only if the mix would clip.

That is an honest **spoken vocal layer**. The words are audible and land on the
beat, but they do **not** follow the generated melody.

## 3. Usage

```bash
promptmusic vocals \
  --midi outputs/samples/<song>.mid \
  --wav  outputs/samples/<song>.wav \
  --lyrics-file data/lyrics/ta_love_song.txt
# -> outputs/samples/<song>_vocal.wav
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--lyrics` / `--lyrics-file` | — | one line per phrase, UTF-8 |
| `--voice` | `facebook/mms-tts-tam` | any `transformers` VITS TTS repo |
| `--stride-bars` | spread evenly | bars between line starts |
| `--gain` | `-4` dB | vocal level relative to the music |
| `--device` | `cpu` | `cpu` or `mps` for the speech model |
| `--out` | `<wav>_vocal.wav` | output path |

The demo run: 8 Tamil lines, 81.6 s, lines every 10 s, speech energy present at
every placement (mix RMS rose from 0.060 to 0.079).

## 4. Quality expectations

| Aspect | Reality |
|---|---|
| Words intelligible? | yes, as speech |
| On the beat? | yes — lines start on bar boundaries |
| Melody-matched? | **no** — pitch follows speech prosody |
| Voice naturalness? | decent MMS-TTS quality; robotic on long lines |
| Language | Tamil (default voice); any MMS language via `--voice` |
| Singing? | **no** — this is not singing voice synthesis |

If a line is longer than its slot, the next line is pushed later; if the song
ends first, the tail is truncated and a warning is printed. Instrumentals
shorter than a few seconds are not useful for this stage.

## 5. Licence warning

`facebook/mms-tts-tam` is **CC-BY-NC 4.0 — non-commercial only**. Swap
`--voice` for a permissively licensed VITS model before any commercial use.
See [05_MODELS_AND_LICENSES.md](05_MODELS_AND_LICENSES.md).

## 6. The upgrade path to real singing

True singing voice synthesis (SVS) needs three things this project does not
have:

1. A model that maps **phonemes + note pitches + durations** to a waveform
   (DiffSinger, VISinger, Sinsy-class).
2. A **Tamil phonemiser** (grapheme-to-phoneme) and a Tamil SVS checkpoint.
3. A corpus of **sung** Tamil with aligned notes — not publicly available at
   useful scale.

Practical options, in order of effort:

* **Pitch-shaped speech** (research project): estimate the speech F0, snap it
  to the generated melody's scale, time-stretch each syllable to the nearest
  note. Convincing enough for demos; still not true singing.
* **Fine-tune an open SVS model** on a small sung-Tamil dataset you record or
  license. This is the realistic route to a real singer.
* **Use a hosted text-to-music service** if the goal is a finished song rather
  than a learning project.

The project deliberately does not pretend the vocal layer is SVS. The stage is
modular, so a future `vocals` backend can implement the real thing without
touching the music model.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `voice=None ... not a valid model identifier` | older build passed `None` | fixed; `--voice` now falls back to the default |
| Voice silent | MIDI tempo unreadable | tempo sanitisation now prevents 0 BPM; check the MIDI with `promptmusic evaluate` |
| Lines overlap | slots shorter than speech | pass `--stride-bars` or shorten lines |
| `ModuleNotFoundError: transformers` | wrong interpreter | use `.venv/bin/promptmusic` |
| First run slow | 139 MB model download | one-time; cached in `.hf_cache` |
