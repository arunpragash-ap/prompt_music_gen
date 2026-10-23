# 07 — Vocals: from spoken lines to melody-aware singing

This page explains why a MIDI model has no singer, what this project does about
it, and — just as importantly — what it still cannot do.

---

## 1. Why the generated song has no singing

The model generates **MIDI**, which is a score, not audio:

```
Bar_None Position_0 Program_0 Pitch_60 Velocity_80 Duration_1.0.0
```

There is no voice track and no words. A General MIDI soundfont can play a
"Choir Aahs" pad, but that is a vowel texture, not Tamil.

When you pass `--lyrics`, the text is folded into the prompt:

```
"{prompt}\n{Tamil|English} lyrics: {lyrics}"
```

The frozen mT5 encoder reads it and steers the **mood and structure** of the
music. Unless you run a vocal stage, the words are never heard.

## 2. Two vocal stages

| Command | What it does | When to use |
|---|---|---|
| `promptmusic vocals` | Speaks each line and places it on a bar line (speech, no melody) | quick narration-style demo |
| `promptmusic sing` | Sings each syllable on a note of the generated melody, then masters the mix | the realistic option in this project |

`sing` is **pitch-shaped TTS**, not a neural singing voice. It is a large step
up from speech-on-a-beat: the words follow the tune, hold notes, and get
vibrato. It is still a synthetic voice.

## 3. The `sing` pipeline

`src/promptmusic/singing.py`:

1. **Syllabify the lyrics.** Tamil is split on vowel signs, virama and
   independent vowels; English uses a vowel-group heuristic. Each line becomes
   a list of syllables. Spaces are kept for phrase boundaries.
2. **Find the melody.** `extract_lead_track()` scores every non-drum track for
   note count, monophony and register, and picks the lead line. This beats a
   "skyline" (highest note at each instant), which is too sparse because the
   model writes chords.
3. **Quantise and simplify.** `quantize_melody()` snaps onsets to an
   eighth-note grid at the song's *actual* tempo, merges repeated pitches and
   joins legato notes up to 1.2 s. `simplify_melody()` removes ornaments a
   singer would not articulate.
4. **Segment phrases and allocate lines.** `segment_phrases()` uses rests;
   `allocate_line_sections()` gives each lyric line its own stretch of the
   song, in order.
5. **Place syllables.** If a section has at least 0.8 melody notes per
   syllable, `plan_line()` sings note-for-note. Otherwise `plan_line_grid()`
   gives every syllable one rhythmic step inside the section's active runs and
   takes its pitch from the nearest melody note. This is what stops 13
   syllables being crushed into a 0.05 s grace note.
6. **Pitch-shape the speech.** Each syllable is spoken once by the TTS model,
   then `fit_pitch()` folds the note into ±5.5 semitones of the speaker's own
   F0 (E3-ish for the Tamil MMS voice) and compresses the interval by 0.7 so
   the result stays natural. `pitch_shift()` uses a phase vocoder plus
   resampling; `time_stretch()` uses the same STFT engine.
7. **Closed-loop correction.** After shifting, `render_line()` measures the
   syllable's F0 and corrects up to ±4 semitones in at most two passes.
8. **Human detail.** `vibrato()` adds a delayed 5 Hz oscillation (about 25
   cents) and a short fade avoids clicks.
9. **Mix and master.** `master_mix()` in `src/promptmusic/master.py` normalises
   the instrumental's active RMS to 0.09, sets the vocal to
   `instrumental × 10^(gain/20)` (default **+1.5 dB**), high-passes the vocal,
   adds presence and compression, widens the music, ducks it slightly under
   the voice, adds a procedural reverb, and limits peaks (never boosts quiet
   material) to −1 dBFS. `--stems` also writes the vocal and instrumental.

## 4. Clean the arrangement first

Raw model output is valid MIDI but a poor arrangement: 10–18 instruments, a
bass above the melody, notes outside the key, every note 0.12 s long.
`promptmusic arrange` fixes that before you render:

```bash
promptmusic arrange --midi take.mid --out take_arranged.mid
```

It detects the key (Krumhansl-Schmuckler), keeps the lead/bass/pad/drums plus a
few supporting parts (default six non-drum tracks), snaps out-of-key notes,
folds each role into a sensible octave range, balances velocities, removes
duplicate notes, and **sustains** short notes until the next onset (a pad chord
can ring for six seconds). The report prints the key, the tracks kept and
removed, and how many notes were changed.

## 5. Usage

```bash
# 1. generate (26-60 s of music)
promptmusic generate \
  -p "an intimate tamil love ballad, soft voice, acoustic guitar, strings" \
  --lyrics-file data/lyrics/ta_love_song.txt --language ta \
  --max-tokens 1024 --temperature 0.95 --top-p 0.92 \
  --out outputs/samples --no-render

# 2. make the MIDI musical
promptmusic arrange --midi outputs/samples/<take>.mid --out /tmp/take.mid

# 3. render + sing + master in one step
promptmusic sing --midi /tmp/take.mid \
  --lyrics-file data/lyrics/ta_love_song.txt \
  --out outputs/samples/tamil_love_song.wav --stems
```

If you already rendered a WAV, pass `--wav`; `sing` will use it instead of
rendering again. Otherwise it humanises the MIDI and renders with fluidsynth.

| Flag | Default | Meaning |
|---|---|---|
| `--midi` | required | melody + timing |
| `--wav` | render the MIDI | use an existing instrumental |
| `--lyrics` / `--lyrics-file` | — | UTF-8, one line per phrase |
| `--out` | `<midi>_sing.wav` | final mix |
| `--voice` | `facebook/mms-tts-tam` | any `transformers` VITS repo |
| `--gain` | `+1.5` dB | vocal RMS relative to the music |
| `--device` | `cpu` | `cpu` or `mps` for the speech model |
| `--rate` | `44100` | output sample rate |
| `--stems` | off | also write `_vocals.wav` / `_instrumental.wav` |
| `--no-master` | off | skip EQ/compression/reverb/limiting |
| `--no-humanize` | off | render notes exactly as written |
| `--seed` | `0` | humanisation seed |
| `--backend` | `auto` | renderer when `--wav` is absent |

## 6. Quality expectations (honest)

| Aspect | Reality |
|---|---|
| Follows the tune? | **yes** — each syllable is pitch-shifted to a melody note |
| Holds notes / vibrato? | yes, with synthetic vibrato |
| Words intelligible? | mostly; pitch shifting smears consonants |
| Sounds like a human singer? | **no** — it is a TTS voice with a melody imposed |
| Tamil? | yes, with the MMS Tamil voice |
| Voice naturalness? | decent for short syllables; artefacts on large shifts |
| Commercial use? | no with the default voice (CC-BY-NC) |

The demo (`data/lyrics/ta_love_song.txt`, 8 lines, 108 syllables) produces a
≈50 s track with 36 melodic targets and 8 phrases; the vocal sits about 4 dB
above the instrumental and the mix peaks at −1 dBFS.

## 7. Why not real Tamil singing?

True singing-voice synthesis (SVS) needs:

1. a model that maps **phonemes + note pitches + durations** to a waveform
   (DiffSinger, VISinger, Sinsy-class);
2. a **Tamil grapheme-to-phoneme** front end and a Tamil SVS checkpoint;
3. a corpus of **sung** Tamil with aligned notes.

Open SVS voicebanks exist mainly for Chinese, Japanese and English; there is no
public Tamil voicebank at useful scale. The realistic paths are:

* **Pitch-shaped TTS** — what `sing` does; good enough to hear the tune and
  words together, not a singer.
* **Fine-tune an open SVS model** on sung Tamil you record or license — the
  only route to a convincing result.
* **A hosted text-to-music service** — if the goal is a finished song rather
  than a local learning project.

The stage is modular (`sing_song()` returns the stems), so a real SVS backend
can replace the TTS synthesis step without touching the music model.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `pass --lyrics or --lyrics-file` | nothing to sing | pass the text |
| Vocal is silent | the MIDI has no melodic notes | run `promptmusic arrange` first, or generate a longer take |
| Words are rushed | too many syllables for the section | shorten lines or generate more bars |
| Big pitch jumps | wide melody range | `fit_pitch` folds by octave; reduce `--gain` if it is harsh |
| Buzzy consonants | large pitch shift | expected; use shorter syllables or a closer voice |
| Vocals too loud/quiet | gain staging | `--gain 0` for equal active RMS, negative to sit back |
| `ModuleNotFoundError: transformers` | wrong interpreter | use `.venv/bin/promptmusic` |
| First run slow | ~80 MB TTS download | one-time; cached |

## 9. Licence warning

`facebook/mms-tts-tam` is **CC-BY-NC 4.0 — non-commercial only**. The code is
Apache-2.0, the model weights are not. Swap `--voice` for a permissively
licensed VITS model before any commercial use. See
[05_MODELS_AND_LICENSES.md](05_MODELS_AND_LICENSES.md).
