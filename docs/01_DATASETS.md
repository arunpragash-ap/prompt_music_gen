# 01 — Datasets: what the model learns from

Everything in this project is trained from public data. This page explains what
each dataset is, why it was chosen, how to download it, how much disk it uses,
and what its licence allows.

---

## 1. MidiCaps (the main dataset)

**What it is.** `amaai-lab/MidiCaps` is a collection of ~168 000 MIDI files
paired with natural-language captions. It is derived from the **Lakh MIDI
Dataset**, a well-known research corpus of pop/rock/electronic music scraped
from the web. MidiCaps adds an English description to every piece, written by
a large language model from the MIDI's measurable features (tempo, key,
instruments, note density, …). That description is exactly the "prompt" side
of our training pairs.

**Size on disk.** ~5.5 GB for all 168 385 MIDI files, plus a 428 MB caption
file. Shards are named `0–9` and `a–f`.

**Format of the caption file.** Despite the `.json` extension,
`train.json` is **JSON Lines** — one JSON object per line, not a JSON array.
The loader sniffs the first character (`{` vs `[`) and handles both.
Each record contains a caption plus metadata such as MIDI path, genre and
mood lists.

**How to download.**

```bash
promptmusic download --source midicaps          # resumes; safe to re-run
```

The downloader stores MIDI under `dataset/midicaps/` and the caption file
under `data/downloads/train.json`. Both are gitignored: they are large and
regenerable.

**Licence.** MidiCaps is distributed for **non-commercial research use**
(CC BY-NC 4.0 on Hugging Face at the time of writing). Lakh MIDI, its source,
carries its own attribution requirements. If you intend to ship music made
with this model commercially, read both dataset cards first. The project
treats MidiCaps as *non-commercial* in [05_MODELS_AND_LICENSES.md](05_MODELS_AND_LICENSES.md).

**Why the captions are imperfect.** They were auto-generated, so some are
generic ("a song with piano and drums"). That is fine for conditioning
practice; it is one reason the model's output is generic too.

### Making a smaller, focused subset

Training on all 168 k pieces is unnecessary for a first result. `data/processed/love_subset.jsonl`
was built by scanning the captions for romantic/gentle keywords (38 519 hits)
and keeping 300 romantic pieces plus 100 general ones for diversity:

```
data/processed/love_subset.jsonl   500 caption records
data/processed/love_en_400.jsonl   400 tokenised examples (10.8 s to build)
```

The filtering code lives in the data pipeline; the resulting files are small
enough to inspect with any text editor.

---

## 2. MIDI → tokens (the REMI cache)

The model never sees MIDI directly. `promptmusic tokenize` converts each file
with the **miditok** REMI tokenizer into a token sequence and caches it as
JSON Lines.

```bash
promptmusic tokenize                       # whole dataset (slow first time)
promptmusic tokenize -c ta_love_test       # the 400-piece love subset
```

What REMI tokens look like (simplified):

```
Bar_None  Position_0  Instrument_0  Pitch_60  Velocity_80  Duration_1.0.0 ...
```

Special tokens: `<pad>` = 0, `<bos>` = 1, `<eos>` = 2. The music vocabulary is
**423 tokens**. Two numbers matter for generation:

| Corpus | Median tokens/second |
|---|---|
| Raga corpus (solo classical) | ~12 |
| MidiCaps love subset (slow songs) | ~40 |
| MidiCaps overall (dense pop) | ~60 |

`generate --duration 60` uses `tokens_per_second` (default 40) to size the
request; the real duration is then measured from the generated MIDI.

**Tempo safety.** Undertrained models can emit an invalid `Tempo` token.
`save_midi` now replaces any tempo outside 40–250 BPM with the midpoint, so
saved files always have a usable duration. (Before this fix, a 0 BPM event
made `symusic` report a zero-length score — the vocal alignment stage found
this bug.)

---

## 3. Lyrics

Two distinct uses of text:

1. **Conditioning** (implemented): lyrics are folded into the prompt as
   `"{prompt}\n{Tamil|English} lyrics: {lyrics}"` before the mT5 encoder reads
   it. This steers mood and structure.
2. **Token-level lyric interleaving** (Phase 3, scaffolded): lyric tokens as a
   separate stream interleaved with music tokens. The config switch is
   `tokenizer.lyrics.enabled`; the corpus downloader is
   `promptmusic lyrics-build`, which reads `asigalov61/Lyrics-MIDI-Dataset`
   (see `data.lyrics` in `configs/default.yaml`).

The demo used a hand-written 8-line Tamil love song:
`data/lyrics/ta_love_song.txt`. Any UTF-8 file works — one line per phrase.

---

## 4. Hindustani ragas (synthesised corpus)

`data/ragas/raga_table.yaml` describes Hindustani ragas: scale degrees,
ascending/descending patterns, characteristic phrases, and typical instruments.
`promptmusic raga-build` composes synthetic MIDI exercises from the table
(`outputs/raga_corpus/`, ~2.6 MB for the smoke set), giving the model exposure
to scales that barely appear in MidiCaps. Measured density is ~12 tokens/s,
so a 60 s raga needs far fewer tokens than a pop song.

Enable it in training with `raga.enabled: true` and a matching `data.mix`.

---

## 5. Reference music (no download needed)

`promptmusic analyze <file>` turns *your* favourite song into a text fragment:
tempo, key, instruments, density, pitch range. For MIDI references the opening
bars can also prime the decoder (`generate --ref-midi`). Audio references need
the optional `librosa` install; without it the command degrades gracefully.

---

## 6. Disk summary

| Artifact | Size | Regenerable? |
|---|---|---|
| `dataset/midicaps/` (168 k MIDI) | 5.5 GB | yes — `promptmusic download` |
| `data/downloads/train.json` | 428 MB | yes |
| `data/processed/*.jsonl` | < 2 GB | yes — `promptmusic tokenize` |
| `.hf_cache/` (mT5 + TTS models) | 4.5 GB | yes — re-downloaded on demand |
| `outputs/` (checkpoints + samples) | 3.2 GB | yes — retrain / regenerate |

See [10_DISK_CLEANUP.md](10_DISK_CLEANUP.md) before deleting anything.
