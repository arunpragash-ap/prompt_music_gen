# 08 — Quickstart: zero to music in 15 minutes

Every command is copy-paste ready. Times are for the reference machine
(Apple M2, 8 GB, MPS) with a warm internet connection.

---

## 0. What you need

* macOS or Linux, Python 3.10+
* ~3 GB free disk for the smoke path (12 GB for the full pipeline)
* Optional but recommended: `brew install fluid-synth`

## 1. Install (3–5 min)

```bash
cd prompt_music_gen
uv venv --python 3.12 .venv
uv pip install -e ".[mac]"
.venv/bin/python -m pytest -q -m "not slow"     # 200 passed
```

No `uv`? Use `python -m venv .venv` and `.venv/bin/pip install -e ".[mac]"`.

## 2. Check the machine (5 s)

```bash
.venv/bin/promptmusic info
```

Look for: `torch` version, `mps_available: True`, `fluidsynth: .../bin/fluidsynth`,
`dataset: .../dataset/midicaps` (empty until step 4).

## 3. First music — no download needed (1–2 min)

The smoke config can train on the synthetic **raga corpus**, which is
generated locally:

```bash
.venv/bin/promptmusic raga-build -c smoke --variations 4    # ~100 MIDI files
.venv/bin/promptmusic tokenize -c smoke --limit 200 \
  -o paths.data_dir=. \
  -o data.midi_subdir=outputs/raga_corpus \
  -o data.caption_file=outputs/raga_corpus/captions.json \
  --cache data/processed/raga_smoke.jsonl
.venv/bin/promptmusic train -c smoke \
  -o data.cache_file=data/processed/raga_smoke.jsonl
.venv/bin/promptmusic generate -c smoke \
  --prompt "a meditative hindustani raga with sitar and tabla" \
  --duration 8
```

You should get a `.mid` and `.wav` in `outputs/samples/`. Play the WAV:

```bash
open outputs/samples/*.wav        # macOS
```

The music will be rough — 60 training steps is a pipeline test, not a song.

## 4. The real dataset (download ~5.5 GB)

```bash
.venv/bin/promptmusic download --source midicaps
```

Resumable; re-running continues where it stopped. Captions land in
`data/downloads/train.json` (JSON Lines).

## 5. The Tamil love-song demo (30–40 min)

This needs the filtered 400-piece subset. Build it once from the full captions:

```python
# save as build_love_subset.py and run with .venv/bin/python
import json
from pathlib import Path

KEYWORDS = ("love", "romantic", "ballad", "gentle", "emotional", "heart")
rows, romantic, general = [], [], []
for line in Path("data/downloads/train.json").open(encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    rows.append(rec)
    text = (rec.get("caption") or "").lower()
    if sum(k in text for k in KEYWORDS) >= 2:
        romantic.append(rec)
    else:
        general.append(rec)

subset = romantic[:300] + general[:100]
Path("data/processed").mkdir(parents=True, exist_ok=True)
with Path("data/processed/love_subset.jsonl").open("w", encoding="utf-8") as fh:
    for rec in subset:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"{len(romantic)} romantic hits; kept {len(subset)} pieces")
```

Then tokenise and train:

```bash
.venv/bin/promptmusic tokenize -c ta_love_test \
  --cache data/processed/love_en_400.jsonl
.venv/bin/promptmusic train -c ta_love_test
```

`configs/ta_love_test.yaml` contains an absolute `caption_file` from the
reference machine — override it if your path differs:

```bash
-o data.caption_file=$PWD/data/processed/love_subset.jsonl
```

Training stops at 200 steps (`save_every: 100` writes `best.pt` around step
100). On an 8 GB M2 expect roughly **10 s/step** at 1 024 tokens once the
machine starts swapping, so a 200-step run is a 30–40 minute coffee break.

## 6. Generate the song, then sing it

Generate a short take (keep `--max-tokens` near the 1 024 the model was trained
on; longer sequences extrapolate and wander):

```bash
.venv/bin/promptmusic generate -c ta_love_test \
  --prompt "an intimate tamil love ballad, soft voice, acoustic guitar, warm strings" \
  --lyrics-file data/lyrics/ta_love_song.txt \
  --language ta --max-tokens 1024 --no-render \
  --checkpoint outputs/prompt_music_gen/checkpoints/best.pt
```

Then clean the arrangement (key, instruments, ranges, note lengths) and sing:

```bash
.venv/bin/promptmusic arrange \
  --midi outputs/samples/<take>.mid --out /tmp/song.mid

.venv/bin/promptmusic sing \
  --midi /tmp/song.mid \
  --lyrics-file data/lyrics/ta_love_song.txt \
  --out outputs/samples/tamil_love_song.wav --stems
```

`sing` renders the instrumental, pitch-shifts each syllable onto the melody,
adds vibrato, and masters the mix. The lyrics follow the tune, but the voice is
synthetic — it is not a human singer. Read
[07_VOCALS.md](07_VOCALS.md) for the pipeline, the controls, and the honest
ceiling.

## 7. What to try next

```bash
# different seeds / temperatures
.venv/bin/promptmusic generate --prompt "a tense cinematic chase with drums" --seed 7

# inspect/score a result
.venv/bin/promptmusic evaluate outputs/samples/<song>.mid

# use a reference song
.venv/bin/promptmusic analyze my_favourite.mid
.venv/bin/promptmusic generate --ref-midi my_favourite.mid --prompt "in this style"

# export the model for another machine
.venv/bin/promptmusic export outputs/prompt_music_gen/checkpoints/best.pt -f pt

# see what is using disk
.venv/bin/promptmusic disk-report
```

## 8. If something fails

| Error | Fix |
|---|---|
| `command not found: promptmusic` | use `.venv/bin/promptmusic` |
| `MPS backend out of memory` | add `-o tokenizer.max_tokens=768`, close other apps |
| `config not found` | `-c` expects a name in `configs/` or a YAML path |
| no WAV produced | install fluidsynth; check `promptmusic render --backends` |
| mT5 download restarts | leave `.hf_cache` alone; `paths.py` sets `HF_HOME` for you |

More detail: [03_TRAINING.md](03_TRAINING.md), [04_GENERATION.md](04_GENERATION.md),
[06_DEPLOYMENT.md](06_DEPLOYMENT.md).
