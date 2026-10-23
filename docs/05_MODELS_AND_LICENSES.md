# 05 — Models, packages and licences

Every external component is listed here with its size, where it is stored, what
it does, and its licence. **Read the non-commercial flags before shipping
anything.**

---

## 1. Models downloaded at runtime

| Model | Role | Params | Size | Licence |
|---|---|---|---|---|
| `google/mt5-base` | frozen text encoder; reads English/Tamil prompts | ~580 M (encoder ~277 M used) | 4.3 GB in `.hf_cache` | **Apache-2.0** |
| `facebook/mms-tts-tam` | Tamil speech model for the vocal layer | 36.3 M | 139 MB | **CC-BY-NC 4.0 — non-commercial** |
| `ai4bharat/indictrans2-en-indic-dist-200M` | Phase 2 EN→TA caption translation (optional) | 200 M | ~1 GB | check the model card (research-oriented) |

Downloads go to `.hf_cache/hub/` because `paths.py` sets `HF_HOME` for the
process. Nothing is downloaded twice; deleting `.hf_cache` simply triggers
re-download on the next run.

### The non-commercial flag, plainly

* `facebook/mms-tts-tam` is **CC-BY-NC 4.0**: you may use it for learning and
  research, not in a commercial product. The vocal layer is therefore a
  non-commercial feature unless you swap the voice.
* **MidiCaps** (the training data) is distributed for non-commercial research
  use (CC BY-NC 4.0 on Hugging Face), and it descends from Lakh MIDI which has
  its own attribution terms. A model trained on it inherits that constraint in
  practice.
* Everything else in the core pipeline (torch, transformers, miditok, mT5,
  fluidsynth, the project code itself) is permissive.

This is why `pyproject.toml` declares **Apache-2.0** for the *code* while this
page flags the *data and voice* as non-commercial. The distinction matters.

## 2. Python packages

| Package | Purpose | Licence |
|---|---|---|
| `torch` | neural network runtime (MPS/CPU) | BSD-3-Clause |
| `transformers` | mT5 + VITS model loading | Apache-2.0 |
| `huggingface_hub` | model downloads | Apache-2.0 |
| `safetensors` | safe weight files | Apache-2.0 |
| `miditok` | MIDI ⇄ REMI tokens | MIT |
| `symusic` | fast MIDI score object | Apache-2.0 |
| `mido` | MIDI file IO | MIT |
| `pretty_midi` | MIDI parsing, tempo/key helpers | MIT |
| `numpy`, `scipy` | arrays, resampling, filters | BSD-3-Clause |
| `soundfile` | WAV/FLAC read/write | BSD-3-Clause |
| `pyyaml` | config files | MIT |
| `ijson` | streaming the 428 MB caption file | BSD-3-Clause |
| `einops` | readable tensor reshaping | MIT |
| `tqdm` | progress bars | MPL-2.0 / MIT |
| `matplotlib` | training curves/figures | PSF-based (BSD-compatible) |
| `librosa` *(optional)* | audio reference analysis | ISC |
| `sf2utils` *(optional)* | pure-Python SoundFont rendering | MIT |
| `pyfluidsynth` *(optional)* | Python bindings to fluidsynth | LGPL-2.1 |
| `tensorboard` *(optional)* | training curves | Apache-2.0 |
| `pytest`, `ruff` *(dev)* | tests, lint | MIT |

The system `fluidsynth` binary (installed with `brew install fluid-synth`) is
**LGPL-2.1**. It is invoked as a separate process, which keeps the licence
boundary clean.

## 3. Soundfonts

| Soundfont | Where found | Licence |
|---|---|---|
| MuseScore General | `~/Library/Application Support/soundfont/MuseScore_General.sf2` | MIT / FluidR3 lineage — check the copy you install |
| FluidR3_GM | Homebrew `share/soundfonts/` | MIT (FluidR3 lineage) |
| TimGM6mb | bundled with `pretty_midi` | GPL-2.0 — **only a last-resort fallback** |

Because the TimGM6mb fallback is GPL, a commercial deployment should ship an
explicitly licensed soundfont via `render.soundfont` rather than rely on the
fallback.

## 4. Datasets

| Dataset | Use | Licence |
|---|---|---|
| MidiCaps (`amaai-lab/MidiCaps`) | training pairs | **CC BY-NC 4.0 (non-commercial)** |
| Lakh MIDI (upstream of MidiCaps) | origin of the MIDI files | attribution required; check the Lakh terms |
| Lyrics-MIDI-Dataset (`asigalov61`) | optional lyric corpus | check the repository card before redistribution |
| Hindustani raga table (`data/ragas/raga_table.yaml`) | synthetic raga corpus | written for this project |

## 5. Sources (where to read more)

* mT5 — <https://huggingface.co/google/mt5-base> · paper: Xue et al., 2020
* MidiCaps — <https://huggingface.co/datasets/amaai-lab/MidiCaps>
* Lakh MIDI — <https://colinraffel.com/projects/lmd/>
* REMI / miditok — <https://github.com/Natooz/MidiTok>
* Text2MIDI (the paper this architecture follows) — <https://arxiv.org/abs/2412.16526>
* MMS-TTS — <https://huggingface.co/facebook/mms-tts-tam>
* IndicTrans2 — <https://github.com/AI4Bharat/IndicTrans2>
* fluidsynth — <https://www.fluidsynth.org/>

## 6. Attribution snippet

If you publish results, a minimal attribution block is:

```
Music generated with prompt_music_gen (Apache-2.0).
Trained on MidiCaps (CC BY-NC 4.0), derived from the Lakh MIDI Dataset.
Text encoder: google/mt5-base (Apache-2.0).
Vocal layer: facebook/mms-tts-tam (CC BY-NC 4.0).
Rendered with fluidsynth + a General MIDI soundfont.
```
