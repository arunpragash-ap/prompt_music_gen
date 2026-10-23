# 02 — Architecture: how the model works

This page explains the model from the bottom up. No AI background assumed;
each idea is introduced with its plain-English meaning.

---

## 1. Why music becomes tokens

Neural networks work on sequences of numbers, not on pianos. The **REMI**
encoding (implemented by the `miditok` library) turns a MIDI file into a list
of tokens, one event at a time:

```
Bar_None  Position_0  Instrument_0  Pitch_60  Velocity_80  Duration_1.0.0
Position_16  Pitch_64  Velocity_72  Duration_1.0.0  ...
```

* `Bar` starts a measure.
* `Position` says where in the measure the event happens.
* `Pitch` is the MIDI note number (60 = middle C).
* `Velocity` is how hard the note is struck.
* `Instrument` is the General MIDI program (0 = piano, 40 = violin, …).
* `Duration` is how long the note lasts.

Because a song is now just a long sequence of symbols, generating music is the
same operation as generating text: predict the next symbol, over and over.
The music vocabulary for this project is **423 tokens**.

## 2. The two halves of the model

```
   prompt text                                    music tokens so far
 "a romantic tamil                            [BOS, Bar_None, Position_0, …]
  love song, gentle"                                     │
        │                                                ▼
        ▼                                     ┌──────────────────────┐
  mT5 tokenizer                               │  MusicDecoder        │
        │                                     │  12 layers × 512 dim │
        ▼                                     │  causal self-attn    │
 ┌──────────────┐   [B,T,768]   ┌───────────┐ │  cross-attn ◄──┐     │
 │ mT5 encoder  │──────────────►│ linear    │─┘               │     │
 │ FROZEN       │               │ 768 → 512 │─────────────────┘     │
 │ ~277 M params│               └───────────┘   text "memory"       │
 └──────────────┘                                                  │
                                                                   ▼
                                                          next-token probabilities
                                                          (over 423 music tokens)
```

**The text encoder is frozen.** `google/mt5-base` already understands English
and Tamil from pre-training on web text. Freezing it means we never spend
memory or compute updating 277 M parameters, and its language understanding
cannot be damaged by our small dataset. We only train the **projection** (768 →
512) and the **decoder**.

**The decoder is causal.** When predicting the token at position *i* it may
only look at positions *< i*, exactly like a language model. It also performs
**cross-attention** into the text memory at every layer, so every predicted
token can consult the prompt.

## 3. Position matters

Transformers have no inherent sense of order, so we add **positional
information** to the token embeddings. This project uses a fixed sinusoidal
encoding (the original Transformer scheme). A subtle consequence: positions
far beyond the lengths seen in training are extrapolated, not learned. If the
model was trained on 1 024-token examples and you ask for 2 560 tokens, the
tail of the song is generated from positions it never saw — quality can drop.
This is documented again in `configs/ta_love_test.yaml`.

## 4. Parameter budget

| Component | Parameters | Trained? |
|---|---|---|
| mT5-base encoder | ~277 M | no (frozen) |
| projection 768 → 512 | 0.4 M | yes |
| decoder (12 layers, 512 dim, 8 heads) | 51.3 M | yes |
| **Total** | **328.3 M** | **51.3 M trainable** |

The decoder's per-layer cost is dominated by the feed-forward network
(512 → 2 048 → 512) and the two attention blocks (self and cross). 51 M
trainable parameters is small enough to fine-tune on a laptop, large enough to
learn musical structure.

## 5. One training step

1. Tokenise a batch of captions → `input_ids [B, T_text]`.
2. Run the frozen encoder under `torch.no_grad()` → `memory [B, T_text, 768]`.
3. Feed the music tokens `[BOS, t1 … tN]` to the decoder, minus the last one.
4. The decoder predicts the next token at every position in one parallel pass
   (**teacher forcing**).
5. **Cross-entropy loss** against the true next tokens, padded positions
   ignored.
6. Backpropagate through the decoder, clip gradients to norm 1.0, take an
   AdamW step. The learning rate warms up briefly, then follows a cosine decay
   to 5 % of its peak.

## 6. One generation step

1. Encode the prompt once (frozen mT5 → memory).
2. Start the decoder with `[BOS]`.
3. Sample the next token from the model's distribution:
   * **temperature** — flatten/sharpen the distribution (1.0 = as learned),
   * **top-k** — only the k most likely tokens are eligible,
   * **top-p** — smallest set of tokens whose probabilities sum to p,
   * **repetition penalty** — down-weights already-used tokens.
4. Append the token, repeat until `<eos>` or `max_new_tokens`.
5. Decode the token ids back to MIDI (lyric tokens, if any, are stripped).

Sampling is stochastic: two runs with different seeds give different songs.
The seed is configurable (`--seed`, `PROMPTMUSIC_SEED`).

## 7. The code map

| File | Responsibility |
|---|---|
| `src/promptmusic/model/text_encoder.py` | Load mT5, tokenise prompts, expose frozen embeddings |
| `src/promptmusic/model/decoder.py` | Causal transformer layer, position encoding, KV-style step-by-step decoding |
| `src/promptmusic/model/build.py` | Glue: `MusicModel`, loss, sampling, checkpoint save/load |
| `src/promptmusic/data/tokenizer.py` | MIDI ⇄ REMI ids, tempo sanitisation, lyric-token handling |
| `src/promptmusic/data/dataset.py` | Torch datasets, collation, token-cache loading |
| `src/promptmusic/train.py` | Training loop, scheduler, validation, checkpoints |
| `src/promptmusic/generate.py` | Prompt → tokens → MIDI (+ optional render) |
| `src/promptmusic/render.py` | MIDI → WAV via fluidsynth/sf2 fallbacks |
| `src/promptmusic/vocals.py` | Optional lyric vocal layer (see 07) |

## 8. Why this design (and what was traded away)

* **Frozen mT5 instead of a trained text tower** — saves ~4 GB of optimizer
  state on an 8 GB machine, at the cost of not adapting language to music.
* **Small decoder instead of paper-size 768-dim / 18 layers** — 51 M instead
  of ~200 M trainable; `configs/paper_size.yaml` exists if you have the RAM.
* **REMI instead of audio tokens** — MIDI is small, interpretable and
  editable; the price is that timbre comes from a soundfont, not the model.
* **Teacher forcing** — fast and stable, but exposure to its own mistakes is
  limited; generation can drift on long sequences.
