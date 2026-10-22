"""Phase 5: synthesising MIDI that obeys Hindustani raga grammar.

The problem this solves
-----------------------
The model can only *create* with a raga if the raga appears in its training
data. Public raga datasets ship audio and labels, not note-level scores, and
audio would have to be transcribed before it could be tokenised - a lossy extra
step. So we do something more reliable: **generate the MIDI directly from the
grammar** in ``data/ragas/raga_table.yaml``.

That is not "fake data" in a pejorative sense. The scale rules, the vakra
(crooked) contours, the omitted notes in ascent and the pakad are real musical
constraints, and a model trained on hundreds of variations of them learns the
same constraints. Real recordings from the Raga Recognition Dataset and Saraga
are described in ``docs/01_DATASETS.md`` and can be mixed in for timbre and
phrasing diversity.

What a generated piece looks like
---------------------------------
::

    [ALAP]  slow, unmetered exploration - long notes, no drums
    [JOR]   the beat appears, still single-line
    [GAT]   a fixed composition repeats over a tabla cycle

That three-part shape is how a Hindustani performance actually unfolds, and
giving the model that structure means a generated piece has somewhere to go
instead of looping a single phrase.

Determinism
-----------
Every file is generated from a seed derived from ``(raga name, variation
index, global seed)``, so the same command always produces the same corpus.
That matters for reproducibility and for debugging a training run.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..paths import OUTPUTS_DIR, RAGAS_DIR, ensure_dir
from ..utils import logger

LOG = logger("promptmusic.raga")

__all__ = [
    "Raga",
    "load_raga_table",
    "swara_to_semitone",
    "build_scale",
    "RagaGenerator",
    "generate_raga_corpus",
    "list_raga_names",
]

DEFAULT_TABLE = RAGAS_DIR / "raga_table.yaml"


# --------------------------------------------------------------------------- #
# The raga model
# --------------------------------------------------------------------------- #
@dataclass
class Raga:
    """One row of the raga table, decoded into something programmable."""

    name: str
    thaat: str
    jati: str
    aroha: list[str]
    avaroha: list[str]
    pakad: list[str] = field(default_factory=list)
    vakra: bool = False
    vadi: str = ""
    samvadi: str = ""
    time: str = ""
    instruments: list[str] = field(default_factory=lambda: ["sitar"])
    source: str = ""
    notes: str = ""
    aliases: list[str] = field(default_factory=list)

    @property
    def scale_classes(self) -> set[int]:
        """The pitch classes this raga permits, as semitone offsets from Sa."""
        out: set[int] = set()
        for swara in [*self.aroha, *self.avaroha, *self.pakad]:
            out.add(swara_to_semitone(_base(swara)) % 12)
        return out

    @property
    def is_pentatonic(self) -> bool:
        return len(self.scale_classes) == 5

    def prompt_fragment(self) -> str:
        """The phrase appended to a caption so the model learns the name."""
        bits = [f"raga {self.name.replace('_', ' ')}"]
        if self.thaat:
            bits.append(f"{self.thaat} thaat")
        if self.time:
            bits.append(f"{self.time} mood")
        return "(" + ", ".join(bits) + ")"


def _base(swara: str) -> str:
    """Strip an octave marker: ``"S'"`` -> ``"S"``, ``"D,"`` -> ``"D"``."""
    return swara.rstrip("',")


def swara_to_semitone(swara: str, table: dict[str, int] | None = None) -> int:
    """Convert one swara (e.g. ``"M^'"``) to semitones above the tonic.

    Octave markers shift the result by ``+12`` (``'``) or ``-12`` (`,`).
    """
    table = table or _DEFAULT_SWARA
    base = _base(swara)
    if base not in table:
        raise KeyError(f"unknown swara {swara!r}; known: {sorted(table)}")
    offset = table[base]
    if swara.endswith("'"):
        offset += 12
    elif swara.endswith(","):
        offset -= 12
    return offset


_DEFAULT_SWARA = {
    "S": 0, "r": 1, "R": 2, "g": 3, "G": 4, "M": 5,
    "M^": 6, "P": 7, "d": 8, "D": 9, "n": 10, "N": 11,
}


def load_raga_table(path: Path | str | None = None) -> tuple[dict[str, int], list[Raga]]:
    """Read the YAML raga table into ``(swara_offsets, ragas)``."""
    p = Path(path) if path else DEFAULT_TABLE
    if not p.is_file():
        raise FileNotFoundError(
            f"raga table not found: {p}\n"
            "It ships with the repository at data/ragas/raga_table.yaml."
        )
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))

    offsets = {str(k): int(v) for k, v in (raw.get("swara_semitones") or _DEFAULT_SWARA).items()}
    ragas: list[Raga] = []
    for row in raw.get("ragas", []):
        ragas.append(
            Raga(
                name=row["name"],
                thaat=row.get("thaat", ""),
                jati=row.get("jati", ""),
                aroha=list(row.get("aroha", [])),
                avaroha=list(row.get("avaroha", [])),
                pakad=list(row.get("pakad", [])),
                vakra=bool(row.get("vakra", False)),
                vadi=row.get("vadi", ""),
                samvadi=row.get("samvadi", ""),
                time=row.get("time", ""),
                instruments=list(row.get("instruments", ["sitar"])),
                source=row.get("source", ""),
                notes=row.get("notes", ""),
                aliases=list(row.get("aliases", [])),
            )
        )
    LOG.info("loaded %d ragas from %s", len(ragas), p)
    return offsets, ragas


def list_raga_names(path: Path | str | None = None) -> list[str]:
    return [r.name for r in load_raga_table(path)[1]]


def build_scale(raga: Raga, tonic: int = 60, offsets: dict[str, int] | None = None) -> dict[str, int]:
    """Map each named swara to a concrete MIDI pitch.

    ``tonic`` is the MIDI note number of Sa (60 = C4). The same swara can
    appear in three octaves, so we return three keys per swara: the middle
    octave, the upper (``Sa'``) and the lower (``Sa,``).
    """
    offsets = offsets or _DEFAULT_SWARA
    out: dict[str, int] = {}
    for swara, semitone in offsets.items():
        out[swara] = tonic + semitone          # middle octave (Sa..Ni)
        out[f"{swara}'"] = tonic + semitone + 12
        out[f"{swara},"] = tonic + semitone - 12
    return out


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #
class RagaGenerator:
    """Turns a :class:`Raga` into a ``pretty_midi`` score.

    Args:
        raga: the raga to realise.
        offsets: the swara table (defaults to the standard one).
        tempo: BPM of the metered sections. Ragas have a conventional tempo
            range; this default sits in the middle of it.
        tonic: MIDI note for Sa. Vary it between samples so the model does not
            bind a raga to one absolute pitch.
    """

    # Section lengths in beats.
    ALAP_BEATS = 16
    JOR_BEATS = 16
    GAT_CYCLES = 4          # Teental = 16 beats, so 4 cycles = 64 beats

    GM_TABLA = {"dha": 64, "na": 63, "tin": 62, "ge": 61, "te": 60}
    GM_INSTRUMENT = {
        "sitar": 105, "sarod": 105, "bansuri": 73, "shehnai": 111,
        "santoor": 15, "harmonium": 20, "tanpura_drone": 48,
    }

    def __init__(
        self,
        raga: Raga,
        *,
        offsets: dict[str, int] | None = None,
        tempo: int = 72,
        tonic: int = 60,
        seed: int = 0,
    ):
        self.raga = raga
        self.offsets = offsets or _DEFAULT_SWARA
        self.tempo = max(40, min(160, int(tempo)))
        self.tonic = int(tonic)
        self.rng = random.Random(seed)
        self.scale = build_scale(raga, self.tonic, self.offsets)
        self.beat_sec = 60.0 / self.tempo

    # -- public ------------------------------------------------------------ #
    def build(self):
        """Compose the score and return a ``pretty_midi.PrettyMIDI``."""
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)
        lead = pretty_midi.Instrument(
            program=self.GM_INSTRUMENT.get(self.raga.instruments[0], 105),
            name=f"{self.raga.name}-lead",
        )
        drone = pretty_midi.Instrument(program=48, name="tanpura")
        tabla = pretty_midi.Instrument(program=0, is_drum=True, name="tabla")

        cursor = self._alap(lead)
        cursor = self._jor(lead, cursor)
        self._gat(lead, tabla, cursor)

        # The drone is added LAST so it can span the *actual* written music.
        # The alap randomises each note's length, so the real duration always
        # differs from the nominal beat count; measuring it avoids the drone
        # cutting out under the final phrase.
        span = max((n.end for n in lead.notes), default=0.0)
        self._add_drone(drone, span)

        pm.instruments.extend([drone, lead, tabla])
        return pm

    # -- sections ----------------------------------------------------------- #
    def _add_drone(self, drone, span: float) -> None:
        """A tanpura-like Sa-Pa-Sa' drone under the whole piece.

        Hindustani practice tunes the tanpura to the raga's most stable
        intervals: Sa-Pa-Sa'-Pa for most ragas, but **Sa-Ma** for ragas that
        omit Pa (Malkauns, Marwa and others marked as such in the table). A
        Pa drone under a Pa-less raga would contradict the raga on every beat,
        so the drone follows the grammar.

        Args:
            drone: the pretty_midi instrument to fill.
            span: total duration in seconds, measured from the finished lead.
        """
        import pretty_midi

        legal = self.raga.scale_classes
        # Pa is semitone 7 from Sa; fall back to Ma (5) when Pa is absent.
        interval = 7 if 7 in legal else (5 if 5 in legal else None)
        # Tanpura string order, low to high: Sa, Pa/Ma, Sa', Pa/Ma.
        if interval is None:
            strings = [(self.tonic, 42), (self.tonic + 12, 46)]
        else:
            strings = [
                (self.tonic, 42),
                (self.tonic + interval, 38),
                (self.tonic + 12, 46),
                (self.tonic + interval, 34),
            ]
        for pitch, velocity in strings:
            drone.notes.append(
                pretty_midi.Note(velocity=velocity, pitch=pitch, start=0.0, end=span)
            )

    def _alap(self, lead) -> float:
        """Slow, unmetered opening: one long note at a time, ascending.

        The alap introduces each swara in turn, dwelling on it. Notes are long
        and there is no beat, which is exactly how a performance begins.
        """
        import pretty_midi

        sequence = self._alap_sequence()
        time = 0.0
        for i, pitch in enumerate(sequence):
            # Longer dwell on vadi/samvadi-ish notes and on the tonic.
            base = 2.0 + (1.5 if pitch % 12 == 0 else 0.0)
            dur = base * self.beat_sec * self.rng.uniform(0.8, 1.4)
            velocity = int(min(110, 52 + i * 3 + self.rng.randint(-4, 4)))
            lead.notes.append(
                pretty_midi.Note(velocity=velocity, pitch=pitch, start=time, end=time + dur)
            )
            # A brief ornamental turn on roughly a third of the notes.
            if self.rng.random() < 0.35:
                grace = self._neighbour(pitch)
                if grace is not None:
                    g0 = time + dur - 0.08 * self.beat_sec
                    lead.notes.append(
                        pretty_midi.Note(
                            velocity=max(30, velocity - 14), pitch=grace,
                            start=g0, end=time + dur,
                        )
                    )
            time += dur
        return time

    def _alap_sequence(self) -> list[int]:
        """Choose which swaras the alap visits, in order."""
        ascent = [self.scale[s] for s in self.raga.aroha if s in self.scale]
        if not ascent:
            ascent = [self.tonic]
        # Visit each swara once, then revisit the lower half - that is the
        # shape of a real alap, which starts at Sa and works upward.
        out = list(ascent)
        out.extend(self.rng.sample(ascent[: max(3, len(ascent) // 2 + 1)],
                                   k=min(3, len(ascent))))
        return out

    def _jor(self, lead, start: float) -> float:
        """The middle section: a beat appears, still single-line."""
        import pretty_midi

        pattern = self._phrase(length=8)
        time = start
        cycles = self.JOR_BEATS // max(1, len(pattern))
        for _ in range(max(1, cycles)):
            for pitch in pattern:
                dur = self.beat_sec * self.rng.choice([0.5, 0.5, 1.0])
                lead.notes.append(
                    pretty_midi.Note(
                        velocity=self._velocity(), pitch=pitch,
                        start=time, end=time + dur * 0.92,
                    )
                )
                time += self.beat_sec
        return time

    def _gat(self, lead, tabla, start: float) -> None:
        """The composition: a fixed line repeated over a tabla cycle."""
        import pretty_midi

        gat = self._phrase(length=16)
        time = start
        for cycle in range(self.GAT_CYCLES):
            # The tabla enters only here, so the piece builds.
            self._tabla_cycle(tabla, time)
            for i, pitch in enumerate(gat):
                # Slight variation per cycle so it is not a literal loop.
                p = pitch
                if cycle > 0 and self.rng.random() < 0.18:
                    p = self._neighbour(pitch) or pitch
                dur = self.beat_sec * self.rng.choice([0.5, 0.5, 0.5, 1.0])
                lead.notes.append(
                    pretty_midi.Note(
                        velocity=self._velocity(), pitch=p,
                        start=time, end=time + dur * 0.92,
                    )
                )
                time += self.beat_sec
            # Land on Sa at the end of every cycle (the 'sam').
            lead.notes.append(
                pretty_midi.Note(
                    velocity=self._velocity(), pitch=self.tonic,
                    start=time, end=time + self.beat_sec,
                )
            )
            time += self.beat_sec

    # -- helpers ------------------------------------------------------------ #
    def _phrase(self, length: int = 8) -> list[int]:
        """Build a melodic phrase that respects aroha/avaroha and pakad."""
        ascent = [self.scale[s] for s in self.raga.aroha if s in self.scale]
        descent = [self.scale[s] for s in self.raga.avaroha if s in self.scale]
        if not ascent:
            ascent = [self.tonic]
        if not descent:
            descent = list(reversed(ascent))

        # Start from the pakad when the raga has one - the pakad is what makes
        # a raga recognisable, so it should appear in the training data.
        seed_phrase = [self.scale[s] for s in self.raga.pakad[:6] if s in self.scale]

        out: list[int] = []
        while len(out) < length:
            if seed_phrase and self.rng.random() < 0.5:
                out.extend(seed_phrase)
            elif self.rng.random() < 0.55:
                out.extend(ascent if self.rng.random() < 0.6 else list(reversed(ascent)))
            else:
                out.extend(descent)
        return out[:length]

    def _neighbour(self, pitch: int) -> int | None:
        """A note a step away *within the raga*, for ornamentation.

        Scale offsets are measured from Sa, not from C, so a legal neighbour
        is always built as ``tonic + offset (+/- octaves)``. Treating the
        offset as a pitch class would silently transpose the ornament outside
        the raga whenever Sa is not C - e.g. with Sa = E, "Pa" is 7 semitones
        above E (B, MIDI 71), not pitch class 7 (G).
        """
        offsets = sorted(self.raga.scale_classes)   # 0..11 relative to Sa
        if not offsets:
            return None
        # Every legal pitch within two octaves of the tonic.
        pool = [
            self.tonic + off + 12 * k
            for off in offsets
            for k in (-1, 0, 1, 2)
        ]
        # Keep only close neighbours, never the note itself, never a leap.
        close = [c for c in pool if 0 < abs(c - pitch) <= 4]
        if not close:
            return None
        return min(close, key=lambda c: abs(c - pitch))

    def scale_classes_abs(self) -> set[int]:
        return self.raga.scale_classes

    def _velocity(self) -> int:
        return int(min(112, max(42, self.rng.gauss(78, 10))))

    def _tabla_cycle(self, tabla, start: float) -> None:
        """One Teental (16-beat) cycle of light tabla strokes."""
        import pretty_midi

        # Dha Ge Na Ti | Na Ka Dhi Na | Dha Ge Na Ti | Na Ka Ta Na
        bols = [
            ["dha", "ge", "na", "tin"], ["na", "te", "dha", "na"],
            ["dha", "ge", "na", "tin"], ["na", "te", "tin", "na"],
        ]
        t = start
        for beat_group in bols:
            for bol in beat_group:
                pitch = self.GM_TABLA.get(bol, 63)
                tabla.notes.append(
                    pretty_midi.Note(
                        velocity=int(min(100, max(48, self.rng.gauss(74, 8)))),
                        pitch=pitch, start=t, end=t + 0.18 * self.beat_sec,
                    )
                )
                t += self.beat_sec


# --------------------------------------------------------------------------- #
# Corpus generation
# --------------------------------------------------------------------------- #
def generate_raga_corpus(
    *,
    table_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    variations_per_raga: int = 24,
    seed: int = 7,
    tonic_jitter: int = 0,
    format: str = "mid",
) -> dict[str, Any]:
    """Synthesise a corpus of raga MIDI files and a matching caption index.

    Args:
        table_path: where the YAML table lives.
        out_dir: destination; defaults to ``outputs/raga_corpus``.
        variations_per_raga: how many distinct pieces per raga. Tempo, tonic
            and phrase selection vary between them.
        seed: master seed. Reproducible.
        tonic_jitter: if > 0, shift each sample's Sa by a random number of
            semitones in ``[-tonic_jitter, +tonic_jitter]`` (clamped to a
            sensible MIDI range). Helps the model treat Sa as relative.
        format: ``"mid"`` (MIDI, used for training) or ``"wav"`` (also render
            audio - slower, needs a soundfont).

    Returns:
        A summary dict with the file count, per-raga counts and the output
        directory. A ``captions.json`` index is written alongside the files.
    """
    offsets, ragas = load_raga_table(table_path)
    out_dir = ensure_dir(Path(out_dir) if out_dir else OUTPUTS_DIR / "raga_corpus")

    index: list[dict[str, Any]] = []
    per_raga: dict[str, int] = {}
    failures: list[dict[str, str]] = []

    LOG.info(
        "generating %d variations x %d ragas -> %s",
        variations_per_raga, len(ragas), out_dir,
    )

    for raga in ragas:
        count = 0
        for i in range(variations_per_raga):
            rng_seed = _stable_seed(raga.name, i, seed)
            tempo = _tempo_for(raga, rng_seed)
            tonic = 60
            if tonic_jitter:
                j = random.Random(rng_seed).randint(-tonic_jitter, tonic_jitter)
                tonic = max(48, min(72, 60 + j))

            gen = RagaGenerator(raga, offsets=offsets, tempo=tempo, tonic=tonic, seed=rng_seed)
            try:
                pm = gen.build()
            except Exception as exc:  # noqa: BLE001 - one bad raga must not stop the run
                failures.append({"raga": raga.name, "error": str(exc)})
                LOG.warning("failed to generate %s variation %d: %s", raga.name, i, exc)
                continue

            fname = f"{raga.name.lower()}_{i:03d}.{format}"
            fpath = out_dir / fname
            try:
                pm.write(str(fpath))
            except Exception as exc:  # noqa: BLE001
                failures.append({"raga": raga.name, "error": str(exc)})
                continue

            index.append(_caption_for(raga, fname, tempo, tonic, pm))
            count += 1
        per_raga[raga.name] = count
        LOG.info("  %s: %d files", raga.name, count)

    index_path = out_dir / "captions.json"
    index_path.write_text(
        json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    total = sum(per_raga.values())
    summary = {
        "out_dir": str(out_dir),
        "index": str(index_path),
        "num_ragas": len(ragas),
        "num_files": total,
        "variations_per_raga": variations_per_raga,
        "seed": seed,
        "per_raga": per_raga,
        "failures": failures,
    }
    LOG.info("generated %d raga MIDI files (%d failures)", total, len(failures))
    return summary


def _caption_for(raga: Raga, fname: str, tempo: int, tonic: int, pm: Any) -> dict[str, Any]:
    """Build a MidiCaps-shaped caption record for a generated raga file.

    The shape matches MidiCaps so the two corpora can be concatenated without
    the trainer caring where a row came from.
    """
    jati_word = {
        "audav": "pentatonic", "shadav": "hexatonic",
        "sampurna": "heptatonic",
    }.get(raga.jati.split("-")[-1], raga.jati)
    shape = "vakra" if raga.vakra else "straight"

    caption = (
        f"A Hindustani classical solo in raga {raga.name.replace('_', ' ')}, "
        f"a {jati_word} {raga.thaat} thaat raga with a {shape} melodic contour, "
        f"played at {tempo} bpm in the {raga.time} mood, opening with a slow "
        f"alap and building into a tabla-accompanied composition."
    )
    duration = float(pm.get_end_time()) if hasattr(pm, "get_end_time") else 0.0
    return {
        "location": fname,
        "caption": caption,
        "genre": "hindustani classical",
        "genre_prob": 1.0,
        "mood": raga.time,
        "mood_prob": 1.0,
        "key": f"Sa={_pitch_name(tonic)}",
        "time_signature": "4/4",
        "tempo": tempo,
        "tempo_word": "moderate" if tempo < 90 else "fast",
        "duration": round(duration, 2),
        "duration_word": _duration_word(duration),
        "raga": raga.name,
        "thaat": raga.thaat,
        "jati": raga.jati,
        "vakra": raga.vakra,
        "vadi": raga.vadi,
        "samvadi": raga.samvadi,
        "source": raga.source,
        "synthetic": True,
    }


def _tempo_for(raga: Raga, rng_seed: int) -> int:
    """Pick a tempo in a range that suits the raga's time of day."""
    rng = random.Random(rng_seed)
    slow_times = {"midnight", "late morning", "pre-dawn", "early morning"}
    if raga.time in slow_times:
        return rng.randint(56, 76)
    if raga.time in {"any", "noon"}:
        return rng.randint(72, 96)
    return rng.randint(80, 112)


def _stable_seed(name: str, index: int, master: int) -> int:
    """A deterministic seed derived from (name, index, master)."""
    digest = hashlib.sha256(f"{name}:{index}:{master}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pitch_name(note: int) -> str:
    return f"{_PITCH_NAMES[note % 12]}{note // 12 - 1}"


def _duration_word(seconds: float) -> str:
    if seconds < 30:
        return "short"
    if seconds < 90:
        return "medium"
    return "long"
