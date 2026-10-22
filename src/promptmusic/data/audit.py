"""Phase 6: auditing and rebalancing which instruments appear in the data.

The problem
-----------
MidiCaps is a slice of the Lakh MIDI Dataset, and the Lakh MIDI Dataset is a
scrape of what people uploaded to the internet. That population is *not*
balanced: guitar, piano and drums dominate, while sitar, koto, bagpipe and
shakuhachi are rare. A model trained naively on that distribution learns that
"a cheerful song" means "guitar" and will reach for a guitar almost every time.

Why this matters for this project specifically
-----------------------------------------------
Two of the stated goals are *every instrument* and *Hindustani ragas*. Both
are defeated by an unbalanced corpus: a raga conditioned on a name may still
come out on piano, and an uncommon General MIDI voice may never be selected at
all because the decoder rarely saw it in training.

What we do about it
-------------------
1. **Measure.** :func:`audit_instrument_distribution` walks the corpus and
   counts how often each General MIDI program occurs, both in raw MIDI files
   and in tokenised examples.
2. **Report.** :func:`rebalance_report` turns those counts into human-readable
   advice: which instruments are over-represented, which are absent, and what
   the entropy of the distribution is (entropy near ``log(N)`` means balanced).
3. **Correct.** :func:`write_rebalanced_index` produces a per-instrument
   sampling weight. The dataset loader then oversamples rare instruments and
   undersamples common ones *by example*, which is the cheapest possible fix -
   no resynthesis, no augmentation, just a different order of presentation.
4. **Optionally filter.** :func:`filter_by_instrument` can also restrict the
   corpus to specific programs, e.g. "give me only the pieces that contain
   a flute".

Rebalancing is a *data* operation, not a model change, so it can be toggled
in the config (``data.rebalance: true``) and ablated against a run without it.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..paths import OUTPUTS_DIR, ensure_dir
from ..utils import logger

LOG = logger("promptmusic.audit")

__all__ = [
    "ProgramStat",
    "InstrumentAudit",
    "audit_instrument_distribution",
    "rebalance_report",
    "write_rebalanced_index",
    "load_rebalanced_index",
    "filter_by_instrument",
    "gm_program_name",
]

# --------------------------------------------------------------------------- #
# General MIDI program names (0-127), the same table used by analyze.py.
# Kept here too so the audit module has no import cycle with analyze.
# --------------------------------------------------------------------------- #
_GM = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
    "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord",
    "Clavinet", "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ",
    "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ",
    "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)",
    "Acoustic Guitar (steel)", "Electric Guitar (jazz)",
    "Electric Guitar (clean)", "Electric Guitar (muted)",
    "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)",
    "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1",
    "Synth Bass 2", "Violin", "Viola", "Cello", "Contrabass",
    "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1",
    "Synth Strings 2", "Choir Aahs", "Voice Oohs", "Synth Voice",
    "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe",
    "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute", "Recorder",
    "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)",
    "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)",
    "Lead 8 (bass+lead)", "Pad 1 (new age)", "Pad 2 (warm)",
    "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)",
    "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)", "FX 1 (rain)",
    "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bag pipe", "Fiddle",
    "Shanai", "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
]


def gm_program_name(program: int) -> str:
    """General MIDI program number -> readable name (0-indexed)."""
    if 0 <= int(program) < len(_GM):
        return _GM[int(program)]
    return f"program_{int(program)}"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class ProgramStat:
    """How often one General MIDI program appears, and in what role."""

    program: int
    name: str
    files: int = 0          # distinct files containing this program
    tracks: int = 0         # track instances
    notes: int = 0          # note instances
    roles: Counter = field(default_factory=Counter)  # {"melody"|"chord"|"bass"|"drum"}

    @property
    def file_rate(self) -> float:
        """Fraction of audited files that use this program."""
        return self.files / max(self._total_files, 1)

    # Set by the audit that produced this stat.
    _total_files: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program,
            "name": self.name,
            "files": self.files,
            "tracks": self.tracks,
            "notes": self.notes,
            "file_rate": round(self.file_rate, 4),
            "roles": dict(self.roles),
        }


@dataclass
class InstrumentAudit:
    """The whole instrument picture of a corpus."""

    total_files: int = 0
    files_with_drums: int = 0
    stats: dict[int, ProgramStat] = field(default_factory=dict)
    duration_hours: float = 0.0
    path: str = ""

    @property
    def used_programs(self) -> list[int]:
        return sorted(self.stats)

    @property
    def unused_gm_programs(self) -> list[int]:
        """GM programs 0-127 that appear in zero files."""
        return [p for p in range(128) if p not in self.stats]

    @property
    def entropy_bits(self) -> float:
        """Shannon entropy of the file-level distribution, in bits.

        ``log2(128) = 7.0`` would be perfectly even across every GM voice;
        a real corpus lands much lower. Higher is more balanced.
        """
        counts = [s.files for s in self.stats.values() if s.files > 0]
        total = sum(counts)
        if total == 0:
            return 0.0
        h = 0.0
        for c in counts:
            p = c / total
            h -= p * math.log2(p)
        return h

    @property
    def gini(self) -> int:
        """How many General MIDI programs are absent (0 = none, 128 = all)."""
        return len(self.unused_gm_programs)

    @property
    def concentration(self) -> float:
        """Share of all file-appearances held by the single most common program."""
        total = sum(s.files for s in self.stats.values())
        if total == 0:
            return 0.0
        return max((s.files for s in self.stats.values()), default=0) / total

    def top(self, n: int = 10) -> list[ProgramStat]:
        return sorted(self.stats.values(), key=lambda s: -s.files)[:n]

    def bottom(self, n: int = 10) -> list[ProgramStat]:
        return sorted(
            (s for s in self.stats.values() if s.files > 0),
            key=lambda s: s.files,
        )[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_files": self.total_files,
            "files_with_drums": self.files_with_drums,
            "duration_hours": round(self.duration_hours, 2),
            "programs_used": len(self.stats),
            "programs_unused": self.gini,
            "entropy_bits": round(self.entropy_bits, 3),
            "concentration": round(self.concentration, 4),
            "top": [s.to_dict() for s in self.top(15)],
            "bottom": [s.to_dict() for s in self.bottom(15)],
            "path": self.path,
        }


# --------------------------------------------------------------------------- #
# Auditing
# --------------------------------------------------------------------------- #
def audit_instrument_distribution(
    paths: Iterable[Path | str] | None = None,
    *,
    root: Path | str | None = None,
    limit: int | None = None,
    roles: bool = True,
) -> InstrumentAudit:
    """Count instrument usage across a set of MIDI files.

    Args:
        paths: explicit MIDI files to audit. If ``None``, ``root`` is scanned.
        root: directory to search for ``*.mid`` / ``*.midi`` when ``paths``
            is not given.
        limit: stop after this many files (useful for a quick sample).
        roles: classify each track as melody/chord/bass by pitch register.
            This costs almost nothing and makes the report much more useful.

    Returns:
        An :class:`InstrumentAudit`. Files that fail to parse are skipped and
        counted, never fatal - a scrape of internet MIDI will always contain
        some broken files.

    Example::

        audit = audit_instrument_distribution(root="dataset/midicaps", limit=2000)
        for stat in audit.top(5):
            print(stat.name, stat.file_rate)
    """
    root_path = Path(root) if root else None
    if paths is None and root_path is None:
        raise ValueError("pass either `paths` or `root`")

    if paths is None:
        files: list[Path] = sorted(
            list(root_path.rglob("*.mid")) + list(root_path.rglob("*.midi"))  # type: ignore[union-attr]
        )
    else:
        files = [Path(p) for p in paths]

    if limit:
        files = files[: int(limit)]

    audit = InstrumentAudit()
    audit.path = str(root_path or f"{len(files)} explicit paths")
    broken = 0

    for i, path in enumerate(files):
        try:
            info = _file_programs(path, classify=roles)
        except Exception:  # noqa: BLE001 - broken MIDI must not stop the audit
            broken += 1
            continue

        if info is None:
            continue

        audit.total_files += 1
        audit.duration_hours += info["duration_sec"] / 3600.0
        if info["drums"]:
            audit.files_with_drums += 1

        seen_in_file: set[int] = set()
        for program, track_notes, role in info["tracks"]:
            if program is None:  # drum channel
                continue
            stat = audit.stats.get(program)
            if stat is None:
                stat = ProgramStat(program=program, name=gm_program_name(program))
                audit.stats[program] = stat
            stat.tracks += 1
            stat.notes += track_notes
            if roles:
                stat.roles[role] += 1
            if program not in seen_in_file:
                stat.files += 1
                seen_in_file.add(program)

        # Attach the file count denominator so file_rate works.
        for stat in audit.stats.values():
            stat._total_files = audit.total_files

        if (i + 1) % 500 == 0:
            LOG.info("  audited %d/%d files", i + 1, len(files))

    if broken:
        LOG.warning("%d files could not be parsed and were skipped", broken)
    LOG.info(
        "audit: %d files, %d GM programs used, %d unused, entropy %.2f bits",
        audit.total_files, len(audit.stats), audit.gini, audit.entropy_bits,
    )
    return audit


def _file_programs(path: Path | str, *, classify: bool) -> dict[str, Any] | None:
    """Program/notes/role per track, plus duration and a drum flag."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    duration = float(pm.get_end_time())
    if duration <= 0:
        return None

    tracks: list[tuple[int | None, int, str]] = []
    has_drums = False
    for inst in pm.instruments:
        if inst.is_drum:
            has_drums = True
            continue
        role = _classify_track(inst, duration) if classify else "unknown"
        tracks.append((int(inst.program), len(inst.notes), role))
    return {"duration_sec": duration, "tracks": tracks, "drums": has_drums}


def _classify_track(inst: Any, duration: float) -> str:
    """Heuristic role for a track, from its pitch register and note density.

    * **bass** - median pitch below C3 (48)
    * **chord** - many short notes sounding together (sustained harmony)
    * **melody** - everything else

    This is a *statistical* label used only to make the audit readable; it is
    never fed to the model.
    """
    if not inst.notes:
        return "unknown"
    pitches = sorted(n.pitch for n in inst.notes)
    median = pitches[len(pitches) // 2]
    if median < 48:
        return "bass"

    # How many notes overlap on average?
    events = []
    for n in inst.notes:
        events.append((n.start, 1))
        events.append((n.end, -1))
    events.sort()
    level = peak = 0
    for _, delta in events:
        level += delta
        peak = max(peak, level)
    avg_voices = peak / max(len(pitches), 1)
    if avg_voices >= 2.5 and duration > 0:
        return "chord"
    return "melody"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def rebalance_report(audit: InstrumentAudit, *, threshold: float = 0.02) -> str:
    """A human-readable summary of an audit, as plain text.

    Args:
        audit: the audit to describe.
        threshold: programs used by fewer than this fraction of files are
            flagged as *under-represented*; above twice it, *over-represented*.
    """
    lines: list[str] = []
    lines.append("=" * 66)
    lines.append("INSTRUMENT DISTRIBUTION AUDIT")
    lines.append("=" * 66)
    lines.append(f"files audited        : {audit.total_files}")
    lines.append(f"duration             : {audit.duration_hours:.1f} hours")
    lines.append(f"GM programs used     : {len(audit.stats)} / 128")
    lines.append(f"GM programs absent   : {audit.gini}")
    lines.append(f"files with drums     : {audit.files_with_drums}"
                 f" ({100 * audit.files_with_drums / max(audit.total_files, 1):.1f}%)")
    lines.append(f"entropy              : {audit.entropy_bits:.2f} bits"
                 f" (7.00 = perfectly even across 128 programs)")
    lines.append(f"top-program share    : {100 * audit.concentration:.1f}% of all appearances")
    lines.append("")

    lines.append("MOST COMMON")
    lines.append(f"  {'program':>7}  {'files':>7}  {'rate':>6}  name")
    for stat in audit.top(10):
        lines.append(
            f"  {stat.program:>7}  {stat.files:>7}  {100 * stat.file_rate:>5.1f}%  {stat.name}"
        )
    lines.append("")

    used = [s for s in audit.stats.values() if s.files > 0]
    rare = sorted(used, key=lambda s: s.files)[:10]
    if rare:
        lines.append("RAREST IN USE")
        lines.append(f"  {'program':>7}  {'files':>7}  {'rate':>6}  name")
        for stat in rare:
            lines.append(
                f"  {stat.program:>7}  {stat.files:>7}  {100 * stat.file_rate:>5.1f}%  {stat.name}"
            )
        lines.append("")

    absent = audit.unused_gm_programs
    if absent:
        names = ", ".join(gm_program_name(p) for p in absent[:12])
        more = f" (+{len(absent) - 12} more)" if len(absent) > 12 else ""
        lines.append(f"NEVER APPEAR ({len(absent)} programs)")
        lines.append(f"  {names}{more}")
        lines.append("")

    # Advice
    over = [s for s in audit.stats.values() if s.file_rate > 2 * threshold]
    under = [s for s in audit.stats.values() if 0 < s.file_rate < threshold]
    lines.append("RECOMMENDATION")
    if not used:
        lines.append("  No instruments found - check the MIDI directory.")
    else:
        lines.append(
            f"  * {len(over)} program(s) exceed {2 * threshold:.0%} of files. "
            "Cap their weight so the model does not default to them."
        )
        lines.append(
            f"  * {len(under)} program(s) fall below {threshold:.0%} of files. "
            "Oversample them so rare voices reach the model."
        )
        if audit.gini > 60:
            lines.append(
                f"  * {audit.gini} of 128 GM programs are absent entirely. "
                "Those instruments cannot be generated no matter what the "
                "prompt says; consider a targeted source for them."
            )
        if audit.concentration > 0.30:
            top = audit.top(1)[0]
            lines.append(
                f"  * One program ({top.name}) is {100 * audit.concentration:.0f}% "
                "of appearances. This is the single biggest source of bias."
            )
    lines.append("=" * 66)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Rebalancing weights
# --------------------------------------------------------------------------- #
def write_rebalanced_index(
    audit: InstrumentAudit,
    *,
    out_path: Path | str | None = None,
    strategy: str = "sqrt_invsqrt",
    target_entropy_bits: float | None = None,
) -> Path:
    """Write a JSON file mapping GM program -> sampling weight.

    Strategies
    ----------
    ``"uniform"``
        Every *used* program gets weight 1.0 (no correction).
    ``"sqrt_invsqrt"``
        Weight = ``1 / sqrt(file_rate)`` normalised to mean 1.0. A program
        used by 1% of files gets roughly 3x the pull of one used by 100%.
        This is the default: assertive enough to matter, gentle enough not to
        collapse the distribution onto the rarest voice.
    ``"full"``
        Weight = ``1 / file_rate`` normalised to mean 1.0 - fully flattens
        the distribution. Use only with a large dataset, because rare
        examples will then dominate every batch.

    The result is consumed by :func:`load_rebalanced_index`, which the dataset
    loader calls when ``data.rebalance`` is enabled.

    Returns:
        The path written.
    """
    if not audit.stats:
        raise ValueError("audit has no statistics; run the audit first")

    out = Path(out_path) if out_path else (OUTPUTS_DIR / "instrument_weights.json")
    ensure_dir(out.parent)

    rates = {p: s.file_rate for p, s in audit.stats.items() if s.files > 0}
    weights: dict[str, float] = {}

    if strategy == "uniform":
        raw = {p: 1.0 for p in rates}
    elif strategy == "sqrt_invsqrt":
        raw = {p: 1.0 / math.sqrt(max(r, 1e-6)) for p, r in rates.items()}
    elif strategy == "full":
        raw = {p: 1.0 / max(r, 1e-6) for p, r in rates.items()}
    else:
        raise ValueError(f"unknown strategy {strategy!r}; use uniform|sqrt_invsqrt|full")

    mean = sum(raw.values()) / len(raw)
    for p, w in raw.items():
        weights[str(p)] = round(w / mean, 4)

    payload = {
        "strategy": strategy,
        "target_entropy_bits": target_entropy_bits,
        "num_programs": len(weights),
        "mean_weight": 1.0,
        "weights": {gm_program_name(int(p)): w for p, w in sorted(weights.items(), key=lambda kv: int(kv[0]))},
        "_by_program": weights,  # machine-readable, program numbers as strings
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s (strategy=%s, %d programs)", out, strategy, len(weights))
    return out


def load_rebalanced_index(path: Path | str) -> dict[int, float]:
    """Read a weight file back into ``{program: weight}``.

    Programs absent from the file default to weight 1.0, so the index can
    never *remove* an instrument - only reweight it.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("_by_program") or data.get("weights") or {}
    out: dict[int, float] = {}
    for key, value in raw.items():
        try:
            # Keys are program numbers in `_by_program`, names in `weights`.
            prog = int(key)
        except ValueError:
            prog = _gm_lookup(str(key))
            if prog < 0:
                continue
        out[prog] = float(value)
    return out


def _gm_lookup(name: str) -> int:
    try:
        return _GM.index(name)
    except ValueError:
        return -1


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def filter_by_instrument(
    root: Path | str,
    *,
    include: list[int] | None = None,
    exclude: list[int] | None = None,
    require_all: bool = False,
    limit: int | None = None,
) -> list[Path]:
    """Return the MIDI files under ``root`` that match a program filter.

    Args:
        root: directory to scan.
        include: keep files containing *any* (or *all*, if ``require_all``)
            of these GM programs.
        exclude: drop files containing any of these programs.
        require_all: with ``include``, require every listed program present.
        limit: cap the result size.

    Example::

        # only pieces that contain a flute (GM 73) or sitar (GM 105)
        flute_and_sitar = filter_by_instrument(
            "dataset/midicaps", include=[73, 105],
        )
    """
    root_path = Path(root)
    files = sorted(list(root_path.rglob("*.mid")) + list(root_path.rglob("*.midi")))
    include_set = set(include or [])
    exclude_set = set(exclude or [])
    kept: list[Path] = []

    for path in files:
        try:
            programs = _programs_in(path)
        except Exception:  # noqa: BLE001
            continue
        if include_set:
            if require_all and not include_set.issubset(programs):
                continue
            if not require_all and not (include_set & programs):
                continue
        if exclude_set and (exclude_set & programs):
            continue
        kept.append(path)
        if limit and len(kept) >= int(limit):
            break

    LOG.info(
        "filter: %d/%d files matched (include=%s exclude=%s require_all=%s)",
        len(kept), len(files), sorted(include_set), sorted(exclude_set), require_all,
    )
    return kept


def _programs_in(path: Path | str) -> set[int]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    return {
        int(i.program) for i in pm.instruments if not i.is_drum
    }


# --------------------------------------------------------------------------- #
# Optional: a per-example weight map for the DataLoader
# --------------------------------------------------------------------------- #
def example_weights(
    audit: InstrumentAudit,
    programs_per_example: list[set[int]],
    strategy: str = "sqrt_invsqrt",
) -> list[float]:
    """Per-training-example weights derived from its instruments.

    Args:
        audit: the corpus audit providing base rates.
        programs_per_example: for each example, the GM programs it uses (in
            file order, matching the dataset order).
        strategy: as :func:`write_rebalanced_index`.

    Returns:
        One weight per example, normalised so the mean is 1.0. Pass the result
        to ``torch.utils.data.WeightedRandomSampler``.

    An example containing several rare instruments takes the *maximum* of
    their weights rather than the product, so one unusual voice is enough to
    lift the example without rare-instrument stacks swamping the batch.
    """
    weights_map = _weights_for_strategy(audit, strategy)
    out: list[float] = []
    for programs in programs_per_example:
        if not programs:
            out.append(1.0)
            continue
        w = max(weights_map.get(p, 1.0) for p in programs)
        out.append(w)
    mean = sum(out) / max(len(out), 1)
    if mean > 0:
        out = [w / mean for w in out]
    return out


def _weights_for_strategy(audit: InstrumentAudit, strategy: str) -> dict[int, float]:
    rates = {p: s.file_rate for p, s in audit.stats.items() if s.files > 0}
    if not rates:
        return {}
    if strategy == "uniform":
        raw = {p: 1.0 for p in rates}
    elif strategy == "full":
        raw = {p: 1.0 / max(r, 1e-6) for p in rates}
    else:
        raw = {p: 1.0 / math.sqrt(max(r, 1e-6)) for p, r in rates.items()}
    mean = sum(raw.values()) / len(raw)
    return {p: w / mean for p, w in raw.items()}
