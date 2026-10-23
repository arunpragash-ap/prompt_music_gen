"""``promptmusic`` - the command-line interface.

Every capability of this project is reachable from one command. The structure
is ``promptmusic <action>``, and each action has its own ``--help``::

    promptmusic info                       # what can this machine do?
    promptmusic download --source midicaps # get the data
    promptmusic tokenize                   # MIDI -> token cache
    promptmusic train --config smoke       # train
    promptmusic generate --prompt "..."    # make music
    promptmusic render <file.mid>          # MIDI -> WAV
    promptmusic evaluate outputs/...       # score it
    promptmusic export <checkpoint>        # package it

Action order for a first-time run
---------------------------------
The subcommands below are listed in the order a beginner should run them, and
the ``make`` targets in the Makefile wrap the same sequence::

    info -> download -> tokenize -> train -> generate -> evaluate -> export

Phases 2-6 add: ``translate`` (Tamil), ``lyrics-build``, ``analyze`` (reference
files), ``raga-build`` (Hindustani ragas) and ``audit`` (instrument balance).

Design notes for readers who have not written a CLI before
----------------------------------------------------------
* ``argparse`` is part of the Python standard library; we use subparsers so
  each action gets its own flag set.
* Every action function returns an ``int``: ``0`` means success, anything else
  is an error code. :func:`main` catches exceptions, prints a readable
  message (not a traceback) unless ``--verbose`` was passed, and exits with
  that code.
* Output is printed as a human summary by default and as JSON with ``--json``,
  so the same command works for both a terminal and a script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .config import DEFAULT_CONFIG, available_configs, load_config
from .paths import PROJECT_ROOT
from .utils import human_bytes, logger

LOG = logger("promptmusic.cli")


def list_configs() -> list[str]:
    """Names of the YAML configs in ``configs/``, plus the built-in default."""
    names = [p.stem for p in available_configs()]
    return names or ["default"]

__all__ = ["main", "build_parser"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _print(data: Any, as_json: bool, *, title: str | None = None) -> None:
    """Print a summary for humans, or JSON for scripts."""
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return
    if title:
        print(f"\n== {title} " + "=" * max(0, 66 - len(title)))
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)) and len(str(value)) > 200:
                print(f"{key:<24}:")
                print(f"  {json.dumps(value, indent=2, ensure_ascii=False, default=str)[:1500]}")
            else:
                print(f"{key:<24}: {value}")
    elif isinstance(data, list):
        for item in data:
            # A list of dicts (e.g. render/evaluate summaries) reads far
            # better as aligned key: value lines than as a raw repr.
            if isinstance(item, dict):
                for k, v in item.items():
                    if isinstance(v, (dict, list)) and len(str(v)) > 120:
                        print(f"  {k:<22}:")
                        print(f"    {json.dumps(v, ensure_ascii=False, default=str)[:400]}")
                    else:
                        print(f"  {k:<22}: {v}")
                print()
            else:
                print(f"  - {item}")
    else:
        print(data)
    if title:
        print("=" * 66)


def _load(args: argparse.Namespace):
    """Build a Config from ``--config`` plus any ``-o key=value`` overrides."""
    name = getattr(args, "config", None) or DEFAULT_CONFIG
    overrides = getattr(args, "overrides", None) or []
    return load_config(name, overrides)


def _add_common(parser: argparse.ArgumentParser, *, config: bool = True) -> None:
    """Flags every subcommand shares."""
    if config:
        parser.add_argument(
            "-c", "--config", metavar="NAME",
            help=f"config name or YAML path (default: {DEFAULT_CONFIG}; "
                 f"available: {', '.join(list_configs())})",
        )
        parser.add_argument(
            "-o", "--override", dest="overrides", action="append",
            metavar="KEY=VALUE", default=[],
            help="override a config value, e.g. -o data.limit=1000 (repeatable)",
        )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="show full tracebacks and debug logs")


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
def cmd_info(args: argparse.Namespace) -> int:
    """Environment report: hardware, packages, soundfont, data on disk."""
    from .utils import env_report

    report = env_report()
    if not args.json:
        print("\npromptmusic environment report")
        print("=" * 66)
        for section, entries in report.items():
            print(f"\n[{section}]")
            if isinstance(entries, dict):
                for k, v in entries.items():
                    print(f"  {k:<28}: {v}")
            else:
                print(f"  {entries}")
        print("\n" + "=" * 66)
        print(f"project root: {PROJECT_ROOT}")
        print(f"configs     : {', '.join(list_configs())}")
        print("=" * 66 + "\n")
    else:
        _print(report, True)
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download datasets and model files listed in the source registry."""
    from .data.download import download_source, list_sources

    if args.list:
        rows = [
            {
                "key": s["key"], "title": s["title"], "kind": s["kind"],
                "license": ", ".join(s["licenses"]), "homepage": s["homepage"],
                "files": s["files"],
                "bytes": s["total_bytes"],
            }
            for s in list_sources()
        ]
        if args.json:
            _print(rows, True, title="available sources")
        else:
            print(f"\n== available sources " + "=" * 47)
            for r in rows:
                mb = r["bytes"] / 1e6
                print(f"  {r['key']:<14} {mb:8.1f} MB  {r['files']} file(s)  [{r['license'] or '?'}]")
                print(f"  {'':<14} {r['title']}")
            print("=" * 66)
        return 0

    sources = [args.source] if args.source else [s["key"] for s in list_sources()]
    summaries = []
    for key in sources:
        try:
            summary = download_source(
                key,
                data_root=Path(args.dest) if args.dest else None,
                verify=not args.no_verify,
                extract=not args.no_extract,
                max_bytes=args.max_bytes,
            )
            summaries.append(summary)
        except Exception as exc:
            if args.json:
                summaries.append({"source": key, "error": str(exc)})
                continue
            print(f"error downloading {key}: {exc}", file=sys.stderr)
            return 2

    if args.json:
        _print(summaries, True)
    else:
        for s in summaries:
            print(f"\n{s.get('source', '?')}: {s.get('status', '')}")
            for k, v in s.items():
                if k not in {"source", "status", "files"}:
                    print(f"  {k:<22}: {v}")
    return 0


def cmd_tokenize(args: argparse.Namespace) -> int:
    """MIDI files + captions -> the token cache training reads."""
    from .data.dataset import build_tokenized_dataset
    from .data.tokenizer import build_tokenizer

    cfg = _load(args)
    t0 = time.time()
    tok = build_tokenizer(cfg.section("tokenizer"))
    examples = build_tokenized_dataset(
        cfg, tok,
        language=args.language,
        limit=args.limit if args.limit is not None else cfg.get("data.limit"),
        use_features=not args.caption_only,
        cache_path=args.cache,
    )
    lengths = sorted(len(e) for e in examples) or [0]
    summary = {
        "examples": len(examples),
        "vocab_size": tok.vocab_size,
        "language": args.language,
        "tokens_min": lengths[0],
        "tokens_median": lengths[len(lengths) // 2],
        "tokens_max": lengths[-1],
        "elapsed_sec": round(time.time() - t0, 1),
        "cache": args.cache or "(default)",
    }
    _print(summary, args.json, title="tokenized dataset")
    return 0


def cmd_translate(args: argparse.Namespace) -> int:
    """Phase 2: English captions -> Tamil."""
    from .data.captions import load_caption_index
    from .data.translate import TranslationEngine, save_translations, translate_records

    cfg = _load(args)
    caption_file = cfg.get("data.caption_file", "train.json")
    # Resolve relative to the data dir the same way the dataset loader does.
    candidates = [
        Path(caption_file),
        Path(cfg.get("paths.data_dir", "dataset")) / caption_file,
        Path(cfg.get("paths.data_dir", "dataset")) / "MidiCaps" / caption_file,
    ]
    caption_path = next((c for c in candidates if c.is_file()), None)
    if caption_path is None:
        print(
            "caption file not found. Run `promptmusic download --source midicaps` first.\n"
            "looked in:\n  " + "\n  ".join(str(c) for c in candidates),
            file=sys.stderr,
        )
        return 2

    limit = args.limit if args.limit is not None else cfg.get("data.translation.limit")
    records = load_caption_index(caption_path, limit=int(limit) * 4 if limit else None)

    engine = TranslationEngine(
        model_name=args.model or cfg.get("data.translation.model",
                                         "ai4bharat/indictrans2-en-indic-dist-200M"),
        device=args.device or "auto",
        batch_size=int(cfg.get("data.translation.batch_size", 16)),
    )
    translate_records(
        records, engine=engine, limit=int(limit) if limit else None,
        cache_dir=args.cache_dir,
    )

    out = args.out or (Path(cfg.get("paths.data_dir", "dataset")) / "captions_ta.jsonl")
    written = save_translations(records, out)
    done = sum(1 for r in records if r.caption_ta)
    _print(
        {"translated": done, "of": len(records), "output": str(written), "limit": limit},
        args.json, title="translation",
    )
    return 0


def cmd_lyrics_build(args: argparse.Namespace) -> int:
    """Phase 3: assemble the lyric corpus and vocabulary."""
    from .data.lyrics import download_lyrics_dataset, prepare_lyric_corpus

    cfg = _load(args)
    root = Path(args.root) if args.root else Path(cfg.get("paths.data_dir", "dataset")) / "lyrics_midi"

    if args.download or not root.exists():
        root = download_lyrics_dataset(root.parent, limit=args.limit)

    pairs, vocab = prepare_lyric_corpus(root, limit=args.limit, out_path=args.cache)
    tamil = sum(1 for p in pairs if p.is_tamil)
    summary = {
        "corpus_files": len(pairs),
        "tamil_files": tamil,
        "english_files": len(pairs) - tamil,
        "vocab_size": len(vocab),
        "vocab_mode": getattr(vocab, "mode", "?"),
        "root": str(root),
        "warning": "Lyrics-MIDI-Dataset is CC-BY-NC-SA-4.0 (non-commercial)",
    }
    _print(summary, args.json, title="lyrics corpus")
    return 0


def cmd_raga_build(args: argparse.Namespace) -> int:
    """Phase 5: synthesise Hindustani raga MIDI from the grammar table."""
    from .data.raga_gen import generate_raga_corpus, list_raga_names

    if args.list:
        _print(list_raga_names(args.table), args.json, title="ragas in table")
        return 0

    cfg = _load(args)
    summary = generate_raga_corpus(
        table_path=args.table,
        out_dir=args.out or (Path(cfg.get("paths.output_dir", "outputs")) / "raga_corpus"),
        variations_per_raga=args.variations,
        seed=args.seed,
        tonic_jitter=args.tonic_jitter,
    )
    _print(summary, args.json, title="raga corpus")
    return 0 if not summary.get("failures") else 1


def cmd_audit(args: argparse.Namespace) -> int:
    """Phase 6: instrument distribution audit and rebalancing weights."""
    from .data.audit import (
        audit_instrument_distribution,
        rebalance_report,
        write_rebalanced_index,
    )

    if not args.root and not args.paths:
        print("pass --root <midi dir> or --paths file.mid [file2.mid ...]", file=sys.stderr)
        return 2

    audit = audit_instrument_distribution(
        paths=args.paths or None, root=args.root, limit=args.limit,
    )

    if args.json:
        payload = audit.to_dict()
        if args.weights:
            payload["weights_path"] = str(
                write_rebalanced_index(audit, strategy=args.strategy)
            )
        _print(payload, True)
    else:
        print(rebalance_report(audit, threshold=args.threshold))
        if args.weights:
            path = write_rebalanced_index(audit, strategy=args.strategy)
            print(f"\nweights written to {path}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the model."""
    from .train import train

    cfg = _load(args)
    summary = train(cfg, run_name=args.run_name)
    _print(summary, args.json, title="training complete")
    if not args.json and summary.get("checkpoint"):
        print("\nnext:")
        print(f"  promptmusic generate --checkpoint {summary['checkpoint']} \\")
        print('           --prompt "a calm piano melody in a minor key"')
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Text prompt -> MIDI (+ WAV)."""
    from .generate import generate

    cfg = _load(args)
    checkpoint = args.checkpoint or cfg.get("train.checkpoint") or _default_checkpoint(cfg)
    if checkpoint is None:
        print(
            "no checkpoint given and none found. Train first:\n"
            "    promptmusic train --config smoke",
            file=sys.stderr,
        )
        return 2

    prompts = list(args.prompt or [])
    if args.prompt_file:
        prompts += [
            line.strip()
            for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not prompts:
        prompts = ["a simple piano melody"]

    # Lyrics: inline text wins, otherwise read the file.
    lyrics = args.lyrics or ""
    if args.lyrics_file:
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8").strip()

    # --duration sizes the token budget: tokens ~ seconds * density.
    max_tokens = args.max_tokens
    if args.duration and not max_tokens:
        tps = float(cfg.get("generate.tokens_per_second", 18.0))
        # +10% headroom so tempo jitter does not clip the tail.
        max_tokens = max(64, int(args.duration * tps * 1.1))

    results = []
    for prompt in prompts:
        results.extend(
            generate(
                prompt,
                checkpoint=checkpoint,
                config=cfg,
                out_dir=args.out,
                max_new_tokens=max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.beams,
                num_samples=args.num_samples,
                render_audio=not args.no_render,
                reference_midi=args.ref_midi,
                reference_audio=args.ref_audio,
                seed=args.seed,
                language=args.language,
                lyrics=lyrics,
            )
        )

    payload = [r.to_dict() for r in results]
    if args.json:
        _print(payload, True)
    else:
        print(f"\ngenerated {len(results)} sample(s):")
        for r in results:
            midi = r.midi_path or "-"
            wav = r.wav_path or "(audio skipped)"
            print(f"  {midi}")
            print(f"    {wav}   {r.num_notes} notes, {r.duration_sec:.1f}s")
    return 0


def cmd_arrange(args: argparse.Namespace) -> int:
    """Clean a generated MIDI: key, instrument roles, ranges, velocities."""
    from .arrange import arrange_midi

    src = Path(args.midi)
    if not src.is_file():
        print(f"not found: {src}", file=sys.stderr)
        return 2
    try:
        out, report = arrange_midi(
            src,
            args.out,
            max_instruments=args.max_instruments,
            snap_key=not args.no_snap,
            balance=not args.no_balance,
            fold_ranges=not args.no_fold,
            dedupe=not args.no_dedupe,
            sustain=not args.no_sustain,
        )
    except Exception as exc:
        print(f"arrange failed: {exc}", file=sys.stderr)
        return 1
    payload = report.to_dict()
    payload["out"] = str(out)
    _print(payload, args.json, title="arranged")
    for w in report.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """MIDI -> WAV using the best available synthesiser."""
    from .render import available_backends, humanize_midi, render_midi

    if args.backends:
        _print(available_backends(), args.json, title="rendering backends")
        return 0

    if not args.midi:
        print("no MIDI files given (or use --backends to list synthesizers)", file=sys.stderr)
        return 2

    cfg = _load(args)
    results = []
    mids = []
    missing = []
    for target in args.midi:
        p = Path(target)
        if p.is_dir():
            mids += sorted(p.glob("*.mid")) + sorted(p.glob("*.midi"))
        elif p.is_file():
            mids.append(p)
        else:
            missing.append(str(p))

    if missing:
        for m in missing:
            print(f"not found: {m}", file=sys.stderr)
        if not mids:
            return 2

    if not mids:
        print("no MIDI files found", file=sys.stderr)
        return 2

    for mid in mids:
        try:
            playable = mid
            if args.humanize:
                playable = humanize_midi(mid, seed=args.seed)
            r = render_midi(
                playable, args.out,
                backend=args.backend or cfg.get("render.backend", "auto"),
                soundfont=args.soundfont,
                sample_rate=args.rate or int(cfg.get("render.sample_rate", 44100)),
                gain=args.gain,
            )
            results.append(r.to_dict())
        except Exception as exc:
            if args.json:
                results.append({"midi": str(mid), "error": str(exc)})
            else:
                print(f"failed: {mid}: {exc}", file=sys.stderr)

    _print(results, args.json, title=f"rendered {len(results)} file(s)")
    failed = sum(1 for r in results if "error" in r)
    # A missing input path is a usage error (2); a render failure is a
    # partial failure (1); full success is 0.
    if missing and not results:
        return 2
    return 0 if failed == 0 else 1


def cmd_vocals(args: argparse.Namespace) -> int:
    """Speak the lyrics and mix them under a rendered instrumental.

    The music model emits MIDI, which has no voice track - until this
    stage exists, ``generate --lyrics ...`` only *conditions* the music
    on the text. See ``promptmusic.vocals`` for what this can and cannot
    do (it is a vocal layer, not melody-matched singing).
    """
    from .vocals import DEFAULT_VOICE, add_vocals

    midi = Path(args.midi)
    wav = Path(args.wav)
    for p in (midi, wav):
        if not p.exists():
            print(f"not found: {p}", file=sys.stderr)
            return 2
    if not args.lyrics and not args.lyrics_file:
        print("pass --lyrics or --lyrics-file (nothing to sing)", file=sys.stderr)
        return 2

    try:
        result = add_vocals(
            midi, wav,
            lyrics=args.lyrics,
            lyrics_file=args.lyrics_file,
            out_path=args.out,
            voice=args.voice or DEFAULT_VOICE,
            device=args.device,
            stride_bars=args.stride_bars,
            vocal_gain_db=args.gain,
        )
    except Exception as exc:
        print(f"vocals failed: {exc}", file=sys.stderr)
        return 1

    _print(asdict(result), args.json, title="vocal mix")
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def cmd_sing(args: argparse.Namespace) -> int:
    """Melody-aware singing: align the lyrics to the generated tune, master it.

    Unlike ``vocals`` (speech placed on bars), this extracts the lead melody
    from the MIDI, assigns syllables to its notes, pitch-shapes each syllable
    to the note, and runs a mastering chain. See ``promptmusic.singing`` for
    the honest limits of pitch-shaped TTS singing.
    """
    from .singing import DEFAULT_VOICE, sing_song

    midi = Path(args.midi)
    if not midi.exists():
        print(f"not found: {midi}", file=sys.stderr)
        return 2
    if args.wav and not Path(args.wav).exists():
        print(f"not found: {args.wav}", file=sys.stderr)
        return 2
    if not args.lyrics and not args.lyrics_file:
        print("pass --lyrics or --lyrics-file (nothing to sing)", file=sys.stderr)
        return 2

    try:
        result = sing_song(
            midi,
            lyrics=args.lyrics,
            lyrics_file=args.lyrics_file,
            out_path=args.out,
            instrumental_wav=args.wav,
            voice=args.voice or DEFAULT_VOICE,
            device=args.device,
            vocal_gain_db=args.gain,
            sample_rate=args.rate,
            master=not args.no_master,
            humanize=not args.no_humanize,
            humanize_seed=args.seed,
            render_backend=args.backend,
            stems=args.stems,
        )
    except Exception as exc:
        print(f"sing failed: {exc}", file=sys.stderr)
        return 1

    _print(result.to_dict(), args.json, title="sung mix")
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Phase 4: describe a reference MIDI or audio file."""
    from .analyze import (
        analysis_to_prompt_fragment,
        analyze_audio,
        analyze_midi,
        analyze_to_file,
        describe_analysis,
    )

    path = Path(args.reference)
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 2

    suffix = path.suffix.lower()
    if suffix in {".mid", ".midi"} or args.kind == "midi":
        analysis = analyze_midi(path)
    else:
        analysis = analyze_audio(path)

    if args.json_file:
        analyze_to_file(analysis, args.json_file)

    fragment = analysis_to_prompt_fragment(analysis)
    if args.json:
        analysis["prompt_fragment"] = fragment
        _print(analysis, True)
    else:
        print(f"\n{path.name}")
        print(f"  {describe_analysis(analysis)}")
        print(f"  prompt fragment: {fragment}")
        if fragment:
            print("\nuse it like:")
            print(f'  promptmusic generate --prompt "my idea {fragment}"')
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Score generated files, optionally against real music."""
    from .evaluate import evaluate_against_reference, evaluate_files, save_report

    target = Path(args.target)
    if target.is_dir():
        files = sorted(list(target.glob("**/*.mid")) + list(target.glob("**/*.midi")))
    elif target.exists():
        files = [target]
    else:
        # A path that does not exist is a usage error, not a bad file.
        print(f"not found: {target}", file=sys.stderr)
        return 2

    if not files:
        print(f"no MIDI files under {target}", file=sys.stderr)
        return 2

    kwargs: dict[str, Any] = {"keep_per_file": args.per_file}
    if args.expect_tempo:
        kwargs["expect_tempo"] = args.expect_tempo
    if args.expect_key:
        kwargs["expect_key"] = args.expect_key
    if args.expect_duration:
        kwargs["expect_duration"] = args.expect_duration
    if args.expect_instruments:
        kwargs["expect_instruments"] = [int(x) for x in args.expect_instruments]

    if args.reference:
        result = evaluate_against_reference(
            files, list(Path(args.reference).glob("**/*.mid")), **kwargs
        )
    else:
        result = evaluate_files(files, **kwargs)

    if args.out:
        save_report(result, args.out)

    if args.json:
        payload = result.to_dict()
        if not args.per_file:
            payload.pop("per_file", None)
        _print(payload, True)
    else:
        print(result.summary())
        if args.out:
            print(f"\nreport written to {args.out}")
    return 0 if result.validity_rate > 0 else 1


def cmd_export(args: argparse.Namespace) -> int:
    """Package a checkpoint for deployment."""
    from .export import export_model, export_and_report

    checkpoint = args.checkpoint
    if not checkpoint:
        cfg = _load(args)
        checkpoint = _default_checkpoint(cfg)
        if checkpoint is None:
            print("no checkpoint given and none found; pass --checkpoint", file=sys.stderr)
            return 2

    if args.all:
        reports = export_and_report(checkpoint, ["pt", "torchscript", "onnx"])
        _print(reports, args.json, title="export")
        return 0 if any("error" not in r for r in reports.values()) else 1

    result = export_model(
        checkpoint, out_path=args.out, fmt=args.format,
        opset=args.opset, dynamic_axes=not args.static,
    )
    _print(result.to_dict(), args.json, title="export")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Describe a checkpoint, export, ONNX file or vocabulary."""
    from .export import inspect_artifact

    try:
        info = inspect_artifact(args.path)
    except Exception as exc:
        print(f"could not inspect {args.path}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _print(info, True)
    else:
        print(f"\n{info['path']}")
        print(f"  kind        : {info.get('kind')}")
        print(f"  size        : {info.get('size_human')}")
        for key in (
            "num_parameters_m", "num_tensors", "step", "epoch", "metrics",
            "vocab_size", "text_encoder", "num_vocab_tokens", "num_nodes",
            "inputs", "groups", "keys", "sample_tokens",
        ):
            if key in info and info[key] not in (None, [], {}):
                print(f"  {key:<11}: {info[key]}")
        if info.get("note"):
            print(f"  note        : {info['note']}")
        print()
    return 0


def cmd_disk_report(args: argparse.Namespace) -> int:
    """What on disk is safe to delete (see docs/10_DISK_CLEANUP.md)."""
    from .utils import disk_usage_report

    report = disk_usage_report()
    if args.json:
        _print(report, True)
    else:
        print("\nDISK USAGE - what occupies space, and what is safe to delete")
        print("=" * 66)
        for row in report.get("entries", []):
            mark = "SAFE TO DELETE" if row.get("deletable") else "keep"
            print(f"  {row['path']:<46} {row['size_human']:>9}   [{mark}]")
            if row.get("reason"):
                print(f"      {row['reason']}")
        print("-" * 66)
        print(f"  total tracked: {report.get('total_human', '?')}")
        print(f"  reclaimable : {report.get('reclaimable_human', '?')}")
        print("=" * 66)
        print("full guidance: docs/10_DISK_CLEANUP.md\n")
    return 0


# --------------------------------------------------------------------------- #
# Small utilities used by more than one command
# --------------------------------------------------------------------------- #
def _default_checkpoint(cfg) -> str | None:
    """Find the best/latest checkpoint of the configured run, if any."""
    run_dir = Path(cfg.get("paths.output_dir", "outputs")) / cfg.get(
        "project.name", "prompt_music_gen"
    )
    ckpt = run_dir / "checkpoints"
    for name in ("best.pt", "last.pt"):
        p = ckpt / name
        if p.is_file():
            return str(p)
    # Any other run's checkpoint is better than failing the command.
    for p in sorted(Path(cfg.get("paths.output_dir", "outputs")).glob("*/checkpoints/best.pt")):
        return str(p)
    for p in sorted(Path(cfg.get("paths.output_dir", "outputs")).glob("*/checkpoints/last.pt")):
        return str(p)
    return None


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser (exposed so tests can introspect it)."""
    parser = argparse.ArgumentParser(
        prog="promptmusic",
        description=(
            "Text-to-music generation: turn a written prompt (English or Tamil) "
            "into MIDI and audio."
        ),
        epilog=(
            "typical first run:\n"
            "  promptmusic info\n"
            "  promptmusic download --source midicaps\n"
            "  promptmusic tokenize\n"
            "  promptmusic train --config smoke\n"
            "  promptmusic generate --prompt \"a bright piano study\"\n"
            "\nfull documentation lives in docs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"promptmusic {__version__}")
    sub = parser.add_subparsers(dest="action", metavar="<action>")

    # --- info ------------------------------------------------------------- #
    p = sub.add_parser("info", help="report hardware, packages and data availability")
    _add_common(p, config=False)
    p.set_defaults(func=cmd_info)

    # --- download --------------------------------------------------------- #
    p = sub.add_parser(
        "download",
        help="download datasets/models (MidiCaps, soundfont, reference weights)",
        description="Fetch files from the source registry with checksum verification.",
    )
    p.add_argument("-s", "--source", help="one source key; omit to download all")
    p.add_argument("--list", action="store_true", help="list available sources and exit")
    p.add_argument("--dest", help="destination directory (default: data/downloads)")
    p.add_argument("--no-verify", action="store_true", help="skip SHA-256 verification")
    p.add_argument("--no-extract", action="store_true", help="do not unpack archives")
    p.add_argument("--max-bytes", type=int, help="abort if a download exceeds this size")
    _add_common(p)
    p.set_defaults(func=cmd_download)

    # --- tokenize --------------------------------------------------------- #
    p = sub.add_parser(
        "tokenize",
        help="convert MIDI + captions into the token cache training reads",
        description="Stream the caption index, encode each MIDI with miditok, cache to JSONL.",
    )
    p.add_argument("--language", choices=["en", "ta"], default="en",
                   help="which caption to condition on (ta requires `promptmusic translate`)")
    p.add_argument("--limit", type=int, help="max examples (default: data.limit from config)")
    p.add_argument("--caption-only", action="store_true",
                   help="ignore the numeric feature fields; use the raw caption")
    p.add_argument("--cache", help="explicit path for the JSONL cache")
    _add_common(p)
    p.set_defaults(func=cmd_tokenize)

    # --- translate -------------------------------------------------------- #
    p = sub.add_parser(
        "translate",
        help="Phase 2: translate English captions to Tamil (IndicTrans2)",
        description="Incremental and cached; interrupting and restarting resumes.",
    )
    p.add_argument("--limit", type=int, help="max NEW captions to translate")
    p.add_argument("--model", help="Hugging Face translator id (default from config)")
    p.add_argument("--device", choices=["auto", "cpu", "mps"], help="inference device")
    p.add_argument("--out", help="output JSONL path")
    p.add_argument("--cache-dir", help="directory holding the translation cache")
    _add_common(p)
    p.set_defaults(func=cmd_translate)

    # --- lyrics-build ----------------------------------------------------- #
    p = sub.add_parser(
        "lyrics-build",
        help="Phase 3: download/scan the lyric corpus and build its vocabulary",
        description="Uses asigalov61/Lyrics-MIDI-Dataset (CC-BY-NC-SA-4.0, non-commercial).",
    )
    p.add_argument("--root", help="directory containing lyric MIDI files")
    p.add_argument("--limit", type=int, help="max files to process")
    p.add_argument("--download", action="store_true", help="force a (re)download")
    p.add_argument("--cache", help="explicit path for the corpus cache")
    _add_common(p)
    p.set_defaults(func=cmd_lyrics_build)

    # --- raga-build ------------------------------------------------------- #
    p = sub.add_parser(
        "raga-build",
        help="Phase 5: synthesise Hindustani raga MIDI from data/ragas/raga_table.yaml",
        description="Generates alap + jor + gat structure obeying each raga's grammar.",
    )
    p.add_argument("--list", action="store_true", help="list ragas in the table and exit")
    p.add_argument("--table", help="path to raga_table.yaml")
    p.add_argument("--out", help="output directory (default: outputs/raga_corpus)")
    p.add_argument("--variations", type=int, default=24, help="files per raga (default 24)")
    p.add_argument("--seed", type=int, default=7, help="master seed for reproducibility")
    p.add_argument("--tonic-jitter", type=int, default=0,
                   help="randomly shift Sa by up to N semitones per file")
    _add_common(p)
    p.set_defaults(func=cmd_raga_build)

    # --- audit ------------------------------------------------------------ #
    p = sub.add_parser(
        "audit",
        help="Phase 6: instrument distribution audit and rebalancing weights",
        description="Shows which General MIDI programs dominate the corpus and why it matters.",
    )
    p.add_argument("--root", help="MIDI directory to scan")
    p.add_argument("--paths", nargs="+", help="explicit MIDI files instead of --root")
    p.add_argument("--limit", type=int, help="max files to audit")
    p.add_argument("--threshold", type=float, default=0.02,
                   help="file-rate cutoff for over/under-representation (default 0.02)")
    p.add_argument("--weights", action="store_true", help="also write sampling weights")
    p.add_argument("--strategy", choices=["uniform", "sqrt_invsqrt", "full"],
                   default="sqrt_invsqrt", help="weighting strategy (default sqrt_invsqrt)")
    _add_common(p)
    p.set_defaults(func=cmd_audit)

    # --- train ------------------------------------------------------------ #
    p = sub.add_parser(
        "train",
        help="train the model (MPS/CPU aware)",
        description="AdamW + warmup/cosine, gradient accumulation, CSV logs, resumable.",
    )
    p.add_argument("--run-name", help="subdirectory of outputs/ for this run")
    _add_common(p)
    p.set_defaults(func=cmd_train)

    # --- generate --------------------------------------------------------- #
    p = sub.add_parser(
        "generate",
        help="text prompt -> MIDI (+ WAV)",
        description="Generate one or more samples from a prompt.",
    )
    p.add_argument("-p", "--prompt", action="append",
                   help="the music description (repeatable)")
    p.add_argument("--prompt-file", help="file with one prompt per line")
    p.add_argument("--checkpoint", help="path to a trained .pt (default: best found)")
    p.add_argument("--out", help="output directory (default: outputs/samples)")
    p.add_argument("-n", "--num-samples", type=int, default=1, help="samples per prompt")
    p.add_argument("--temperature", type=float, help="sampling temperature (default 1.0)")
    p.add_argument("--top-k", type=int, help="top-k sampling (default 50)")
    p.add_argument("--top-p", type=float, help="nucleus sampling (default 0.95)")
    p.add_argument("--beams", type=int, help="beam search width (default 1 = none)")
    p.add_argument("--max-tokens", type=int, help="max new music tokens")
    p.add_argument("--duration", type=float, metavar="SEC",
                   help="target length in seconds; sizes --max-tokens for you "
                        "(actual duration depends on tempo/density)")
    p.add_argument("--lyrics", help="lyric text to condition on (English or Tamil)")
    p.add_argument("--lyrics-file", help="file containing the lyrics (UTF-8)")
    p.add_argument("--seed", type=int, help="random seed (default from config)")
    p.add_argument("--no-render", action="store_true", help="skip WAV rendering")
    p.add_argument("--ref-midi", help="Phase 4: prime generation from this MIDI file")
    p.add_argument("--ref-audio", help="Phase 4: analyse this audio file for style")
    p.add_argument("--language", choices=["en", "ta"], default="en")
    _add_common(p)
    p.set_defaults(func=cmd_generate)

    # --- arrange ---------------------------------------------------------- #
    p = sub.add_parser(
        "arrange",
        help="clean a generated MIDI (key, instruments, ranges, balance)",
        description=(
            "Detect the key, keep a sensible set of instrument roles, snap "
            "out-of-key notes, fold ranges and balance velocities."
        ),
    )
    p.add_argument("--midi", required=True, help="input .mid file")
    p.add_argument("--out", help="output .mid (default: <name>_arranged.mid)")
    p.add_argument("--max-instruments", type=int, default=6,
                   help="non-drum instruments to keep (default 6)")
    p.add_argument("--no-snap", action="store_true", help="do not snap notes to the key")
    p.add_argument("--no-balance", action="store_true", help="do not rebalance velocities")
    p.add_argument("--no-fold", action="store_true", help="do not fold ranges by role")
    p.add_argument("--no-dedupe", action="store_true", help="keep duplicate notes")
    p.add_argument("--no-sustain", action="store_true",
                   help="keep the model's short note lengths (more plinky)")
    _add_common(p)
    p.set_defaults(func=cmd_arrange)

    # --- render ----------------------------------------------------------- #
    p = sub.add_parser(
        "render",
        help="MIDI -> WAV (fluidsynth, then fallbacks)",
        description="Accepts files or directories. Use --backends to see what is available.",
    )
    p.add_argument("midi", nargs="*", help=".mid files or directories containing them")
    p.add_argument("--out", help="output WAV path (single file only)")
    p.add_argument("--backend", choices=["auto", "fluidsynth", "pyfluidsynth", "numpy"],
                   help="synthesiser to use (default auto)")
    p.add_argument("--soundfont", help="explicit .sf2 path")
    p.add_argument("--rate", type=int, help="sample rate (default 44100)")
    p.add_argument("--gain", type=float, default=0.8, help="output gain 0..1")
    p.add_argument("--humanize", action="store_true",
                   help="nudge note timing/velocity before rendering (less mechanical)")
    p.add_argument("--seed", type=int, default=0, help="humanise seed (default 0)")
    p.add_argument("--backends", action="store_true", help="list backends and exit")
    _add_common(p)
    p.set_defaults(func=cmd_render)

    # --- vocals ----------------------------------------------------------- #
    p = sub.add_parser(
        "vocals",
        help="speak the lyrics and mix them under a rendered instrumental",
        description=(
            "The music model emits MIDI, which has no voice track. This stage "
            "speaks each lyric line with a local TTS model, aligns the lines "
            "to the MIDI's bars, and mixes them under the instrumental WAV. "
            "A vocal layer - not melody-matched singing (see docs/07)."
        ),
    )
    p.add_argument("--midi", required=True, help="the generated .mid (for tempo/bar timing)")
    p.add_argument("--wav", required=True, help="the rendered instrumental .wav")
    p.add_argument("--out", help="output path (default: <wav>_vocal.wav)")
    p.add_argument("--lyrics", help="lyric text (English or Tamil)")
    p.add_argument("--lyrics-file", help="UTF-8 file with one lyric line per line")
    p.add_argument("--voice", help="transformers VITS TTS repo id (default: Tamil MMS)")
    p.add_argument("--stride-bars", type=int,
                   help="bars between lyric lines (default: spread evenly)")
    p.add_argument("--gain", type=float, default=-4.0,
                   help="vocal level in dB relative to the music (default -4)")
    p.add_argument("--device", default="cpu",
                   help="speech model device (cpu or mps; default cpu)")
    _add_common(p)
    p.set_defaults(func=cmd_vocals)

    # --- sing (melody-aware) ---------------------------------------------- #
    p = sub.add_parser(
        "sing",
        help="sing the lyrics on the generated melody, then master the mix",
        description=(
            "Extract the lead melody from the MIDI, split the lyrics into "
            "syllables, assign each syllable to a note, pitch-shape the spoken "
            "syllable to that note (with vibrato), then EQ/compress/reverb/limit "
            "the result. This is pitch-shaped TTS singing, not a neural singing "
            "voice - see docs/07_VOCALS.md."
        ),
    )
    p.add_argument("--midi", required=True, help="the generated .mid (melody + timing)")
    p.add_argument("--wav", help="use this instrumental instead of rendering the MIDI")
    p.add_argument("--out", help="output WAV (default: <midi>_sing.wav)")
    p.add_argument("--lyrics", help="lyric text (Tamil or English)")
    p.add_argument("--lyrics-file", help="UTF-8 file, one lyric line per line")
    p.add_argument("--voice", help="transformers VITS TTS repo id (default: Tamil MMS)")
    p.add_argument("--device", default="cpu", help="speech model device (default cpu)")
    p.add_argument("--gain", type=float, default=1.5,
                   help="vocal RMS level in dB relative to the music (default +1.5)")
    p.add_argument("--rate", type=int, default=44100, help="output sample rate")
    p.add_argument("--no-master", action="store_true",
                   help="skip EQ/compression/reverb/limiting")
    p.add_argument("--no-humanize", action="store_true",
                   help="render the MIDI exactly as written")
    p.add_argument("--seed", type=int, default=0, help="humanise seed (default 0)")
    p.add_argument("--stems", action="store_true",
                   help="also write <out>_vocals.wav and <out>_instrumental.wav")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "fluidsynth", "pyfluidsynth", "numpy"],
                   help="renderer to use when --wav is not given")
    _add_common(p)
    p.set_defaults(func=cmd_sing)

    # --- analyze ---------------------------------------------------------- #
    p = sub.add_parser(
        "analyze",
        help="Phase 4: describe a reference MIDI/audio file",
        description="Prints tempo, key, instruments and a prompt fragment you can reuse.",
    )
    p.add_argument("reference", help="path to a .mid or audio file")
    p.add_argument("--kind", choices=["midi", "audio"], help="force the analysis mode")
    p.add_argument("--json-file", help="also write the full analysis as JSON")
    _add_common(p)
    p.set_defaults(func=cmd_analyze)

    # --- evaluate --------------------------------------------------------- #
    p = sub.add_parser(
        "evaluate",
        help="score generated MIDI (optionally against real music)",
        description="Reference-free metrics; add --reference for divergence scores.",
    )
    p.add_argument("target", help="a MIDI file or a directory of them")
    p.add_argument("--reference", help="directory of real MIDI to compare against")
    p.add_argument("--expect-tempo", type=float, help="tempo the prompt requested")
    p.add_argument("--expect-key", help='key the prompt requested, e.g. "C major"')
    p.add_argument("--expect-duration", type=float, help="requested length in seconds")
    p.add_argument("--expect-instruments", nargs="+", help="GM program numbers requested")
    p.add_argument("--per-file", action="store_true", help="include the per-file table")
    p.add_argument("--out", help="write the JSON report here")
    _add_common(p)
    p.set_defaults(func=cmd_evaluate)

    # --- export ----------------------------------------------------------- #
    p = sub.add_parser(
        "export",
        help="package a checkpoint (pt / torchscript / onnx)",
        description="Produces a lean artifact without optimizer state.",
    )
    p.add_argument("checkpoint", nargs="?", help="checkpoint path (default: best found)")
    p.add_argument("-f", "--format", choices=["pt", "torchscript", "onnx"],
                   default="pt", help="output format (default pt)")
    p.add_argument("--all", action="store_true", help="export every available format")
    p.add_argument("--out", help="output file path")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--static", action="store_true",
                   help="fixed ONNX shapes instead of dynamic axes")
    _add_common(p)
    p.set_defaults(func=cmd_export)

    # --- inspect ---------------------------------------------------------- #
    p = sub.add_parser(
        "inspect",
        help="describe a checkpoint/export/vocabulary without loading it fully",
    )
    p.add_argument("path", help="file to inspect")
    _add_common(p, config=False)
    p.set_defaults(func=cmd_inspect)

    # --- disk-report ------------------------------------------------------ #
    p = sub.add_parser(
        "disk-report",
        help="show what occupies disk and what is safe to delete",
        description="Companion to docs/10_DISK_CLEANUP.md.",
    )
    _add_common(p, config=False)
    p.set_defaults(func=cmd_disk_report)

    return parser


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code (0 = success)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "action", None):
        parser.print_help()
        return 0

    # --json implies tracebacks are unhelpful to humans but useful to scripts.
    verbose = bool(getattr(args, "verbose", False))
    if verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    func: Callable[[argparse.Namespace], int] = args.func
    started = time.time()
    try:
        code = func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError) as exc:
        # Expected, user-fixable errors: print the message, not a traceback.
        print(f"\nerror: {exc}", file=sys.stderr)
        if verbose:
            traceback.print_exc()
        return 2
    except Exception as exc:  # noqa: BLE001
        if verbose:
            traceback.print_exc()
        else:
            print(f"\nunexpected error: {exc}", file=sys.stderr)
            print("run again with -v for a full traceback", file=sys.stderr)
        return 1

    elapsed = time.time() - started
    if elapsed > 5 and not getattr(args, "json", False):
        print(f"\n[{args.action} finished in {elapsed:.1f}s]")
    return int(code or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
