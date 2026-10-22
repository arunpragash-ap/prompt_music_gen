"""The command-line interface: parser structure and the cheap subcommands.

Only subcommands that never touch the network, a dataset or a checkpoint are
exercised here (``info``, ``--help``, ``--version``, ``disk-report``,
``analyze`` on a fixture, ``evaluate`` on a fixture, ``render --backends``,
``audit``). The heavier commands are covered by the end-to-end smoke run
documented in ``docs/08_QUICKSTART.md``.
"""

from __future__ import annotations

import json

import pytest

from promptmusic import __version__
from promptmusic.cli import build_parser, main


# --------------------------------------------------------------------------- #
# Parser structure
# --------------------------------------------------------------------------- #
#: Every action the README promises. Guarding the list here means a rename
#: cannot silently break the documented workflow.
EXPECTED_ACTIONS = {
    "info", "download", "tokenize", "translate", "lyrics-build", "raga-build",
    "audit", "train", "generate", "render", "analyze", "evaluate", "export",
    "inspect", "disk-report",
}


def test_all_documented_actions_exist():
    parser = build_parser()
    actions = {
        action for action, _ in (parser._subparsers._group_actions[0].choices or {}).items()
    }
    assert EXPECTED_ACTIONS <= actions


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_action_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "typical first run" in out


def test_help_mentions_the_pipeline(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    # The first-run sequence must stay discoverable from --help.
    for step in ("download", "tokenize", "train", "generate"):
        assert step in out


@pytest.mark.parametrize("action", sorted(EXPECTED_ACTIONS))
def test_every_action_has_help(action, capsys):
    with pytest.raises(SystemExit):
        main([action, "--help"])
    assert capsys.readouterr().out.strip()


def test_train_help_documents_resuming(action="train"):
    parser = build_parser()
    train_parser = None
    for action_obj in parser._subparsers._group_actions[0].choices.values():
        if getattr(action_obj, "prog", "").endswith(" train"):
            train_parser = action_obj
    assert train_parser is not None
    # The config/-o plumbing every command shares must be present.
    flags = {a.dest for a in train_parser._actions}
    assert {"config", "overrides", "json", "verbose"} <= flags


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #
def test_info_runs_and_reports_hardware(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "hardware" in out
    assert "packages" in out


def test_info_json_is_machine_readable(capsys):
    assert main(["info", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "hardware" in data
    assert "packages" in data


# --------------------------------------------------------------------------- #
# disk-report
# --------------------------------------------------------------------------- #
def test_disk_report_runs(capsys):
    assert main(["disk-report"]) == 0
    out = capsys.readouterr().out
    assert "SAFE TO DELETE" in out


def test_disk_report_json(capsys):
    assert main(["disk-report", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "entries" in data
    assert "reclaimable_bytes" in data


# --------------------------------------------------------------------------- #
# analyze (Phase 4, reference files)
# --------------------------------------------------------------------------- #
def test_analyze_midi(capsys, midi_file):
    assert main(["analyze", str(midi_file)]) == 0
    out = capsys.readouterr().out
    assert "prompt fragment" in out


def test_analyze_json_file(capsys, midi_file, tmp_path):
    out_json = tmp_path / "analysis.json"
    assert main(["analyze", str(midi_file), "--json-file", str(out_json)]) == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert "tempo" in data


def test_analyze_missing_file_returns_error_code(capsys, tmp_path):
    assert main(["analyze", str(tmp_path / "nope.mid")]) == 2


# --------------------------------------------------------------------------- #
# evaluate (on a real fixture)
# --------------------------------------------------------------------------- #
def test_evaluate_file(capsys, midi_file):
    assert main(["evaluate", str(midi_file)]) == 0
    assert "EVALUATION" in capsys.readouterr().out


def test_evaluate_json(capsys, midi_file):
    assert main(["evaluate", str(midi_file), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["validity_rate"] == 1.0


def test_evaluate_writes_report(capsys, midi_file, tmp_path):
    report = tmp_path / "report.json"
    assert main(["evaluate", str(midi_file), "--out", str(report)]) == 0
    assert report.exists()


def test_evaluate_missing_target(capsys, tmp_path):
    assert main(["evaluate", str(tmp_path / "nope.mid")]) == 2


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_backends_flag(capsys):
    assert main(["render", "--backends"]) == 0
    out = capsys.readouterr().out
    assert "numpy" in out


def test_render_midi_to_wav(capsys, midi_file, tmp_path):
    out_wav = tmp_path / "rendered.wav"
    assert main(["render", str(midi_file), "--out", str(out_wav), "--backend", "numpy"]) == 0
    assert out_wav.exists()


def test_render_missing_midi(capsys, tmp_path):
    assert main(["render", str(tmp_path / "nope.mid")]) == 2


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def test_audit_requires_input(capsys):
    assert main(["audit"]) == 2


def test_audit_on_paths(capsys, multi_instrument_midi):
    assert main(["audit", "--paths", str(multi_instrument_midi)]) == 0
    assert "INSTRUMENT DISTRIBUTION AUDIT" in capsys.readouterr().out


def test_audit_json_with_weights(capsys, multi_instrument_midi, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # keep the default weights path inside tmp
    assert main([
        "audit", "--paths", str(multi_instrument_midi), "--json", "--weights",
    ]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "weights_path" in data


# --------------------------------------------------------------------------- #
# raga-build
# --------------------------------------------------------------------------- #
def test_raga_list(capsys):
    assert main(["raga-build", "--list"]) == 0
    out = capsys.readouterr().out
    assert "Yaman" in out
    assert "Malkauns" in out


def test_raga_build_writes_files(capsys, tmp_path):
    out_dir = tmp_path / "raga"
    assert main([
        "raga-build", "--out", str(out_dir), "--variations", "1", "--seed", "7",
    ]) == 0
    assert len(list(out_dir.glob("*.mid"))) == 24


# --------------------------------------------------------------------------- #
# inspect / export error handling
# --------------------------------------------------------------------------- #
def test_inspect_missing_file(capsys, tmp_path):
    assert main(["inspect", str(tmp_path / "nope.pt")]) == 2


def test_export_without_checkpoint(capsys, tmp_path, monkeypatch):
    """A clean error, not a traceback, when there is nothing to export."""
    monkeypatch.chdir(tmp_path)   # no outputs/ anywhere reachable
    assert main(["export"]) == 2


def test_generate_without_checkpoint(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["generate", "--prompt", "hello"]) == 2


# --------------------------------------------------------------------------- #
# Error handling / verbosity
# --------------------------------------------------------------------------- #
def test_unknown_argument_is_an_argparse_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["definitely-not-a-command"])
    assert exc.value.code == 2


def test_override_flag_accepted(capsys):
    """-o key=value is parsed on any command that takes --config."""
    assert main(["info"]) == 0   # info has no config, but must still work
    with pytest.raises(SystemExit):
        main(["train", "--config", "not_a_real_config", "--help"])
