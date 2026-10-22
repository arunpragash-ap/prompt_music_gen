"""Configuration loading, merging and override behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptmusic.config import Config, available_configs, load_config, parse_override


def test_all_shipped_configs_load():
    """Every YAML in configs/ must parse and merge without error."""
    paths = available_configs()
    assert paths, "configs/ directory is empty or missing"
    for path in paths:
        cfg = load_config(path.stem)
        # A loaded config must expose the sections every command relies on.
        for section in ("project", "paths", "data", "tokenizer", "model", "train"):
            assert cfg.section(section), f"{path.name} is missing section {section!r}"


def test_named_and_path_loading_agree():
    named = load_config("smoke")
    explicit = load_config(Path("configs/smoke.yaml"))
    assert named.to_dict() == explicit.to_dict()


def test_defaults_are_complete():
    cfg = load_config(None)
    # Keys the training loop reads unconditionally.
    assert cfg.get("model.text_encoder.name") == "google/mt5-base"
    assert cfg.get("model.decoder.num_layers") == 12
    assert cfg.get("model.decoder.d_model") == 512
    assert cfg.get("train.batch_size") >= 1
    assert cfg.get("generate.temperature") > 0


def test_override_parsing():
    assert parse_override("train.lr=0.001") == (["train", "lr"], 0.001)
    assert parse_override("data.limit=100") == (["data", "limit"], 100)
    assert parse_override("project.name=hello") == (["project", "name"], "hello")
    assert parse_override("train.max_steps=null") == (["train", "max_steps"], None)
    assert parse_override("data.languages=[en,ta]") == (["data", "languages"], ["en", "ta"])


def test_override_bad_format_raises():
    with pytest.raises(ValueError):
        parse_override("no_equals_sign")
    with pytest.raises(ValueError):
        parse_override("=value")


def test_overrides_are_applied():
    cfg = load_config("smoke", ["train.lr=0.5", "data.limit=7"])
    assert cfg.get("train.lr") == 0.5
    assert cfg.get("data.limit") == 7


def test_override_wins_over_yaml():
    """CLI -o must beat the YAML file, which must beat built-in defaults."""
    assert load_config("default", ["data.limit=3"]).get("data.limit") == 3
    # default.yaml itself differs from the built-in DEFAULTS?
    # Either way the override is last, so this holds.
    assert load_config(None, ["data.limit=3"]).get("data.limit") == 3


def test_set_creates_intermediate_keys():
    cfg = Config()
    cfg.set("a.b.c", 1)
    assert cfg.get("a.b.c") == 1


def test_get_default_on_missing_key():
    cfg = Config()
    assert cfg.get("does.not.exist") is None
    assert cfg.get("does.not.exist", 42) == 42


def test_section_returns_dict():
    cfg = load_config("smoke")
    assert isinstance(cfg.section("train"), dict)
    # A section that does not exist yields {} rather than raising.
    assert cfg.section("nope") == {}


def test_contains_dunder():
    cfg = load_config(None)
    assert "train.lr" in cfg
    assert "train.definitely_not_a_key" not in cfg


def test_save_roundtrip(tmp_path):
    cfg = load_config("smoke")
    cfg.set("train.lr", 1.25e-4)
    out = cfg.save(tmp_path / "saved.yaml")
    reloaded = Config.load(out)
    assert reloaded.get("train.lr") == 1.25e-4


def test_smoke_config_is_actually_small():
    """The smoke config exists to be fast; guard against accidental bloat."""
    cfg = load_config("smoke")
    assert cfg.get("train.max_steps") is not None
    assert cfg.get("train.max_steps") <= 100
    assert cfg.get("data.limit") <= 1000
    assert cfg.get("tokenizer.max_tokens") <= 1024


def test_paper_size_matches_released_dimensions():
    """paper_size.yaml is the A/B baseline - its shape is a design decision."""
    cfg = load_config("paper_size")
    assert cfg.get("model.decoder.num_layers") == 18
    assert cfg.get("model.decoder.d_model") == 768
    assert cfg.get("model.decoder.nhead") == 12


def test_default_config_stays_on_laptop_budget():
    """12 layers x 512 dims is the chosen size; guard against silent drift."""
    cfg = load_config("default")
    dec = cfg.section("model.decoder")
    assert dec["num_layers"] == 12
    assert dec["d_model"] == 512
    assert dec["nhead"] == 8
    assert dec["dim_feedforward"] == 2048
