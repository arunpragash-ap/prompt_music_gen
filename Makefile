# Makefile for prompt_music_gen
# See README.md for full usage, or `make help`.
.PHONY: help venv install install-mac install-dev \
        download translate tokenize raga-build lyrics-build train generate render analyze evaluate export inspect \
        test lint format clean clean-outputs clean-data clean-all \
        repo-init disk-report

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
CLI ?= $(PYTHON) -m promptmusic.cli

help:
	@echo "prompt_music_gen - Makefile"
	@echo ""
	@echo "Environment"
	@echo "  make venv        - create .venv with uv (Python 3.12)"
	@echo "  make install     - install runtime deps into .venv"
	@echo "  make install-mac - install mac-friendly extras (librosa,sf2utils,tensorboard)"
	@echo "  make install-dev - install dev deps (pytest,ruff)"
	@echo ""
	@echo "Pipeline (phases 1-6)"
	@echo "  make download    - Phase 1: fetch MidiCaps"
	@echo "  make translate   - Phase 2: EN -> TA via IndicTrans2"
	@echo "  make tokenize    - Phase 1: MIDI -> REMI(+lyric) tokens"
	@echo "  make lyrics-build- Phase 3: build lyric vocabulary/interleave"
	@echo "  make raga-build  - Phase 5: generate synthetic Hindustani raga MIDI"
	@echo "  make train       - Train model (MPS-aware, 12L/512 by default)"
	@echo "  make analyze     - Phase 4: analyze a reference file"
	@echo "  make generate    - Text/ref prompt -> MIDI/WAV"
	@echo "  make render      - MIDI -> WAV"
	@echo "  make evaluate    - Compute metrics"
	@echo "  make export      - Export .pt / TorchScript / ONNX"
	@echo "  make inspect     - Inspect exported artifacts"
	@echo ""
	@echo "Repo / cleanup"
	@echo "  make repo-init   - git init + initial commit + branches"
	@echo "  make test        - run pytest"
	@echo "  make lint        - ruff check src tests"
	@echo "  make clean-outputs- remove outputs/ and runs/ (REGENERATE)"
	@echo "  make clean-data  - remove dataset/ and data/downloads/ (REGENERATE)"
	@echo "  make clean-all   - remove outputs, runs, dataset, .hf_cache, logs (REGENERATE)"
	@echo "  make disk-report - show what's reclaimable (see docs/10_DISK_CLEANUP.md)"

venv:
	uv venv --python 3.12 .venv

install:
	.venv/bin/pip install -e ".[mac,logging]"

install-mac:
	.venv/bin/pip install -e ".[mac,analysis,render,fluidsynth,logging]"

install-dev:
	.venv/bin/pip install -e ".[dev]"

download:
	$(CLI) download

translate:
	$(CLI) translate

tokenize:
	$(CLI) tokenize

lyrics-build:
	$(CLI) lyrics-build

raga-build:
	$(CLI) raga-build

train:
	$(CLI) train

analyze:
	$(CLI) analyze

generate:
	$(CLI) generate

render:
	$(CLI) render

evaluate:
	$(CLI) evaluate

export:
	$(CLI) export

inspect:
	$(CLI) inspect

repo-init:
	./scripts/init_repo.sh

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests 2>&1 | tail -20

format:
	$(PYTHON) -m ruff check --fix src tests 2>&1 | tail -10

clean-outputs:
	rm -rf outputs/ runs/ wandb/ lightning_logs/ logs/ *.mid *.midi *.wav *.flac *.npz 2>&1 | tail -2
	@echo "Removed outputs/runs/logs/waveform artifacts (regenerate with promptmusic train/generate/render)."

clean-data:
	rm -rf dataset/ data/downloads/ data/cache/ data/processed/ *.tar.gz *.tgz *.zip 2>&1 | tail -2
	@echo "Removed downloaded/dataset/cache (regenerate with promptmusic download)."

clean-all:
	rm -rf outputs/ runs/ wandb/ lightning_logs/ logs/ dataset/ data/downloads/ data/cache/ data/processed/ .hf_cache/ hf_cache/ *.mid *.midi *.wav *.flac *.npz *.pkl *.pt *.pth *.safetensors *.onnx *.torchscript *.gguf *.ckpt 2>&1 | tail -3
	@echo "Removed regenerable artifacts (models, dataset, outputs). See docs/10_DISK_CLEANUP.md before doing anything else."

disk-report:
	@./scripts/disk_report.sh
