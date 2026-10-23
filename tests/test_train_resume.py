"""Resume must restore trained weights, not silently re-initialise them.

This pins the fourth checkpoint-loading bug: ``save_checkpoint`` writes the
FULL ``MusicModel`` state dict (``text_encoder.*`` + ``decoder.*``), which
matches nothing when loaded into ``model.decoder`` with ``strict=False``.
Training then continued from random weights on top of a good checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn

from promptmusic.model.build import MusicModel
from promptmusic.model.decoder import MusicDecoder
from promptmusic.train import restore_resume_weights


class _DummyEncoder(nn.Module):
    output_dim = 16

    def forward(self, *args, **kwargs):  # pragma: no cover - never called here
        raise AssertionError("resume should not run the text encoder")


def _model() -> MusicModel:
    torch.manual_seed(0)
    decoder = MusicDecoder(
        vocab_size=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_len=32,
        text_dim=_DummyEncoder.output_dim,
    )
    return MusicModel(text_encoder=_DummyEncoder(), decoder=decoder)


def test_full_checkpoint_restores_decoder_weights() -> None:
    trained = _model()
    trained_state = trained.state_dict()

    fresh = _model()
    # Simulate a fresh/random decoder: zero everything, then resume.
    with torch.no_grad():
        for p in fresh.decoder.parameters():
            p.zero_()

    full, missing, unexpected = restore_resume_weights(fresh, trained_state, "best.pt")

    assert full is True
    assert not [k for k in missing if k.startswith("decoder.")]
    # The decoder is no longer zero: weights came from the checkpoint.
    assert any(p.abs().sum() > 0 for p in fresh.decoder.parameters())
    ref = trained_state["decoder.token_emb.weight"]
    assert torch.equal(fresh.decoder.token_emb.weight, ref)


def test_decoder_only_checkpoint_still_loads() -> None:
    trained = _model()
    fresh = _model()
    with torch.no_grad():
        for p in fresh.decoder.parameters():
            p.zero_()

    full, missing, unexpected = restore_resume_weights(
        fresh, trained.decoder.state_dict(), "last.pt"
    )
    assert full is False
    assert not missing
    assert not unexpected
    assert torch.equal(fresh.decoder.token_emb.weight, trained.decoder.token_emb.weight)


def test_empty_checkpoint_is_rejected_loudly() -> None:
    model = _model()
    try:
        restore_resume_weights(model, {}, "broken.pt")
    except RuntimeError as exc:
        assert "refusing" in str(exc)
    else:  # pragma: no cover - the assertion is the point
        raise AssertionError("an empty state dict must not resume silently")
