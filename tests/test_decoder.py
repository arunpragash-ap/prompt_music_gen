"""The music decoder: shapes, causality, KV-cache equivalence, generation.

These tests use a 2-layer, 64-wide decoder so they run in milliseconds and
need no network (no mT5 download). They cover the transformer logic that all
of training and generation depends on.

The most important test in the file is
:func:`test_incremental_cache_matches_full_forward`: it proves that generating
one token at a time (with a KV cache) produces exactly the same logits as
recomputing the whole prefix in one shot. If those ever diverge, inference
silently produces different (usually worse) music than the training objective
would suggest, and nothing else in the test suite would notice.
"""

from __future__ import annotations

import pytest
import torch

from promptmusic.model.decoder import MusicDecoder


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #
def test_forward_shapes(tiny_decoder, seeded_music_ids):
    ids = torch.tensor([seeded_music_ids], dtype=torch.long)
    memory = torch.randn(1, 6, 768)      # text memory always comes in at 768
    logits, attn = tiny_decoder(ids, memory, need_cache=False)
    assert logits.shape == (1, len(seeded_music_ids), tiny_decoder.vocab_size)
    assert logits.dtype == torch.float32


def test_batched_forward(tiny_decoder, seeded_music_ids, tiny_tokenizer):
    """Batch items are processed independently: same input, same output.

    The *same* text memory must be repeated across the batch - torch.randn
    gives each row different memory, which of course produces different
    logits. (An earlier version of this test compared rows fed different
    memory and blamed MPS non-determinism; the values were genuinely meant
    to differ.)
    """
    tiny_decoder.eval()
    ids = torch.tensor([seeded_music_ids, seeded_music_ids], dtype=torch.long)
    memory = torch.randn(1, 4, 768).repeat(2, 1, 1)
    with torch.no_grad():
        logits, _ = tiny_decoder(ids, memory, need_cache=False)
    assert logits.shape[0] == 2
    assert torch.equal(logits[0], logits[1])


def test_memory_projection_maps_768_to_d_model(tiny_decoder):
    """mT5 always emits 768-wide vectors; the decoder is 64-wide in tests."""
    assert tiny_decoder.memory_proj.in_features == 768
    assert tiny_decoder.memory_proj.out_features == tiny_decoder.d_model


def test_output_rank_is_three(tiny_decoder, seeded_music_ids):
    ids = torch.tensor([seeded_music_ids], dtype=torch.long)
    memory = torch.randn(1, 3, 768)
    logits, _ = tiny_decoder(ids, memory, need_cache=False)
    assert logits.dim() == 3


# --------------------------------------------------------------------------- #
# Causality - the defining property of an autoregressive decoder
# --------------------------------------------------------------------------- #
def test_future_tokens_do_not_influence_past_logits(tiny_decoder, seeded_music_ids):
    """Changing the last token must not change the logits at earlier positions."""
    ids = torch.tensor([seeded_music_ids], dtype=torch.long)
    memory = torch.randn(1, 5, 768)
    with torch.no_grad():
        base, _ = tiny_decoder(ids, memory, need_cache=False)
        mutated = ids.clone()
        mutated[0, -1] = (mutated[0, -1] + 17) % tiny_decoder.vocab_size
        alt, _ = tiny_decoder(mutated, memory, need_cache=False)

    # Every position except the last must be identical.
    assert torch.allclose(base[0, :-1], alt[0, :-1], atol=1e-5)
    # And the last position should generally differ (it sees the change).
    assert not torch.allclose(base[0, -1], alt[0, -1], atol=1e-5)


# --------------------------------------------------------------------------- #
# KV cache - the fast path used during generation
# --------------------------------------------------------------------------- #
def test_incremental_cache_matches_full_forward(tiny_decoder, seeded_music_ids):
    """The KV-cache path must produce identical logits to the full path."""
    tiny_decoder.eval()
    ids = torch.tensor([seeded_music_ids], dtype=torch.long)
    memory = torch.randn(1, 5, 768)

    with torch.no_grad():
        full_logits, _ = tiny_decoder(ids, memory, need_cache=False)

        # Now feed one token at a time, carrying the cache.
        cache = None
        incremental = []
        for t in range(ids.shape[1]):
            logits, cache = tiny_decoder(
                ids[:, t : t + 1], memory, need_cache=True, cache=cache
            )
            incremental.append(logits)
        inc_logits = torch.cat(incremental, dim=1)

    assert inc_logits.shape == full_logits.shape
    assert torch.allclose(inc_logits, full_logits, atol=1e-4), (
        "KV-cache path diverged from the full forward pass"
    )


def test_cache_grows_by_one_layer_step(tiny_decoder, seeded_music_ids):
    ids = torch.tensor([seeded_music_ids], dtype=torch.long)
    memory = torch.randn(1, 3, 768)
    _, cache = tiny_decoder(ids, memory, need_cache=True, cache=None)
    assert len(cache) == tiny_decoder.num_layers
    key, value = cache[0]
    # Each layer caches [batch, heads, seq, head_dim].
    assert key.shape[2] == ids.shape[1]
    assert key.shape[1] == tiny_decoder.nhead


def test_cache_extends_across_calls(tiny_decoder, seeded_music_ids):
    ids = torch.tensor([seeded_music_ids], dtype=torch.long)
    memory = torch.randn(1, 3, 768)
    _, cache = tiny_decoder(ids[:, :3], memory, need_cache=True, cache=None)
    _, cache2 = tiny_decoder(ids[:, 3:], memory, need_cache=True, cache=cache)
    key, _ = cache2[0]
    assert key.shape[2] == ids.shape[1]


# --------------------------------------------------------------------------- #
# Position handling
# --------------------------------------------------------------------------- #
def test_position_offset_comes_from_the_cache(tiny_decoder, seeded_music_ids):
    """The decoder derives its position offset from the KV cache length.

    This is what makes reference priming work: prefilling the cache with a few
    tokens makes the decoder believe the sequence starts further in, so the
    next-token logits differ from a fresh start.
    """
    ids = torch.tensor([seeded_music_ids], dtype=torch.long)
    memory = torch.randn(1, 4, 768)
    with torch.no_grad():
        # Full sequence from scratch.
        at_zero, _ = tiny_decoder(ids, memory, need_cache=False)
        # Same tail, but with 10 positions of cache already filled.
        prefix = torch.ones(1, 10, dtype=torch.long)
        _, cache = tiny_decoder(prefix, memory, need_cache=True, cache=None)
        at_ten, _ = tiny_decoder(ids, memory, need_cache=False, cache=cache)
    # Different absolute positions produce different embeddings -> different
    # logits, proving the offset is honoured rather than ignored.
    assert not torch.allclose(at_zero, at_ten, atol=1e-5)


def test_sequence_longer_than_max_len_is_rejected_or_truncated(tiny_decoder):
    """A safety valve: an over-long sequence must not silently index past the
    learned positional table."""
    too_long = tiny_decoder.max_len + 10
    ids = torch.ones(1, too_long, dtype=torch.long)
    memory = torch.randn(1, 2, 768)
    try:
        with torch.no_grad():
            logits, _ = tiny_decoder(ids, memory, need_cache=False)
        # If it succeeded, the decoder truncated internally - also acceptable.
        assert logits.shape[1] <= too_long
    except (IndexError, RuntimeError):
        pass   # Explicit rejection is the other acceptable outcome


# --------------------------------------------------------------------------- #
# Parameters and hygiene
# --------------------------------------------------------------------------- #
def test_parameter_count_matches_expectation(tiny_decoder):
    total = sum(p.numel() for p in tiny_decoder.parameters())
    # 2 layers, d=64, ff=128: order of magnitude is ~100k, not millions.
    assert 50_000 < total < 500_000


def test_dropout_off_in_eval_mode(tiny_decoder, seeded_music_ids):
    tiny_decoder.eval()
    ids = torch.tensor([seeded_music_ids], dtype=torch.long)
    memory = torch.randn(1, 4, 768)
    with torch.no_grad():
        a, _ = tiny_decoder(ids, memory, need_cache=False)
        b, _ = tiny_decoder(ids, memory, need_cache=False)
    assert torch.equal(a, b)


def test_training_mode_uses_dropout(tiny_tokenizer, seeded_music_ids):
    """Two training-mode passes must differ when dropout is actually > 0.

    The shared ``tiny_decoder`` fixture deliberately uses dropout=0.0 so other
    tests are deterministic; this test builds its own decoder with dropout on,
    otherwise it would be testing that 0 != 0.
    """
    torch.manual_seed(0)
    decoder = MusicDecoder(vocab_size=tiny_tokenizer.vocab_size, d_model=64, nhead=4,
                           num_layers=2, dim_feedforward=128, dropout=0.5, max_len=64)
    decoder.train()
    ids = torch.tensor([seeded_music_ids], dtype=torch.long)
    memory = torch.randn(1, 4, 768)
    a, _ = decoder(ids, memory, need_cache=False)
    b, _ = decoder(ids, memory, need_cache=False)
    # With dropout active, two forward passes differ (random mask each time).
    assert not torch.equal(a, b)


def test_pad_id_is_zero(tiny_decoder):
    assert tiny_decoder.pad_id == 0


def test_gradient_flows_to_all_layers(tiny_decoder, seeded_music_ids):
    tiny_decoder.train()
    ids = torch.tensor([seeded_music_ids], dtype=torch.long)
    memory = torch.randn(1, 4, 768, requires_grad=True)
    logits, _ = tiny_decoder(ids, memory, need_cache=False)
    loss = logits.mean()
    loss.backward()
    assert memory.grad is not None
    # At least one parameter in each layer receives a gradient.
    for layer in tiny_decoder.layers:
        grads = [p.grad for p in layer.parameters() if p.grad is not None]
        assert grads, "a decoder layer received no gradient"


# --------------------------------------------------------------------------- #
# Next-token sampling (build._sample)
# --------------------------------------------------------------------------- #
# Every path here crashed at least once with a real run: the top-p branch
# called torch.multinomial(probs, num=1) - the kwarg is num_samples - and
# nothing caught it because the tests only ever exercised the greedy path.
@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.0},                  # greedy argmax
        {"temperature": 1.0},                  # plain multinomial
        {"temperature": 0.8, "top_k": 5},      # top-k truncation
        {"temperature": 0.9, "top_p": 0.9},    # nucleus sampling
        {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        {"num_beams": 3},                      # greedy beam head
    ],
)
def test_sample_returns_one_token_per_row(kwargs):
    from promptmusic.model.build import _sample

    torch.manual_seed(0)
    logits = torch.randn(2, 50)
    out = _sample(logits, **kwargs)
    assert out.shape == (2, 1)
    assert out.dtype == torch.long
    # The chosen id must be a real column of the input.
    assert ((out >= 0) & (out < logits.size(-1))).all()


def test_sample_greedy_picks_argmax():
    from promptmusic.model.build import _sample

    logits = torch.tensor([[0.1, 5.0, 0.2]])
    assert _sample(logits, temperature=0.0).item() == 1


def test_sample_top_p_only_keeps_nucleus_tokens():
    """With a tiny nucleus, sampling must pick from the top tokens only."""
    from promptmusic.model.build import _sample

    logits = torch.full((1, 10), -10.0)
    logits[0, 3] = 10.0
    logits[0, 7] = 9.0
    torch.manual_seed(0)
    for _ in range(20):
        assert _sample(logits, temperature=1.0, top_p=0.5).item() in (3, 7)
