"""CPU regressions for consecutive requests, including inference tensor mutation."""

import importlib.util
from pathlib import Path

import pytest
import torch

# The bookkeeping can be tested before the native InfiniLM extension is built.
SOURCE = Path(__file__).parents[2] / "python/infinilm/multimodal/minimax_h3_cache.py"
SPEC = importlib.util.spec_from_file_location("h3_cache", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Cache = MODULE.MiniMaxH3ConditionCache


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("mutated", ["text", "positions"])
def test_same_object_mutation_invalidates(inference, mutated):
    with torch.inference_mode(inference):
        text = torch.randn(1, 4, 8)
        positions = torch.arange(24).reshape(8, 3)
        cache = Cache()
        cache.begin_generation()
        cache.update(text, positions, (6, 2))
        assert cache.matches(text.clone(), positions.clone(), (6, 2))
        (text if mutated == "text" else positions).add_(1)
        assert not cache.matches(text, positions, (6, 2))
        old_revision = cache.revision
        cache.update(text, positions, (6, 2))
        assert cache.revision == old_revision + 1
        assert cache.matches(text.clone(), positions.clone(), (6, 2))


def test_data_alias_mutation_invalidates_without_version_counter_change():
    text, positions = torch.zeros(1, 4, 8), torch.zeros(8, 3)
    cache = Cache()
    cache.update(text, positions, ())
    version = text._version
    text.data.add_(1)
    assert text._version == version
    assert not cache.matches(text, positions, ())


def test_new_generation_invalidates_identical_input_and_preserves_revision():
    text, positions = torch.zeros(1, 4, 8), torch.zeros(8, 3)
    cache = Cache()
    assert cache.begin_generation() == 1
    cache.update(text, positions, ())
    assert cache.begin_generation() == 2
    assert not cache.matches(text, positions, ())
    cache.update(text, positions, ())
    assert cache.revision == 2


@pytest.mark.parametrize("change", ["text_shape", "position_shape", "text_dtype", "position_dtype", "geometry"])
def test_layout_and_dtype_invalidate(change):
    text, positions = torch.zeros(1, 4, 8), torch.zeros(8, 3)
    cache = Cache()
    metadata = (6, 2)
    cache.update(text, positions, metadata)
    if change == "text_shape":
        text = text.reshape(1, 8, 4)
    elif change == "position_shape":
        positions = positions[:4]
    elif change == "text_dtype":
        text = text.to(torch.bfloat16)
    elif change == "position_dtype":
        positions = positions.to(torch.int64)
    else:
        metadata = (4, 4)
    assert not cache.matches(text, positions, metadata)


def test_denoising_steps_reuse_unchanged_conditioning():
    text, positions = torch.randn(1, 4, 8), torch.randn(8, 3)
    cache = Cache()
    cache.begin_generation()
    cache.update(text, positions, (6, 2))
    # Varying latent/timestep values are not conditioning inputs and must not
    # force text projection or rotary construction on every denoising step.
    for _ in range(50):
        assert cache.matches(text, positions, (6, 2))
    assert cache.revision == 1
