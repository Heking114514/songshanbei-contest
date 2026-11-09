"""Optional NineToothed kernels for the MiniMax-H3 audio VAE."""

from __future__ import annotations

import os
import types

import torch
from torch.nn.utils import remove_weight_norm

from infinicore.nn.functional import snake_beta_ninetoothed
from infinicore.nn.functional.snake_beta_ninetoothed import prepare_snake_beta


def materialize_audio_weight_norm(audio_vae: torch.nn.Module) -> int:
    """Freeze legacy weight-norm hooks into ordinary inference weights."""
    materialized = 0
    for module in audio_vae.modules():
        if not (hasattr(module, "weight_g") and hasattr(module, "weight_v")):
            continue
        remove_weight_norm(module)
        materialized += 1
    return materialized


def _experiment_flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def _snake_experiment_options():
    prepared = _experiment_flag("H3_EXPERIMENT_SNAKE_PREPARED")
    strided = _experiment_flag("H3_EXPERIMENT_SNAKE_STRIDED")
    if not (prepared or strided):
        return None
    tile = int(os.environ.get("H3_EXPERIMENT_SNAKE_TILE_WIDTH", "512"))
    warps = int(os.environ.get("H3_EXPERIMENT_SNAKE_NUM_WARPS", "4"))
    if tile not in (128, 256, 512, 1024) or warps not in (1, 2, 4, 8):
        raise ValueError("Unsupported experimental SnakeBeta tile/warp policy")
    return dict(strided=strided, tile_width=tile, num_warps=warps)


def _raw_parameter_key(alpha: torch.Tensor, beta: torch.Tensor):
    try:
        versions = (alpha._version, beta._version)
    except RuntimeError:
        # Raw inference tensors cannot safely use a mutation-blind exp cache.
        return None
    return (
        id(alpha), alpha.data_ptr(), versions[0], alpha.dtype, alpha.device,
        tuple(alpha.shape), tuple(alpha.stride()),
        id(beta), beta.data_ptr(), versions[1], beta.dtype, beta.device,
        tuple(beta.shape), tuple(beta.stride()),
    )


def _snake_beta_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if self.training or torch.is_grad_enabled():
        return self._infinilm_snake_beta_original(hidden_states)

    alpha, beta = self.alpha, self.beta
    key = _raw_parameter_key(alpha, beta)
    if key is None or getattr(self, "_infinilm_snake_beta_key", None) != key:
        object.__setattr__(self, "_infinilm_snake_beta_alpha_exp", alpha.detach().exp())
        object.__setattr__(self, "_infinilm_snake_beta_beta_exp", beta.detach().exp())
        object.__setattr__(self, "_infinilm_snake_beta_key", key)
        # Retain raw tensors so object/data-pointer reuse cannot alias this key.
        object.__setattr__(self, "_infinilm_snake_beta_parameter_refs", (alpha, beta))
        object.__setattr__(self, "_infinilm_snake_beta_prepared", None)

    options = getattr(self, "_infinilm_snake_beta_experiment", None)
    prepared = None
    if options is not None:
        prepared = getattr(self, "_infinilm_snake_beta_prepared", None)
        signature = (tuple(hidden_states.shape), tuple(hidden_states.stride()),
                     hidden_states.dtype, hidden_states.device)
        if prepared is None or prepared.signature != signature:
            prepared = prepare_snake_beta(
                hidden_states,
                self._infinilm_snake_beta_alpha_exp,
                self._infinilm_snake_beta_beta_exp,
                **options,
            )
            # A single last-signature plan per module keeps the cache bounded.
            object.__setattr__(self, "_infinilm_snake_beta_prepared", prepared)

    if prepared is None:
        return snake_beta_ninetoothed(
            hidden_states,
            self._infinilm_snake_beta_alpha_exp,
            self._infinilm_snake_beta_beta_exp,
        )
    return snake_beta_ninetoothed(
        hidden_states,
        self._infinilm_snake_beta_alpha_exp,
        self._infinilm_snake_beta_beta_exp,
        prepared=prepared,
    )


def install_ninetoothed_audio_kernels(audio_vae: torch.nn.Module) -> int:
    """Replace BigVGAN SnakeBeta forwards while preserving checkpoint state."""
    options = _snake_experiment_options()
    installed = 0
    for module in audio_vae.modules():
        if module.__class__.__name__ != "MiniMaxH3AudioSnakeBeta":
            continue
        if getattr(module, "_infinilm_snake_beta_nt", False):
            continue
        if module.training:
            raise RuntimeError("Install NineToothed audio kernels after calling eval()")
        object.__setattr__(
            module,
            "_infinilm_snake_beta_original",
            module.forward,
        )
        object.__setattr__(
            module,
            "forward",
            types.MethodType(_snake_beta_forward, module),
        )
        object.__setattr__(module, "_infinilm_snake_beta_experiment", options)
        object.__setattr__(module, "_infinilm_snake_beta_prepared", None)
        object.__setattr__(module, "_infinilm_snake_beta_nt", True)
        installed += 1
    return installed
