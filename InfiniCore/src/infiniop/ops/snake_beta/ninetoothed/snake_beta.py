"""Dynamic-length NineToothed SnakeBeta kernel.

The kernel exponentiates the checkpoint's log-scale alpha and beta values so
those numerical operations stay inside the generated NineToothed program.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

_KERNEL_CACHE: dict[tuple[int, int, int, str], Callable] = {}


def make_kernel(block_m: int = 1, tile_width: int = 512, num_warps: int = 4):
    rows = block_size(block_m, block_m)

    def arrangement(x, alpha_exp, beta_exp, output):
        x_arranged = x.tile((rows, tile_width))
        alpha_arranged = (
            alpha_exp.tile((rows,))
            .unsqueeze(1)
            .expand((-1, x_arranged.shape[1]))
        )
        beta_arranged = (
            beta_exp.tile((rows,))
            .unsqueeze(1)
            .expand((-1, x_arranged.shape[1]))
        )
        return (
            x_arranged,
            alpha_arranged,
            beta_arranged,
            output.tile((rows, tile_width)),
        )

    def application(x, alpha_exp, beta_exp, output):
        alpha = math.exp(alpha_exp)
        beta = math.exp(beta_exp)
        sine = math.sin(alpha * x)
        output = x + sine * sine / (beta + 1e-9)  # noqa: F841

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(1), Tensor(1), Tensor(2)),
        num_warps=num_warps,
    )


def snake_beta(
    hidden_states: torch.Tensor,
    alpha_exp: torch.Tensor,
    beta_exp: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_m: int = 1,
    tile_width: int = 512,
    num_warps: int = 4,
) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError("SnakeBeta expects [batch, channels, length]")
    batch, channels, length = hidden_states.shape
    if alpha_exp.numel() != channels or beta_exp.numel() != channels:
        raise ValueError("SnakeBeta alpha_exp and beta_exp must have channels elements")

    x_rows = hidden_states.reshape(batch * channels, length)
    alpha_rows = alpha_exp.unsqueeze(0).expand(batch, -1).reshape(batch * channels)
    beta_rows = beta_exp.unsqueeze(0).expand(batch, -1).reshape(batch * channels)
    output = torch.empty_like(hidden_states) if out is None else out
    if not output.is_contiguous():
        raise ValueError("SnakeBeta output must be contiguous")

    key = (block_m, tile_width, num_warps, str(hidden_states.device))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = make_kernel(block_m, tile_width, num_warps)
        _KERNEL_CACHE[key] = kernel
    kernel(
        x_rows,
        alpha_rows.contiguous(),
        beta_rows.contiguous(),
        output.reshape(batch * channels, length),
    )
    return output
