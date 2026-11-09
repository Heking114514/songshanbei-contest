"""NineToothed dynamic-length SnakeBeta over torch tensors."""

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
        sine = math.sin(alpha_exp * x)
        output = x + sine * sine / (beta_exp + 1e-9)  # noqa: F841

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(1), Tensor(1), Tensor(2)),
        num_warps=num_warps,
    )


def get_kernel(
    device: torch.device,
    *,
    block_m: int = 1,
    tile_width: int = 512,
    num_warps: int = 4,
):
    key = (block_m, tile_width, num_warps, str(device))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = make_kernel(block_m, tile_width, num_warps)
        _KERNEL_CACHE[key] = kernel
    return kernel


def snake_beta(
    hidden_states: torch.Tensor,
    alpha_exp: torch.Tensor,
    beta_exp: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_m: int = 1,
    tile_width: int = 512,
    num_warps: int = 4,
    prepared: PreparedSnakeBeta | None = None,
) -> torch.Tensor:
    """Apply SnakeBeta using precomputed exponentiated parameters.

    Args:
        hidden_states: Contiguous `[batch, channels, length]` tensor.
        alpha_exp: `[channels]` tensor equal to `exp(alpha_log)`.
        beta_exp: `[channels]` tensor equal to `exp(beta_log)`.
        out: Optional contiguous output tensor.
    """
    if prepared is not None:
        if alpha_exp is not prepared.alpha_exp or beta_exp is not prepared.beta_exp:
            raise ValueError("Prepared SnakeBeta must retain the same exponentiated parameters")
        return prepared(hidden_states, out=out)

    if hidden_states.ndim != 3:
        raise ValueError("SnakeBeta expects [batch, channels, length]")
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    batch, channels, length = hidden_states.shape
    if alpha_exp.numel() != channels or beta_exp.numel() != channels:
        raise ValueError("SnakeBeta alpha_exp and beta_exp must have channels elements")

    alpha_rows = (
        alpha_exp.unsqueeze(0).expand(batch, -1).reshape(batch * channels).contiguous()
    )
    beta_rows = (
        beta_exp.unsqueeze(0).expand(batch, -1).reshape(batch * channels).contiguous()
    )
    output = torch.empty_like(hidden_states) if out is None else out
    if not output.is_contiguous() or output.shape != hidden_states.shape:
        raise ValueError("SnakeBeta output must be contiguous with the input shape")

    kernel = get_kernel(
        hidden_states.device,
        block_m=block_m,
        tile_width=tile_width,
        num_warps=num_warps,
    )
    kernel(
        hidden_states.reshape(batch * channels, length),
        alpha_rows,
        beta_rows,
        output.reshape(batch * channels, length),
    )
    return output


_STRIDED_KERNEL_CACHE: dict[tuple[int, int, str], Callable] = {}


def make_strided_kernel(tile_width: int = 512, num_warps: int = 4):
    """Tile B/C/L directly; runtime strides preserve channel padding."""
    def arrangement(x, alpha_exp, beta_exp, output):
        return (
            x.tile((1, 1, tile_width)),
            alpha_exp.tile((1, 1, tile_width)),
            beta_exp.tile((1, 1, tile_width)),
            output.tile((1, 1, tile_width)),
        )

    def application(x, alpha_exp, beta_exp, output):
        sine = math.sin(alpha_exp * x)
        output = x + sine * sine / (beta_exp + 1e-9)  # noqa: F841

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(3), Tensor(3), Tensor(3), Tensor(3)),
        num_warps=num_warps,
        num_stages=1,
        kernel_name="h3_snake_beta_strided_v2",
    )


def _parameter_version(tensor: torch.Tensor) -> int | None:
    # Inference tensors may not have a version counter. Such prepared exp
    # parameters are immutable by contract; the model wrapper tracks raw weights.
    try:
        return tensor._version
    except RuntimeError:
        return None


class PreparedSnakeBeta:
    """One bounded, read-only parameter plan for one shape/stride signature.

    Holds parameters and views, never an activation or a reusable output buffer.
    Rebuild after changing an inference tensor without a version counter.
    """

    def __init__(
        self,
        hidden_states: torch.Tensor,
        alpha_exp: torch.Tensor,
        beta_exp: torch.Tensor,
        *,
        strided: bool = False,
        tile_width: int = 512,
        num_warps: int = 4,
    ):
        if hidden_states.ndim != 3:
            raise ValueError("SnakeBeta expects [batch, channels, length]")
        if tile_width not in (128, 256, 512, 1024) or num_warps not in (1, 2, 4, 8):
            raise ValueError("Unsupported experimental SnakeBeta tile/warp policy")
        batch, channels, length = hidden_states.shape
        if min(batch, channels, length) <= 0:
            raise ValueError("Prepared SnakeBeta requires nonempty dimensions")
        if alpha_exp.numel() != channels or beta_exp.numel() != channels:
            raise ValueError("SnakeBeta parameters must have channels elements")
        if alpha_exp.device != hidden_states.device or beta_exp.device != hidden_states.device:
            raise ValueError("SnakeBeta parameters and activations must share a device")
        if any(stride < 0 for stride in hidden_states.stride()):
            raise ValueError("Prepared SnakeBeta does not support negative strides")
        self.signature = self.input_signature(hidden_states)
        self.alpha_exp, self.beta_exp = alpha_exp, beta_exp
        self.parameter_versions = (_parameter_version(alpha_exp), _parameter_version(beta_exp))
        self.strided = strided
        self.batch, self.channels, self.length = batch, channels, length
        if strided:
            # Zero-stride broadcast views: no B*C repeats and no full input copy.
            self.alpha = alpha_exp.reshape(1, channels, 1).expand(batch, channels, length)
            self.beta = beta_exp.reshape(1, channels, 1).expand(batch, channels, length)
            key = (tile_width, num_warps, str(hidden_states.device))
            kernel = _STRIDED_KERNEL_CACHE.get(key)
            if kernel is None:
                kernel = make_strided_kernel(tile_width, num_warps)
                _STRIDED_KERNEL_CACHE[key] = kernel
            self.kernel = kernel
        else:
            # Keep the original 2D kernel; prepare the duplicated channel rows once.
            self.alpha = alpha_exp.reshape(1, channels).expand(batch, channels).reshape(-1).contiguous()
            self.beta = beta_exp.reshape(1, channels).expand(batch, channels).reshape(-1).contiguous()
            self.kernel = get_kernel(
                hidden_states.device, block_m=1, tile_width=tile_width, num_warps=num_warps
            )

    @staticmethod
    def input_signature(tensor: torch.Tensor):
        return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)

    def __call__(self, hidden_states: torch.Tensor, *, out: torch.Tensor | None = None):
        if self.input_signature(hidden_states) != self.signature:
            raise ValueError("Rebuild the prepared SnakeBeta plan after shape/stride/device/dtype changes")
        # Avoid repeatedly raising on cached immutable inference tensors.
        versions = tuple(
            _parameter_version(parameter) if known is not None else None
            for parameter, known in zip(
                (self.alpha_exp, self.beta_exp), self.parameter_versions
            )
        )
        if versions != self.parameter_versions:
            raise ValueError("Rebuild the prepared SnakeBeta plan after parameter mutation")
        output = (
            torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
            if out is None else out
        )
        if (output.shape != hidden_states.shape or output.device != hidden_states.device
                or output.dtype != hidden_states.dtype or not output.is_contiguous()):
            raise ValueError("SnakeBeta output must be contiguous with matching shape/device/dtype")
        if self.strided:
            self.kernel(hidden_states, self.alpha, self.beta, output)
        else:
            contiguous = hidden_states.contiguous()
            self.kernel(
                contiguous.reshape(self.batch * self.channels, self.length),
                self.alpha, self.beta,
                output.reshape(self.batch * self.channels, self.length),
            )
        return output


def prepare_snake_beta(
    hidden_states: torch.Tensor,
    alpha_exp: torch.Tensor,
    beta_exp: torch.Tensor,
    *,
    strided: bool = False,
    tile_width: int = 512,
    num_warps: int = 4,
) -> PreparedSnakeBeta:
    return PreparedSnakeBeta(
        hidden_states, alpha_exp, beta_exp,
        strided=strided, tile_width=tile_width, num_warps=num_warps,
    )
