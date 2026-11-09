"""Explicit experimental SDPA policy and bounded-query scheduling.

This is a scheduling probe, not a new NineToothed kernel. The native integration
lives in the copied InfiniCore tree and must be separately built and checked.
"""
from __future__ import annotations

import ctypes
from pathlib import Path


def configure_backend(mode: str) -> dict:
    import torch
    if mode not in ("math_bf16", "flash"):
        raise ValueError("Unknown attention backend: " + mode)
    torch.backends.cuda.enable_flash_sdp(mode == "flash")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(mode == "math_bf16")
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(mode == "math_bf16")
    return {"mode": mode, "flash": torch.backends.cuda.flash_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "cudnn": (torch.backends.cuda.cudnn_sdp_enabled()
                      if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else None),
            "bf16_math_reduction":
                torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()}


def query_chunk_attention(query, key, value, chunk: int = 0, *, scale=None):
    import torch
    import torch.nn.functional as functional
    if chunk < 0:
        raise ValueError("query chunk cannot be negative")
    if chunk == 0 or chunk >= query.shape[-2]:
        return functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, scale=scale)
    output = torch.empty_like(query)
    for start in range(0, query.shape[-2], chunk):
        end = min(start + chunk, query.shape[-2])
        output[..., start:end, :] = functional.scaled_dot_product_attention(
            query[..., start:end, :], key, value,
            dropout_p=0.0, is_causal=False, scale=scale)
    return output


def require_native_experiment(library: str | None) -> str:
    if not library:
        raise ValueError("--native-lib is required with --query-chunk > 0")
    path = Path(library).resolve(strict=True)
    handle = ctypes.CDLL(str(path))
    try:
        marker = handle.infinicore_h3_query_chunk_experiment_v1
    except AttributeError as error:
        raise RuntimeError("Selected library lacks the query-chunk experiment; "
                           "do not use the stable AFS library") from error
    marker.argtypes = []
    marker.restype = ctypes.c_int
    if marker() != 1:
        raise RuntimeError("Unexpected native experiment ABI")
    return str(path)


def require_source_module(module, source_root: Path, component: str) -> None:
    expected = (source_root / component).resolve()
    actual = Path(module.__file__).resolve()
    if not actual.is_relative_to(expected):
        raise RuntimeError(f"{module.__name__} came from {actual}, expected {expected}")
