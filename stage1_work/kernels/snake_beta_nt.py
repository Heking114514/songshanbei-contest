"""NineToothed fused SnakeBeta kernel for MiniMax-H3 audio VAE."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor, block_size

BLOCK_M = block_size(1, 1)
WIDTH = Symbol("WIDTH", constexpr=True)


def arrangement(x, alpha, beta, out, WIDTH=WIDTH):
    x_arranged = x.tile((BLOCK_M, WIDTH))
    return (
        x_arranged,
        alpha.tile((BLOCK_M,)).unsqueeze(1).expand((-1, x_arranged.shape[1])),
        beta.tile((BLOCK_M,)).unsqueeze(1).expand((-1, x_arranged.shape[1])),
        out.tile((BLOCK_M, WIDTH)),
    )


def application(x, alpha_exp, beta_exp, out):
    sine = math.sin(alpha_exp * x)
    out = x + sine * sine / (beta_exp + 1e-9)  # noqa: F841


def build(block_m: int = 1, num_warps: int | None = None):
    global BLOCK_M
    BLOCK_M = block_size(block_m, block_m)
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(1), Tensor(1), Tensor(2)),
        num_warps=num_warps,
    )


def reference(x, alpha_exp, beta_exp):
    alpha = alpha_exp[:, None]
    beta = beta_exp[:, None]
    return x + torch.sin(alpha * x).square() / (beta + 1e-9)


def bench(fn, warmup=10, iters=50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--block-m", type=int, default=1)
    parser.add_argument("--warps", type=int, default=None)
    parser.add_argument("--out", default="results/snake_beta_nt.json")
    args = parser.parse_args()

    generator = torch.Generator(device="cuda").manual_seed(20261031)
    x = torch.randn(
        args.batch,
        args.channels,
        args.length,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    alpha = torch.randn(
        args.channels, device="cuda", dtype=torch.float32, generator=generator
    )
    beta = torch.randn(
        args.channels, device="cuda", dtype=torch.float32, generator=generator
    )
    x_rows = x.reshape(args.batch * args.channels, args.length).contiguous()
    alpha_rows = alpha.exp().repeat(args.batch).contiguous()
    beta_rows = beta.exp().repeat(args.batch).contiguous()
    out = torch.empty_like(x_rows)

    kernel = build(args.block_m, args.warps)
    kernel(x_rows, alpha_rows, beta_rows, out, WIDTH=args.length)
    expected = reference(x_rows, alpha_rows, beta_rows)
    torch.cuda.synchronize()

    diff = (out - expected).abs()
    cosine = torch.nn.functional.cosine_similarity(
        out.flatten(), expected.flatten(), dim=0
    ).item()

    def torch_fn():
        return reference(x_rows, alpha_rows, beta_rows)

    def nt_fn():
        kernel(x_rows, alpha_rows, beta_rows, out, WIDTH=args.length)

    torch_runs = [bench(torch_fn) for _ in range(3)]
    nt_runs = [bench(nt_fn) for _ in range(3)]
    t_torch = statistics.median(torch_runs)
    t_nt = statistics.median(nt_runs)
    payload = {
        "hostname": platform.node(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "shape": [args.batch, args.channels, args.length],
        "block_m": args.block_m,
        "num_warps": args.warps,
        "torch_ms": t_torch,
        "ninetoothed_ms": t_nt,
        "speedup": t_torch / t_nt,
        "torch_runs_ms": torch_runs,
        "ninetoothed_runs_ms": nt_runs,
        "correctness": {
            "cosine": cosine,
            "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(),
        },
        "weight_preprocessing": "alpha_log.exp() and beta_log.exp() hoisted out of forward",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(
        f"shape={payload['shape']} torch={t_torch:.4f}ms nine={t_nt:.4f}ms "
        f"speedup={payload['speedup']:.3f}x cosine={cosine:.8f} "
        f"max_abs={payload['correctness']['max_abs']:.3e}",
        flush=True,
    )
    print(f"[write] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
