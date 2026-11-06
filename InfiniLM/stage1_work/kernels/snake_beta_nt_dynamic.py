"""Dynamic-length NineToothed SnakeBeta kernel for H3 audio VAE."""

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
from ninetoothed import Tensor, block_size


def make_kernel(block_m: int, tile_width: int, num_warps: int | None):
    rows = block_size(block_m, block_m)

    def arrangement(x, alpha, beta, out):
        x_arranged = x.tile((rows, tile_width))
        alpha_arranged = (
            alpha.tile((rows,)).unsqueeze(1).expand((-1, x_arranged.shape[1]))
        )
        beta_arranged = (
            beta.tile((rows,)).unsqueeze(1).expand((-1, x_arranged.shape[1]))
        )
        return (
            x_arranged,
            alpha_arranged,
            beta_arranged,
            out.tile((rows, tile_width)),
        )

    def application(x, alpha_exp, beta_exp, out):
        sine = math.sin(alpha_exp * x)
        out = x + sine * sine / (beta_exp + 1e-9)  # noqa: F841

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


def run_case(batch, channels, length, block_m, tile_width, warps):
    generator = torch.Generator(device="cuda").manual_seed(20261031 + length)
    x = torch.randn(
        batch, channels, length, device="cuda", dtype=torch.float32, generator=generator
    )
    alpha_log = torch.randn(
        channels, device="cuda", dtype=torch.float32, generator=generator
    )
    beta_log = torch.randn(
        channels, device="cuda", dtype=torch.float32, generator=generator
    )
    x_rows = x.reshape(batch * channels, length).contiguous()
    alpha_rows = alpha_log.exp().repeat(batch).contiguous()
    beta_rows = beta_log.exp().repeat(batch).contiguous()
    out = torch.empty_like(x_rows)

    kernel = make_kernel(block_m, tile_width, warps)
    kernel(x_rows, alpha_rows, beta_rows, out)
    expected = reference(x_rows, alpha_rows, beta_rows)
    torch.cuda.synchronize()
    diff = (out - expected).abs()
    cosine = torch.nn.functional.cosine_similarity(
        out.flatten(), expected.flatten(), dim=0
    ).item()

    def torch_fn():
        return reference(x_rows, alpha_rows, beta_rows)

    def nt_fn():
        kernel(x_rows, alpha_rows, beta_rows, out)

    torch_runs = [bench(torch_fn) for _ in range(3)]
    nt_runs = [bench(nt_fn) for _ in range(3)]
    torch_ms = statistics.median(torch_runs)
    nt_ms = statistics.median(nt_runs)
    return {
        "shape": [batch, channels, length],
        "block_m": block_m,
        "tile_width": tile_width,
        "num_warps": warps,
        "torch_ms": torch_ms,
        "ninetoothed_ms": nt_ms,
        "speedup": torch_ms / nt_ms,
        "torch_runs_ms": torch_runs,
        "ninetoothed_runs_ms": nt_runs,
        "correctness": {
            "cosine": cosine,
            "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="207,2048,16384,331200")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--block-m", type=int, default=1)
    parser.add_argument("--tile-width", type=int, default=256)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--out", default="results/snake_beta_nt_dynamic.json")
    args = parser.parse_args()

    results = []
    for length in [int(value) for value in args.lengths.split(",")]:
        result = run_case(
            1,
            args.channels,
            length,
            args.block_m,
            args.tile_width,
            args.warps,
        )
        results.append(result)
        print(
            f"length={length} torch={result['torch_ms']:.4f}ms "
            f"nine={result['ninetoothed_ms']:.4f}ms "
            f"speedup={result['speedup']:.3f}x "
            f"cos={result['correctness']['cosine']:.8f} "
            f"max_abs={result['correctness']['max_abs']:.3e}",
            flush=True,
        )

    payload = {
        "hostname": platform.node(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "results": results,
        "weight_preprocessing": "alpha_log.exp() and beta_log.exp() hoisted out of forward",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"[write] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
