"""Validate the new Hygon GroupNorm operator against PyTorch."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import infinicore
import infinicore.nn.functional as icf
from infinicore.tensor import from_torch


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def sync() -> None:
    infinicore.sync_device()
    torch.cuda.synchronize()


def bench(fn, warmup=5, iters=20) -> float:
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - start) / iters * 1000.0


def stats(actual, expected):
    actual_f = actual.float().flatten()
    expected_f = expected.float().flatten()
    diff = (actual_f - expected_f).abs()
    return {
        "cosine": F.cosine_similarity(actual_f, expected_f, dim=0).item(),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
    }


def run_case(shape, dtype, num_groups, eps) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20261031)
    x = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    channels = shape[1]
    weight = torch.randn(channels, device="cuda", dtype=dtype, generator=generator)
    bias = torch.randn(channels, device="cuda", dtype=dtype, generator=generator)
    out = torch.empty_like(x)

    ic_x = from_torch(x)
    ic_weight = from_torch(weight)
    ic_bias = from_torch(bias)
    ic_out = from_torch(out)

    def torch_fn():
        return F.group_norm(x, num_groups, weight, bias, eps)

    def ic_fn():
        icf.group_norm(
            ic_x,
            num_groups,
            ic_weight,
            ic_bias,
            eps,
            out=ic_out,
        )

    expected = torch_fn()
    ic_fn()
    sync()
    timing_torch = [bench(torch_fn) for _ in range(3)]
    timing_ic = [bench(ic_fn) for _ in range(3)]
    result = {
        "shape": list(shape),
        "dtype": str(dtype).split(".")[-1],
        "num_groups": num_groups,
        "torch_ms": statistics.median(timing_torch),
        "infinicore_ms": statistics.median(timing_ic),
        "correctness": stats(out, expected),
        "torch_runs_ms": timing_torch,
        "infinicore_runs_ms": timing_ic,
    }
    result["speed_ratio"] = result["torch_ms"] / result["infinicore_ms"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/group_norm_validation.json")
    args = parser.parse_args()
    infinicore.set_device(infinicore.device("cuda", 0))

    cases = [
        ((2, 64, 31, 33), 32, 1e-6),
        ((1, 128, 256, 256), 32, 1e-6),
    ]
    results = []
    for shape, groups, eps in cases:
        result = run_case(shape, torch.float32, groups, eps)
        results.append(result)
        print(
            f"group_norm shape={shape} groups={groups} "
            f"ic={result['infinicore_ms']:.4f}ms "
            f"torch={result['torch_ms']:.4f}ms "
            f"ratio={result['speed_ratio']:.3f} "
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
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"[write] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
