"""Benchmark low-risk operator substitutions for the native H3 path.

The benchmark uses only public InfiniCore operators and preallocated outputs.
It compares the exact expressions currently used by
``minimax_h3_transformer.cpp`` with mathematically equivalent ``addcmul`` forms.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

import infinicore
from infinicore.tensor import from_torch


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def sync() -> None:
    infinicore.sync_device()
    torch.cuda.synchronize()


def bench(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - start) / iters * 1000.0


def stable_bench(fn, rounds: int = 3) -> list[float]:
    return [bench(fn) for _ in range(rounds)]


def stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float().flatten()
    expected_f = expected.float().flatten()
    diff = (actual_f - expected_f).abs()
    return {
        "cosine": torch.nn.functional.cosine_similarity(
            actual_f, expected_f, dim=0
        ).item(),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
    }


def torch_addcmul_ref(base, lhs, rhs):
    return (base.float() + lhs.float() * rhs.float()).to(base.dtype)


def run_adaln(seq: int, dtype: torch.dtype) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20261031)
    x = torch.randn(seq, 5376, device="cuda", dtype=dtype, generator=generator)
    scale = torch.randn(
        seq, 5376, device="cuda", dtype=dtype, generator=generator
    ) * 0.1
    shift = torch.randn(
        seq, 5376, device="cuda", dtype=dtype, generator=generator
    ) * 0.1

    baseline_out = torch.empty_like(x)
    candidate_out = torch.empty_like(x)
    baseline_tmp1 = torch.empty_like(x)
    baseline_tmp2 = torch.empty_like(x)
    candidate_tmp = torch.empty_like(x)

    ic_x = from_torch(x)
    ic_scale = from_torch(scale)
    ic_shift = from_torch(shift)
    ic_baseline_out = from_torch(baseline_out)
    ic_candidate_out = from_torch(candidate_out)
    ic_baseline_tmp1 = from_torch(baseline_tmp1)
    ic_baseline_tmp2 = from_torch(baseline_tmp2)
    ic_candidate_tmp = from_torch(candidate_tmp)

    def baseline():
        infinicore.mul(ic_x, ic_scale, out=ic_baseline_tmp1)
        infinicore.add(ic_x, ic_baseline_tmp1, out=ic_baseline_tmp2)
        infinicore.add(ic_baseline_tmp2, ic_shift, out=ic_baseline_out)

    def candidate():
        infinicore.addcmul(
            ic_x, ic_x, ic_scale, 1.0, out=ic_candidate_tmp
        )
        infinicore.add(ic_candidate_tmp, ic_shift, out=ic_candidate_out)

    baseline()
    candidate()
    sync()

    baseline_runs = stable_bench(baseline)
    candidate_runs = stable_bench(candidate)
    ref = (
        x.float() + x.float() * scale.float() + shift.float()
    ).to(dtype)
    return {
        "name": "adaln",
        "seq": seq,
        "dtype": str(dtype).split(".")[-1],
        "baseline_ms": statistics.median(baseline_runs),
        "candidate_ms": statistics.median(candidate_runs),
        "baseline_runs": baseline_runs,
        "candidate_runs": candidate_runs,
        "baseline_correctness": stats(baseline_out, ref),
        "candidate_correctness": stats(candidate_out, ref),
        "candidate_vs_baseline": stats(candidate_out, baseline_out),
    }


def run_gated_residual(seq: int, dtype: torch.dtype) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20261032)
    residual = torch.randn(
        seq, 5376, device="cuda", dtype=dtype, generator=generator
    )
    gate = torch.randn(
        seq, 5376, device="cuda", dtype=dtype, generator=generator
    ) * 0.1
    branch = torch.randn(
        seq, 5376, device="cuda", dtype=dtype, generator=generator
    )

    baseline_out = torch.empty_like(residual)
    candidate_out = torch.empty_like(residual)
    baseline_tmp = torch.empty_like(residual)

    ic_residual = from_torch(residual)
    ic_gate = from_torch(gate)
    ic_branch = from_torch(branch)
    ic_baseline_out = from_torch(baseline_out)
    ic_candidate_out = from_torch(candidate_out)
    ic_baseline_tmp = from_torch(baseline_tmp)

    def baseline():
        infinicore.mul(ic_gate, ic_branch, out=ic_baseline_tmp)
        infinicore.add(ic_residual, ic_baseline_tmp, out=ic_baseline_out)

    def candidate():
        infinicore.addcmul(
            ic_residual, ic_gate, ic_branch, 1.0, out=ic_candidate_out
        )

    baseline()
    candidate()
    sync()

    baseline_runs = stable_bench(baseline)
    candidate_runs = stable_bench(candidate)
    ref = torch_addcmul_ref(residual, gate, branch)
    return {
        "name": "gated_residual",
        "seq": seq,
        "dtype": str(dtype).split(".")[-1],
        "baseline_ms": statistics.median(baseline_runs),
        "candidate_ms": statistics.median(candidate_runs),
        "baseline_runs": baseline_runs,
        "candidate_runs": candidate_runs,
        "baseline_correctness": stats(baseline_out, ref),
        "candidate_correctness": stats(candidate_out, ref),
        "candidate_vs_baseline": stats(candidate_out, baseline_out),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", default="8192,32768")
    parser.add_argument("--dtypes", default="bf16,fp16")
    parser.add_argument("--out", default="results/formal_candidates.json")
    args = parser.parse_args()

    infinicore.set_device(infinicore.device("cuda", 0))
    print(
        f"torch={torch.__version__} hip={torch.version.hip} "
        f"devices={torch.cuda.device_count()} "
        f"name={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    results = []
    for dtype_name in args.dtypes.split(","):
        dtype = DTYPES[dtype_name]
        for seq in (int(value) for value in args.seqs.split(",")):
            for runner in (run_adaln, run_gated_residual):
                result = runner(seq, dtype)
                result["speedup"] = (
                    result["baseline_ms"] / result["candidate_ms"]
                )
                results.append(result)
                print(
                    f"{result['name']:<15} seq={seq:<6} {dtype_name:<4} "
                    f"baseline={result['baseline_ms']:.4f}ms "
                    f"candidate={result['candidate_ms']:.4f}ms "
                    f"speedup={result['speedup']:.3f}x "
                    f"cos={result['candidate_vs_baseline']['cosine']:.8f}",
                    flush=True,
                )

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
