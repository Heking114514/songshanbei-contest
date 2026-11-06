"""Compile/load the cached AOT SnakeBeta artifact and verify it on DCU."""

from __future__ import annotations

import argparse
import json
import platform
import pickle
import statistics
import time
from pathlib import Path

import torch

from ninetoothed.compiler import load_built_artifact

def bench(fn, warmup=10, iters=50):
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
    parser.add_argument("--lengths", default="16384,331200")
    parser.add_argument("--out", default="results/aot_snake_beta_runtime.json")
    args = parser.parse_args()

    artifact_dir = Path.home() / "work/infinilm/kernels/aot_snake_beta"
    with (artifact_dir / "built.pkl").open("rb") as handle:
        built = pickle.load(handle)
    artifact_path = built.binary_path
    kernel = load_built_artifact(built)

    results = []
    for length in [int(value) for value in args.lengths.split(",")]:
        generator = torch.Generator(device="cuda").manual_seed(20261031)
        x = torch.randn(128, length, device="cuda", generator=generator)
        alpha_log = torch.randn(128, device="cuda", generator=generator)
        beta_log = torch.randn(128, device="cuda", generator=generator)
        alpha_exp = alpha_log.exp()
        beta_exp = beta_log.exp()
        actual = torch.empty_like(x)
        kernel(x, alpha_exp, beta_exp, actual)
        expected = (
            x
            + torch.sin(alpha_exp[:, None] * x).square()
            / (beta_exp[:, None] + 1e-9)
        )
        torch.cuda.synchronize()
        diff = (actual - expected).abs()
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item()
        row_cosine = torch.nn.functional.cosine_similarity(
            actual, expected, dim=1
        )
        bad_rows = (row_cosine < 0.999).nonzero().flatten()
        print(
            f"debug length={length} rows={actual.shape[0]} "
            f"bad_rows={bad_rows.numel()} first_bad={bad_rows[:8].tolist()}",
            flush=True,
        )
        print(
            f"debug row0 actual={actual[0, :8].tolist()} "
            f"expected={expected[0, :8].tolist()} "
            f"alpha={alpha_exp[0].item():.8f} beta={beta_exp[0].item():.8f}",
            flush=True,
        )

        def torch_fn():
            return (
                x
                + torch.sin(alpha_exp[:, None] * x).square()
                / (beta_exp[:, None] + 1e-9)
            )

        def aot_fn():
            kernel(x, alpha_exp, beta_exp, actual)

        torch_runs = [bench(torch_fn) for _ in range(3)]
        aot_runs = [bench(aot_fn) for _ in range(3)]
        result = {
            "length": length,
            "torch_ms": statistics.median(torch_runs),
            "aot_ms": statistics.median(aot_runs),
            "speedup": statistics.median(torch_runs)
            / statistics.median(aot_runs),
            "torch_runs_ms": torch_runs,
            "aot_runs_ms": aot_runs,
            "correctness": {
                "cosine": cosine,
                "max_abs": diff.max().item(),
                "mean_abs": diff.mean().item(),
            },
        }
        results.append(result)
        print(
            f"length={length} torch={result['torch_ms']:.4f}ms "
            f"aot={result['aot_ms']:.4f}ms speedup={result['speedup']:.3f}x "
            f"cos={cosine:.8f} max_abs={diff.max().item():.3e}",
            flush=True,
        )

    payload = {
        "hostname": platform.node(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "artifact": str(artifact_path),
        "results": results,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"[write] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
