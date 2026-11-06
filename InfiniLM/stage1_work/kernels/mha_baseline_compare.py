"""Compare Hygon SDPA MHA with the previous explicit dense-attention path."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

import infinicore
from infinicore.ops.mha import mha as ic_mha
from infinicore.tensor import from_torch

HEADS = 56
HEAD_DIM = 128


def sync():
    infinicore.sync_device()
    torch.cuda.synchronize()


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - start) / iters * 1000.0


def compare(actual, expected):
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


def run_case(seq: int):
    generator = torch.Generator(device="cuda").manual_seed(20261031)
    q = torch.randn(
        1, seq, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output = torch.empty_like(q)
    ic_q, ic_k, ic_v, ic_out = map(from_torch, (q, k, v, output))
    scale = HEAD_DIM**-0.5

    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    def dense():
        scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * scale
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, v_t).transpose(1, 2)

    def sdpa():
        ic_mha(ic_q, ic_k, ic_v, None, scale, False, out=ic_out)

    expected = dense()
    sdpa()
    dense_runs = [bench(dense, 1, 3) for _ in range(2)]
    sdpa_runs = [bench(sdpa, 2, 5) for _ in range(2)]
    dense_ms = statistics.median(dense_runs)
    sdpa_ms = statistics.median(sdpa_runs)
    return {
        "seq": seq,
        "dense_ms": dense_ms,
        "sdpa_ms": sdpa_ms,
        "speedup": dense_ms / sdpa_ms,
        "dense_runs_ms": dense_runs,
        "sdpa_runs_ms": sdpa_runs,
        "correctness": compare(output, expected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", default="1024,4096")
    parser.add_argument("--out", default="results/mha_baseline_compare.json")
    args = parser.parse_args()
    infinicore.set_device(infinicore.device("cuda", 0))

    results = []
    for seq in [int(value) for value in args.seqs.split(",")]:
        result = run_case(seq)
        results.append(result)
        print(
            f"MHA seq={seq} dense={result['dense_ms']:.4f}ms "
            f"sdpa={result['sdpa_ms']:.4f}ms speedup={result['speedup']:.3f}x "
            f"cos={result['correctness']['cosine']:.8f} "
            f"max_abs={result['correctness']['max_abs']:.3e}",
            flush=True,
        )
        torch.cuda.empty_cache()

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
