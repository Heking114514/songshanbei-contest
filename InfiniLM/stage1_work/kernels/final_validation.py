"""Batched correctness and performance validation for the H3 operator changes.

Run this as one DCU job after rebuilding InfiniCore and InfiniLM. It covers:
1. addcmul on the real strided AdaLN/gate layouts;
2. H3 Q/K RMSNorm row width;
3. H3 mRoPE dimensions and schedule;
4. fixed-length non-causal MHA correctness and long-sequence memory.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

import infinicore
from infinicore.tensor import from_torch
import infinicore.nn.functional as icf
from infinicore.ops.mha import mha as ic_mha
from infinicore.ops.mrope import mrope as ic_mrope


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
ROPE_FREQ_DIM = 16


def sync() -> None:
    infinicore.sync_device()
    torch.cuda.synchronize()


def bench(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return [(time.perf_counter() - start) / iters * 1000.0]


def stable_bench(fn, rounds: int = 3, warmup: int = 3, iters: int = 10) -> dict:
    runs = [bench(fn, warmup, iters)[0] for _ in range(rounds)]
    return {"median_ms": statistics.median(runs), "runs_ms": runs}


def stats(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_f = actual.float().flatten()
    expected_f = expected.float().flatten()
    diff = (actual_f - expected_f).abs()
    return {
        "cosine": F.cosine_similarity(actual_f, expected_f, dim=0).item(),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
    }


def attention_reference(q, k, v, scale, is_causal=False):
    qf = q.transpose(1, 2).float()
    kf = k.transpose(1, 2).float()
    vf = v.transpose(1, 2).float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    if is_causal:
        seq = scores.shape[-1]
        mask = torch.ones(
            seq, seq, dtype=torch.bool, device=scores.device
        ).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vf).transpose(1, 2).to(q.dtype)


def validate_adaln(seq: int, dtype: torch.dtype) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20261031)
    modulation = torch.randn(
        1, seq, 6 * HIDDEN, device="cuda", dtype=dtype, generator=generator
    )
    x = torch.randn(seq, HIDDEN, device="cuda", dtype=dtype, generator=generator)
    attention = torch.randn(
        seq, HIDDEN, device="cuda", dtype=dtype, generator=generator
    )

    scale = modulation.narrow(2, HIDDEN, HIDDEN).squeeze(0)
    shift = modulation.narrow(2, 2 * HIDDEN, HIDDEN).squeeze(0)
    gate_msa = modulation.narrow(2, 3 * HIDDEN, HIDDEN).squeeze(0)
    gate_mlp = modulation.narrow(2, 4 * HIDDEN, HIDDEN).squeeze(0)
    mlp = torch.randn(seq, HIDDEN, device="cuda", dtype=dtype, generator=generator)

    base_out = torch.empty_like(x)
    candidate_out = torch.empty_like(x)
    base_tmp1 = torch.empty_like(x)
    base_tmp2 = torch.empty_like(x)
    candidate_tmp = torch.empty_like(x)
    residual_base = torch.empty_like(x)
    residual_candidate_out = torch.empty_like(x)
    residual_tmp = torch.empty_like(x)

    ic_x = from_torch(x)
    ic_modulation = from_torch(modulation)
    ic_scale = ic_modulation.narrow(2, HIDDEN, HIDDEN).squeeze(0)
    ic_shift = ic_modulation.narrow(2, 2 * HIDDEN, HIDDEN).squeeze(0)
    ic_base_out = from_torch(base_out)
    ic_candidate_out = from_torch(candidate_out)
    ic_base_tmp1 = from_torch(base_tmp1)
    ic_base_tmp2 = from_torch(base_tmp2)
    ic_candidate_tmp = from_torch(candidate_tmp)

    def baseline():
        infinicore.mul(ic_x, ic_scale, out=ic_base_tmp1)
        infinicore.add(ic_x, ic_base_tmp1, out=ic_base_tmp2)
        infinicore.add(ic_base_tmp2, ic_shift, out=ic_base_out)

    def candidate():
        infinicore.addcmul(ic_x, ic_x, ic_scale, 1.0, out=ic_candidate_tmp)
        infinicore.add(ic_candidate_tmp, ic_shift, out=ic_candidate_out)

    baseline()
    candidate()
    sync()
    baseline_bench = stable_bench(baseline)
    candidate_bench = stable_bench(candidate)
    reference = (x.float() + x.float() * scale.float() + shift.float()).to(dtype)

    ic_gate = ic_modulation.narrow(2, 3 * HIDDEN, HIDDEN).squeeze(0)
    ic_attention = from_torch(attention)
    ic_residual_base = from_torch(residual_base)
    ic_residual_candidate = from_torch(residual_candidate_out)
    ic_residual_tmp = from_torch(residual_tmp)

    def residual_baseline():
        infinicore.mul(ic_gate, ic_attention, out=ic_residual_tmp)
        infinicore.add(ic_x, ic_residual_tmp, out=ic_residual_base)

    def residual_candidate():
        infinicore.addcmul(
            ic_x, ic_gate, ic_attention, 1.0, out=ic_residual_candidate
        )

    residual_baseline()
    residual_candidate()
    sync()
    residual_baseline_bench = stable_bench(residual_baseline)
    residual_candidate_bench = stable_bench(residual_candidate)
    residual_reference = (
        x.float() + gate_msa.float() * attention.float()
    ).to(dtype)

    return {
        "case": "adaln_real_stride",
        "seq": seq,
        "dtype": str(dtype).split(".")[-1],
        "baseline_ms": baseline_bench["median_ms"],
        "candidate_ms": candidate_bench["median_ms"],
        "speedup": baseline_bench["median_ms"] / candidate_bench["median_ms"],
        "candidate_correctness": stats(candidate_out, reference),
        "candidate_vs_baseline": stats(candidate_out, base_out),
        "residual_baseline_ms": residual_baseline_bench["median_ms"],
        "residual_candidate_ms": residual_candidate_bench["median_ms"],
        "residual_speedup": (
            residual_baseline_bench["median_ms"]
            / residual_candidate_bench["median_ms"]
        ),
        "residual_correctness": stats(
            residual_candidate_out, residual_reference
        ),
        "residual_vs_baseline": stats(
            residual_candidate_out, residual_base
        ),
    }


def validate_rms_norm(seq: int, dtype: torch.dtype) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20261032)
    x = torch.randn(
        seq, HEADS, HEAD_DIM, device="cuda", dtype=dtype, generator=generator
    )
    w = torch.randn(HEAD_DIM, device="cuda", dtype=dtype, generator=generator)
    actual = torch.empty_like(x)
    ic_x = from_torch(x)
    ic_w = from_torch(w)
    ic_actual = from_torch(actual)

    def native():
        icf.rms_norm(ic_x, [HEAD_DIM], ic_w, 1e-5, out=ic_actual)

    native()
    sync()
    timing = stable_bench(native, warmup=5, iters=30)
    expected = F.rms_norm(x.float(), [HEAD_DIM], w.float(), 1e-5).to(dtype)
    return {
        "case": "rms_norm_head_dim",
        "shape": [seq, HEADS, HEAD_DIM],
        "dtype": str(dtype).split(".")[-1],
        "native_ms": timing["median_ms"],
        "runs_ms": timing["runs_ms"],
        "correctness": stats(actual, expected),
    }


def build_mrope_reference(q, k, cos, sin, positions):
    half = 3 * ROPE_FREQ_DIM
    rotary_dim = 2 * half
    q_out = q.clone()
    k_out = k.clone()

    def rotate(tensor):
        heads = tensor.reshape(tensor.shape[0], HEADS, HEAD_DIM)
        prefix = heads[..., :rotary_dim].reshape(
            tensor.shape[0], HEADS, 2, half
        )
        x0, x1 = prefix.unbind(dim=2)
        axis_ids = torch.arange(
            half, device=tensor.device, dtype=torch.long
        ) // ROPE_FREQ_DIM
        rows = torch.arange(tensor.shape[0], device=tensor.device)
        selected_positions = positions.long()[rows[:, None], axis_ids[None, :]]
        dims = torch.arange(half, device=tensor.device)
        c = cos[selected_positions, dims[None, :]].float()[:, None, :]
        s = sin[selected_positions, dims[None, :]].float()[:, None, :]
        rotated = torch.stack((x0.float() * c - x1.float() * s,
                               x1.float() * c + x0.float() * s), dim=2)
        rotated = rotated.reshape(tensor.shape[0], HEADS, rotary_dim)
        return torch.cat(
            (rotated.to(tensor.dtype), heads[..., rotary_dim:]),
            dim=-1,
        ).reshape(tensor.shape[0], HEADS * HEAD_DIM)

    return rotate(q_out), rotate(k_out)


def validate_mrope(seq: int, dtype: torch.dtype) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20261033)
    q = torch.randn(seq, HEADS * HEAD_DIM, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(seq, HEADS * HEAD_DIM, device="cuda", dtype=dtype, generator=generator)
    positions = torch.stack(
        (
            torch.arange(seq, device="cuda") % 1024,
            (torch.arange(seq, device="cuda") * 3) % 1024,
            (torch.arange(seq, device="cuda") * 7) % 1024,
        ),
        dim=-1,
    ).to(torch.int64)
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(
                0, 2 * ROPE_FREQ_DIM, 2, dtype=torch.float32, device="cuda"
            )
            / (2 * ROPE_FREQ_DIM)
        )
    )
    angles = torch.stack(
        [
            positions[:, axis, None].float() * inv_freq[None, :]
            for axis in range(3)
        ],
        dim=1,
    ).reshape(seq, 3 * ROPE_FREQ_DIM)
    cos = angles.cos()
    sin = angles.sin()
    # Runtime cache layout is [cos, sin]; the public mRoPE operator takes them
    # as separate position-indexed tables.
    table = torch.cat((cos, sin), dim=-1).to(dtype)
    cos_table = table[:, : 3 * ROPE_FREQ_DIM].contiguous()
    sin_table = table[:, 3 * ROPE_FREQ_DIM :].contiguous()

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    ic_q = from_torch(q)
    ic_k = from_torch(k)
    ic_cos = from_torch(cos_table)
    ic_sin = from_torch(sin_table)
    ic_positions = from_torch(positions.transpose(0, 1).contiguous())
    ic_q_out = from_torch(q_out)
    ic_k_out = from_torch(k_out)

    def native():
        ic_mrope(
            ic_q,
            ic_k,
            ic_cos,
            ic_sin,
            ic_positions,
            HEAD_DIM,
            2 * 3 * ROPE_FREQ_DIM,
            ROPE_FREQ_DIM,
            ROPE_FREQ_DIM,
            ROPE_FREQ_DIM,
            False,
            out=(ic_q_out, ic_k_out),
        )

    native()
    sync()
    timing = stable_bench(native, warmup=3, iters=10)
    expected_q, expected_k = build_mrope_reference(
        q, k, cos_table, sin_table, positions
    )
    return {
        "case": "mrope_h3_shape",
        "seq": seq,
        "dtype": str(dtype).split(".")[-1],
        "native_ms": timing["median_ms"],
        "runs_ms": timing["runs_ms"],
        "q_correctness": stats(q_out, expected_q),
        "k_correctness": stats(k_out, expected_k),
    }


def validate_attention(seq: int, dtype: torch.dtype) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20261034)
    q = torch.randn(
        1, seq, HEADS, HEAD_DIM, device="cuda", dtype=dtype, generator=generator
    )
    k = torch.randn(
        1, seq, HEADS, HEAD_DIM, device="cuda", dtype=dtype, generator=generator
    )
    v = torch.randn(
        1, seq, HEADS, HEAD_DIM, device="cuda", dtype=dtype, generator=generator
    )
    out = torch.empty_like(q)
    ic_q, ic_k, ic_v, ic_out = map(from_torch, (q, k, v, out))
    scale = HEAD_DIM**-0.5

    def native():
        ic_mha(ic_q, ic_k, ic_v, None, scale, False, out=ic_out)

    torch.cuda.reset_peak_memory_stats()
    native()
    sync()
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    timing = stable_bench(native, rounds=2, warmup=1, iters=3 if seq >= 8192 else 10)
    correctness = (
        stats(out, attention_reference(q, k, v, scale, False))
        if seq <= 1024
        else {"skipped": "long-sequence case measures memory/scaling only"}
    )
    backend_info = {}
    backend_info["implementation"] = (
        "explicit QK^T-scale-softmax-PV Hygon implementation"
    )
    return {
        "case": "mha_non_causal",
        "seq": seq,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "dtype": str(dtype).split(".")[-1],
        "native_ms": timing["median_ms"],
        "runs_ms": timing["runs_ms"],
        "peak_allocated_gib": peak_gib,
        "correctness": correctness,
        "backend": backend_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="adaln,rms,mrope,mha")
    parser.add_argument("--out", default="results/final_validation.json")
    args = parser.parse_args()
    groups = {value.strip() for value in args.groups.split(",") if value.strip()}

    infinicore.set_device(infinicore.device("cuda", 0))
    print(
        f"host={platform.node()} torch={torch.__version__} hip={torch.version.hip} "
        f"device={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    results = []
    errors = []
    dtype = torch.bfloat16
    output = Path(args.out)

    def save() -> None:
        payload = {
            "hostname": platform.node(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "results": results,
            "errors": errors,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))

    def run_case(label, fn, *case_args) -> None:
        try:
            result = fn(*case_args)
            results.append(result)
            save()
            print(f"{label} ok={result}", flush=True)
        except Exception as exc:  # noqa: BLE001
            error = {
                "case": label,
                "error": str(exc),
                "trace": traceback.format_exc()[-1200:],
            }
            errors.append(error)
            save()
            print(f"{label} FAILED: {exc}", flush=True)

    if "adaln" in groups:
        for seq in (8192, 32768):
            run_case(f"adaln seq={seq}", validate_adaln, seq, dtype)
    if "rms" in groups:
        for seq in (4096, 16384):
            run_case(f"rms_norm seq={seq}", validate_rms_norm, seq, dtype)
    if "mrope" in groups:
        for seq in (4096, 16384, 39216):
            run_case(f"mrope seq={seq}", validate_mrope, seq, dtype)
    if "mha" in groups:
        # Correctness first at a small exact-H3 shape, then memory scaling.
        run_case("mha seq=1024", validate_attention, 1024, dtype)
        for seq in (4096, 8192, 39216):
            run_case(f"mha seq={seq}", validate_attention, seq, dtype)
            torch.cuda.empty_cache()

    save()
    print(f"[write] {output}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
