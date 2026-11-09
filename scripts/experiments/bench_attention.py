"""Measure strict SDPA backends and query chunks on a single GPU.

The Python chunk schedule mirrors the native experiment; these measurements
do not prove that the native library or end-to-end pipeline is faster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from attention_policy import configure_backend, query_chunk_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", default="2048")
    parser.add_argument("--heads", type=int, default=7, help="56 / TP8")
    parser.add_argument("--layout", choices=("bshd", "bhsd"), default="bshd",
                        help="bshd models the native contiguous B,S,H,D input")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunks", default="0,256,512,1024")
    parser.add_argument("--backends", default="math_bf16,flash")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference-queries", type=int, default=32)
    parser.add_argument("--max-score-mib", type=int, default=512,
                        help="Budget for a conservative estimate of score temporaries")
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    parser.add_argument("--max-abs", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokens = [int(x) for x in args.tokens.split(",")]
    chunks = [int(x) for x in args.chunks.split(",")]
    modes = args.backends.split(",")
    if (min(tokens) < 1 or min(chunks) < 0 or args.heads < 1
            or args.head_dim < 1 or args.repeats < 1 or args.warmup < 0
            or args.reference_queries < 1 or args.max_score_mib < 1):
        parser.error("Invalid shape, repetition count or memory budget")
    if any(x not in ("math_bf16", "flash") for x in modes):
        parser.error("Backends must be math_bf16 or flash")
    if args.output.exists():
        parser.error("Use a new output file")
    import torch
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("A platform HIP GPU is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "single-GPU Python scheduling probe",
              "torch": torch.__version__, "hip": torch.version.hip,
              "gpu": torch.cuda.get_device_name(device), "args": vars(args).copy(),
              "cases": [], "complete": False}
    report["args"]["output"] = str(args.output)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    with torch.inference_mode():
        for length in tokens:
            generator = torch.Generator(device=device).manual_seed(42)
            shape = ((1, length, args.heads, args.head_dim) if args.layout == "bshd"
                     else (1, args.heads, length, args.head_dim))
            tensors = [torch.randn(shape, generator=generator, device=device,
                                   dtype=torch.bfloat16) for _ in range(3)]
            q, k, v = ([t.permute(0, 2, 1, 3) for t in tensors]
                       if args.layout == "bshd" else tensors)
            del tensors
            rows = torch.linspace(0, length - 1,
                                  min(length, args.reference_queries),
                                  device=device).round().long().unique()
            # Bounded FP32 reference retains ALL keys; only query rows are sampled.
            configure_backend("math_bf16")
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.index_select(-2, rows).float(), k.float(), v.float(),
                dropout_p=0.0, is_causal=False)
            for mode in modes:
                for chunk in chunks:
                    entry = {"tokens": length, "heads": args.heads,
                             "backend": mode, "query_chunk": chunk,
                             "query_strides": list(q.stride()), "layout": args.layout}
                    # Two score-sized FP32 buffers plus one BF16 buffer.
                    estimate = (min(chunk or length, length) * length
                                * args.heads * (4 + 4 + 2))
                    if mode == "math_bf16" and estimate > args.max_score_mib * 1024**2:
                        entry.update(status="skipped", reason="score memory budget",
                                     estimated_temporary_bytes=estimate)
                    else:
                        output = None
                        try:
                            entry["policy"] = configure_backend(mode)
                            for _ in range(args.warmup):
                                output = query_chunk_attention(q, k, v, chunk)
                                del output
                                output = None
                            torch.cuda.synchronize(device)
                            torch.cuda.reset_peak_memory_stats(device)
                            starting = torch.cuda.memory_allocated(device)
                            samples = []
                            for _ in range(args.repeats):
                                torch.cuda.synchronize(device)
                                started = time.perf_counter()
                                output = query_chunk_attention(q, k, v, chunk)
                                torch.cuda.synchronize(device)
                                samples.append(time.perf_counter() - started)
                                del output
                                output = None
                            peak = torch.cuda.max_memory_allocated(device)
                            output = query_chunk_attention(q, k, v, chunk)
                            actual = output.index_select(-2, rows).float()
                            difference = actual - reference
                            relative = float(difference.norm() /
                                             reference.norm().clamp_min(1e-12))
                            maximum = float(difference.abs().max())
                            finite = bool(torch.isfinite(output).all())
                            passed = (finite and relative <= args.max_relative_l2
                                      and maximum <= args.max_abs)
                            entry.update(status="measured", seconds=samples,
                                         median_seconds=statistics.median(samples),
                                         peak_allocated_bytes=peak,
                                         extra_peak_bytes=max(0, peak - starting),
                                         sampled_queries=len(rows), all_finite=finite,
                                         relative_l2=relative, max_abs=maximum,
                                         numerical_probe_passed=passed)
                            del actual, difference
                        except (RuntimeError, NotImplementedError) as error:
                            # Strict flash errors are recorded, never silently replaced.
                            entry.update(status="failed", error=str(error))
                        finally:
                            if output is not None:
                                del output
                            torch.cuda.empty_cache()
                    report["cases"].append(entry)
                    print(json.dumps(entry), flush=True)
                    save()
            del q, k, v, reference, rows
            torch.cuda.empty_cache()
    report["complete"] = True
    report["scope_note"] = ("Thresholds are experimental probes, not official "
                            "acceptance criteria or full-model parity.")
    save()


if __name__ == "__main__":
    main()
