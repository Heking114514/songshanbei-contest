"""Phase-1 SDPA experiment: identical BSHD output layout, all K/V retained."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

from attention_policy import configure_backend, query_chunk_attention


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", default="2048,8192,27570")
    parser.add_argument("--chunks", default="0,256,512,1024,2048,4096")
    parser.add_argument("--heads", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite evidence")
    import torch
    import torch.nn.functional as F
    assert torch.cuda.is_available() and torch.version.hip
    assert torch.cuda.device_count() == 1, "Single GPU required"
    torch.cuda.set_device(0)
    report = dict(scope="single-GPU attention; TP8 local 7 heads; BSHD I/O",
                  torch=torch.__version__, hip=torch.version.hip,
                  gpu=str(torch.cuda.get_device_properties(0)), cases=[], complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    def emit(row):
        report["cases"].append(row)
        print("PHASE1_CASE " + json.dumps(row), flush=True)
        save()
    # Strict flash must fail visibly if unavailable; never allow math fallback.
    configure_backend("flash")
    small = torch.randn(1, args.heads, 129, 128, device="cuda", dtype=torch.bfloat16)
    try:
        F.scaled_dot_product_attention(small, small, small)
        torch.cuda.synchronize()
        report["strict_flash_available"] = True
    except (RuntimeError, NotImplementedError) as error:
        report["strict_flash_available"] = False
        report["strict_flash_error"] = str(error)
        print("PHASE1_STRICT_FLASH_UNAVAILABLE", str(error), flush=True)
    del small
    save()

    def attention(q, k, v, chunk):
        # Native MHA copies the result back to BSHD storage in BOTH paths.
        # empty_like preserves the dense transposed strides of BSHD inputs.
        output = torch.empty_like(q)
        if chunk == 0 or chunk >= q.shape[-2]:
            output.copy_(F.scaled_dot_product_attention(q, k, v))
        else:
            for start in range(0, q.shape[-2], chunk):
                output[..., start:start+chunk, :].copy_(
                    F.scaled_dot_product_attention(q[..., start:start+chunk, :], k, v))
        return output

    with torch.inference_mode():
        for length in [int(x) for x in args.tokens.split(",")]:
            torch.manual_seed(42)
            q, k, v = [torch.randn(1, length, args.heads, 128,
                                  dtype=torch.bfloat16, device="cuda").permute(0, 2, 1, 3)
                       for _ in range(3)]
            rows = torch.linspace(0, length-1, min(length, 64), device="cuda").round().long().unique()
            configure_backend("math_bf16")
            # Full native-math output is the schedule reference, not FP32 oracle.
            baseline = attention(q, k, v, 0)
            torch.cuda.synchronize()
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
            fp32 = F.scaled_dot_product_attention(
                q.index_select(-2, rows).float(), k.float(), v.float())
            del_fp32 = False
            for mode in (["math_bf16", "flash"] if report["strict_flash_available"] else ["math_bf16"]):
                for chunk in [int(x) for x in args.chunks.split(",")]:
                    entry = dict(tokens=length, heads=args.heads, mode=mode, query_chunk=chunk)
                    output = None
                    try:
                        configure_backend(mode)
                        output = attention(q, k, v, chunk)
                        torch.cuda.synchronize()
                        del output
                        torch.cuda.empty_cache()
                        starting = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        samples = []
                        for _ in range(args.repeats):
                            t = time.perf_counter()
                            output = attention(q, k, v, chunk)
                            torch.cuda.synchronize()
                            samples.append(time.perf_counter()-t)
                            del output
                            output = None
                        peak = torch.cuda.max_memory_allocated()
                        output = attention(q, k, v, chunk)
                        delta = output.float() - baseline.float()
                        sampled = output.index_select(-2, rows).float() - fp32
                        finite = bool(torch.isfinite(output).all())
                        entry.update(status="measured", seconds=samples,
                                     median_seconds=statistics.median(samples),
                                     extra_peak_bytes=peak-starting,
                                     baseline_max_abs=float(delta.abs().max()),
                                     baseline_relative_l2=float(delta.norm()/baseline.float().norm()),
                                     baseline_bitwise_equal=bool(torch.equal(output, baseline)),
                                     fp32_sample_max_abs=float(sampled.abs().max()),
                                     fp32_sample_relative_l2=float(sampled.norm()/fp32.norm()),
                                     finite=finite)
                        del delta, sampled
                    except (RuntimeError, NotImplementedError) as error:
                        entry.update(status="failed", error=str(error))
                    finally:
                        if output is not None:
                            del output
                        torch.cuda.empty_cache()
                    emit(entry)
            del q, k, v, rows, baseline, fp32
            torch.cuda.empty_cache()
    report["complete"] = True
    save()
    print("PHASE1_PROBE_END", json.dumps(report), flush=True)


if __name__ == "__main__":
    run()
