"""Numerical/native dispatch test for the independently built experiment library."""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import statistics
import time

from attention_policy import configure_backend, require_native_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lib", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", default="0,256,512,1024,2048,4096")
    parser.add_argument("--tokens", default="129,2048,27570")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use new evidence output")
    library = require_native_experiment(args.native_lib)
    import torch
    import infinicore
    from infinicore.ops.mha import mha
    from infinicore.tensor import from_torch
    assert torch.cuda.device_count() == 1 and torch.version.hip
    configure_backend("math_bf16")
    print("PHASE1_NATIVE_LIBRARY", library, flush=True)
    report = dict(library=library, torch=torch.__version__, hip=torch.version.hip,
                  scope="one GPU native MHA, not a DiT or full model test",
                  cases=[], complete=False)
    def emit(row):
        print("PHASE1_NATIVE_CASE", json.dumps(row), flush=True)
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    with torch.inference_mode():
        for length in map(int, args.tokens.split(",")):
            torch.manual_seed(42)
            shape = (2 if length == 129 else 1, length, 7, 128)
            q, k, v = [torch.randn(shape, dtype=torch.bfloat16, device="cuda")
                       for _ in range(3)]
            output = torch.empty_like(q)
            qi, ki, vi, oi = map(from_torch, (q, k, v, output))
            def invoke(chunk):
                os.environ["H3_EXPERIMENT_MHA_QUERY_CHUNK"] = str(chunk)
                return mha(qi, ki, vi, scale=1/math.sqrt(128), out=oi)
            invoke(0)
            torch.cuda.synchronize()
            reference = output.clone()
            for chunk in map(int, args.chunks.split(",")):
                invoke(chunk)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                starting = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                samples = []
                for _ in range(5):
                    begin = time.perf_counter()
                    invoke(chunk)
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter()-begin)
                peak = torch.cuda.max_memory_allocated()
                delta = output.float() - reference.float()
                row = dict(tokens=length, batch=shape[0], query_chunk=chunk,
                           median_seconds=statistics.median(samples), seconds=samples,
                           extra_peak_bytes=peak-starting,
                           max_abs=float(delta.abs().max()),
                           relative_l2=float(delta.norm()/reference.float().norm()),
                           bitwise_equal=bool(torch.equal(output, reference)),
                           finite=bool(torch.isfinite(output).all()))
                emit(row)
                assert row["finite"] and row["max_abs"] <= 0.05 and row["relative_l2"] <= 0.02
                del delta
            del q,k,v,qi,ki,vi,oi,output,reference
            torch.cuda.empty_cache()
        # Chunk scheduling must reject unsupported semantics rather than alter them.
        q = torch.randn(1,129,7,128,dtype=torch.bfloat16,device="cuda")
        qi = from_torch(q)
        os.environ["H3_EXPERIMENT_MHA_QUERY_CHUNK"]="64"
        try:
            mha(qi,qi,qi,scale=1/math.sqrt(128),is_causal=True)
        except Exception as error:
            assert "noncausal" in str(error), str(error)
            report["causal_rejected"] = True
        else:
            raise AssertionError("Causal input was not rejected")
        os.environ["H3_EXPERIMENT_MHA_QUERY_CHUNK"]="invalid"
        try:
            mha(qi,qi,qi)
        except Exception as error:
            assert "nonnegative integer" in str(error), str(error)
            report["invalid_chunk_rejected"] = True
        else:
            raise AssertionError("Invalid chunk was not rejected")
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print("PHASE1_NATIVE_END", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
