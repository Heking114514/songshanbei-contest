"""Replay a captured real native DiT input for controlled timing and comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

from attention_policy import (configure_backend, require_native_experiment,
                              require_source_module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--backend", choices=("math_bf16", "flash"), default="math_bf16")
    parser.add_argument("--query-chunk", type=int, default=0)
    parser.add_argument("--native-lib")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-state", choices=("request", "steady"), default="request")
    parser.add_argument("--save-output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    devices = [int(x) for x in args.gpus.split(",")]
    if (not devices or len(set(devices)) != len(devices) or min(devices) < 0
            or args.query_chunk < 0 or args.warmup < 0 or args.repeats < 1):
        parser.error("Invalid devices, chunk or repetition count")
    for destination in [args.output, args.save_output]:
        if destination and destination.exists():
            parser.error("Use new report/output paths")
    if args.query_chunk:
        require_native_experiment(args.native_lib)
    os.environ["H3_EXPERIMENT_MHA_QUERY_CHUNK"] = str(args.query_chunk)
    policy = configure_backend(args.backend)
    import torch
    import infinilm
    import infinicore
    source = Path(__file__).resolve().parents[2]
    require_source_module(infinilm, source, "InfiniLM")
    require_source_module(infinicore, source, "InfiniCore")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("Platform HIP Torch is required")
    from infinilm.multimodal.minimax_h3 import (
        MiniMaxH3TransformerInputs, MiniMaxH3TransformerRunner)
    capture = torch.load(args.input, map_location="cpu", weights_only=True)
    if capture.get("diagnostic_only") is not True:
        raise RuntimeError("Input is not marked diagnostic_only")
    inputs = MiniMaxH3TransformerInputs(
        **{k: v.to(f"cuda:{devices[0]}") for k, v in capture["inputs"].items()})
    inputs.validate()
    runner = MiniMaxH3TransformerRunner(args.model_path, device="hygon",
                                       tp_device_ids=devices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"diagnostic_only": True, "complete": False, "args": {
              k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "attention_policy": policy, "torch": torch.__version__,
              "hip": torch.version.hip, "samples_seconds": []}

    def sync():
        for device in devices:
            torch.cuda.synchronize(device)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    with torch.inference_mode():
        runner.begin_generation()
        if args.cache_state == "steady":
            runner(inputs)
            sync()
        for iteration in range(args.warmup + args.repeats):
            if args.cache_state == "request":
                runner.begin_generation()
            sync()
            started = time.perf_counter()
            result = runner(inputs)
            sync()
            seconds = time.perf_counter() - started
            if iteration >= args.warmup:
                report["samples_seconds"].append(seconds)
            print("DIT_SAMPLE", iteration, seconds, "cache", args.cache_state, flush=True)
            save()
        outputs = {"video": result.sample.float().cpu(),
                   "audio": result.audio_sample.float().cpu()}
    if args.compare:
        reference = torch.load(args.compare, map_location="cpu", weights_only=True)
        report["differences"] = {}
        for name, actual in outputs.items():
            expected = reference[name]
            if actual.shape != expected.shape:
                raise RuntimeError("Output shape differs for " + name)
            difference = actual - expected
            report["differences"][name] = {
                "all_finite": bool(torch.isfinite(actual).all()),
                "bitwise_equal": bool(torch.equal(actual, expected)),
                "max_abs": float(difference.abs().max()),
                "relative_l2": float(difference.norm() / expected.norm().clamp_min(1e-12))}
    if args.save_output:
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        with args.save_output.open("xb") as destination:
            torch.save(outputs, destination)
    report.update(complete=True,
                  median_seconds=statistics.median(report["samples_seconds"]))
    save()
    print("DIT_BENCH_COMPLETE", report["median_seconds"], flush=True)


if __name__ == "__main__":
    main()
