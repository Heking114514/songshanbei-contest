"""Replay one SHA-bound native DiT input; standalone timing or hipprof child."""
from __future__ import annotations
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import statistics
import sys
import time
from phase8_trace import private_output
from phase9_freeze import fingerprints
from phase10_profile import sha, stamp, device_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--input-snapshot", type=Path, required=True)
    ap.add_argument("--input-sha256", required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--dispatch-bundle", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        ap.error("At least one warmup and two measured repeats required")
    output = private_output(args.workspace_root, args.output)
    snapshot = private_output(args.workspace_root, args.input_snapshot)
    bundle = private_output(args.workspace_root, args.dispatch_bundle)
    if sha(snapshot) != args.input_sha256:
        ap.error("Frozen DiT input SHA mismatch")
    from phase9_attention import verify_bundle
    verify_bundle(bundle)
    output.mkdir(parents=True, exist_ok=False)
    devices = device_ids(args.gpus)
    os.environ.update(H3_EXPERIMENT_FLASH_GFX936="1",
                      H3_FLASH_LIBRARY=str(bundle/"libh3_flash_dispatch.so"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from h3_preflight import bf16_math
    bf16_math()
    import torch
    from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerInputs, MiniMaxH3TransformerRunner
    for device in devices:
        props = torch.cuda.get_device_properties(device)
        if not torch.version.hip or getattr(props, "gcnArchName", "").split(":")[0] != "gfx936":
            raise RuntimeError("Every selected GPU must be actual gfx936")
    report = dict(at=stamp(), complete=False, diagnostic_only=True, input_sha256=args.input_sha256,
                  devices=devices, samples=[], model_acceptance=False, afs_modified=False)
    from phase10_runtime import origins
    report["runtime_origins"]=origins(args.workspace_root,Path(os.environ["INFINI_ROOT"]))
    def sync():
        for device in devices: torch.cuda.synchronize(device)
    try:
        captured = torch.load(snapshot, map_location="cpu", weights_only=True)
        if set(captured) != {f.name for f in fields(MiniMaxH3TransformerInputs)}:
            raise ValueError("DiT input field schema changed")
        report["input_fingerprints"] = fingerprints(captured)
        inputs = MiniMaxH3TransformerInputs(**{k:v.to("cuda:"+str(devices[0])) for k,v in captured.items()})
        inputs.validate()
        runner = MiniMaxH3TransformerRunner(args.model_path, device="nvidia", tp_device_ids=devices)
        original = None
        with torch.inference_mode():
            runner.begin_generation()
            for index in range(args.warmup + args.repeats):
                sync()
                started = time.perf_counter()
                value = runner(inputs)
                sync()
                elapsed = time.perf_counter()-started
                # Hash complete outputs outside measured interval; this perturbs
                # the diagnostic process but never claims production latency.
                current = fingerprints(dict(video=value.sample, audio=value.audio_sample))
                if original is None: original = current
                if current != original:
                    raise RuntimeError("Frozen input replay output changed between repeats")
                if index >= args.warmup: report["samples"].append(elapsed)
                print("PHASE10_DIT_REPLAY", index, elapsed, flush=True)
        report.update(complete=True, output_fingerprints=original,
                      synchronized_median_seconds=statistics.median(report["samples"]))
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["finished_at"] = stamp()
        (output/"run.json").write_bytes((json.dumps(report, indent=2)+"\n").encode())


if __name__ == "__main__":
    main()
