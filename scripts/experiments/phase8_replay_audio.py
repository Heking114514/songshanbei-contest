"""Replay an EXACT captured DiT input for layer diagnostics; no pipeline randomness."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from phase8_trace import TensorTrace, RunnerTrace, private_output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--input-snapshot", type=Path, required=True)
    ap.add_argument("--input-sha256", required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--dump-budget-mib", type=int, default=256)
    args = ap.parse_args()
    if hashlib.sha256(args.input_snapshot.read_bytes()).hexdigest() != args.input_sha256:
        ap.error("Frozen input SHA mismatch")
    output = private_output(args.workspace_root, args.output)
    import torch
    import sys
    source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source/"scripts"))
    from h3_preflight import bf16_math
    bf16_math()
    from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerInputs, MiniMaxH3TransformerRunner
    devices = [int(d) for d in args.gpus.split(",")]
    if not devices or len(set(devices)) != len(devices):
        ap.error("Select distinct GPUs")
    writer = TensorTrace(output, args.dump_budget_mib)
    tracer = RunnerTrace(writer, {0}, ())
    restore = tracer.install(MiniMaxH3TransformerRunner, devices)
    record = dict(complete=False, diagnostic_only=True, input_sha256=args.input_sha256,
                  plugin=os.environ.get("H3_FLASH_LIBRARY"), flags={k:v for k,v in os.environ.items()
                  if k.startswith("H3_EXPERIMENT")})
    try:
        captured = torch.load(args.input_snapshot, map_location="cpu", weights_only=True)
        inputs = MiniMaxH3TransformerInputs(**{k:v.to(f"cuda:{devices[0]}") for k,v in captured.items()})
        runner = MiniMaxH3TransformerRunner(args.model_path, device="nvidia", tp_device_ids=devices)
        with torch.inference_mode():
            runner.begin_generation()
            runner(inputs)
        record["complete"] = True
    except BaseException as exc:
        record["error"] = repr(exc)
        raise
    finally:
        restore(); writer.close(record["complete"])
        (output/"replay.json").write_bytes((json.dumps(record, indent=2)+"\n").encode())


if __name__ == "__main__":
    main()
