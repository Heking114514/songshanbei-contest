"""Run the copied official CLI with an explicit experimental SDPA policy."""
from __future__ import annotations

import argparse
from dataclasses import fields
import importlib.util
import os
from pathlib import Path
import sys

from attention_policy import (configure_backend, require_native_experiment,
                              require_source_module)


class CaptureComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("math_bf16", "flash"), default="math_bf16")
    parser.add_argument("--query-chunk", type=int, default=0)
    parser.add_argument("--native-lib")
    parser.add_argument("--capture-dit-input", type=Path)
    parser.add_argument("--capture-only", action="store_true")
    args, official_args = parser.parse_known_args()
    if official_args[:1] == ["--"]:
        official_args = official_args[1:]
    if args.query_chunk < 0:
        parser.error("--query-chunk must be nonnegative")
    if args.capture_only and not args.capture_dit_input:
        parser.error("--capture-only requires --capture-dit-input")
    if args.capture_dit_input and args.capture_dit_input.exists():
        parser.error("Capture destination already exists")
    if "--legacy" in official_args:
        parser.error("This experimental entry is for the native path only")
    if not official_args:
        parser.error("Provide official CLI arguments after --")
    if args.query_chunk:
        require_native_experiment(args.native_lib)
    source = Path(__file__).resolve().parents[2]
    os.environ["H3_EXPERIMENT_MHA_QUERY_CHUNK"] = str(args.query_chunk)
    os.environ["H3_SOURCE_ROOT"] = str(source)
    policy = configure_backend(args.backend)
    import infinicore
    import infinilm
    require_source_module(infinicore, source, "InfiniCore")
    require_source_module(infinilm, source, "InfiniLM")
    path = source / "InfiniLM/examples/minimax_h3.py"
    spec = importlib.util.spec_from_file_location("h3_experiment_entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if args.capture_dit_input:
        original = module.MiniMaxH3TransformerRunner.__call__
        captured = False

        def capture(self, inputs, *call_args, **call_kwargs):
            nonlocal captured
            if not captured:
                import torch
                args.capture_dit_input.parent.mkdir(parents=True, exist_ok=True)
                payload = {"diagnostic_only": True, "source_root": str(source),
                           "attention_policy": policy,
                           "inputs": {f.name: getattr(inputs, f.name).detach().cpu().clone()
                                      for f in fields(inputs)}}
                # Exclusive create prevents replacing another capture.
                with args.capture_dit_input.open("xb") as destination:
                    torch.save(payload, destination)
                captured = True
                print("H3_EXPERIMENT_CAPTURE", args.capture_dit_input, flush=True)
                if args.capture_only:
                    raise CaptureComplete()
            return original(self, inputs, *call_args, **call_kwargs)
        module.MiniMaxH3TransformerRunner.__call__ = capture
    print("H3_EXPERIMENT_RUNTIME", {"source": str(source), "attention": policy,
                                    "native_query_chunk": args.query_chunk}, flush=True)
    sys.argv = [str(path), *official_args]
    try:
        module.main()
    except CaptureComplete:
        print("H3_CAPTURE_ONLY_COMPLETE: no inference result was generated", flush=True)


if __name__ == "__main__":
    main()
