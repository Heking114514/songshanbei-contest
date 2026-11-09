"""Future node entry: unchanged generation, optional audio diagnostics, fresh output directory."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
from phase8_trace import TensorTrace, RunnerTrace, private_output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--native-steps", default="0")
    ap.add_argument("--capture-steps", default="0")
    ap.add_argument("--dump-budget-mib", type=int, default=256)
    ap.add_argument("generation_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    argv = args.generation_args
    if argv[:1] == ["--"]: argv = argv[1:]
    if not argv or "--legacy" in argv or "--output" in argv:
        ap.error("Pass native generation arguments after --; output is managed by this wrapper")
    output = private_output(args.workspace_root, args.output)
    output.mkdir(exist_ok=False)
    import torch
    source = Path(__file__).resolve().parents[2]
    entry_path = source/"InfiniLM/examples/minimax_h3.py"
    sys.path.insert(0, str(source/"scripts"))
    from h3_preflight import bf16_math
    bf16_math()
    spec = importlib.util.spec_from_file_location("phase8_h3_entry", entry_path)
    entry = importlib.util.module_from_spec(spec); sys.modules[spec.name] = entry; spec.loader.exec_module(entry)
    gp = argparse.ArgumentParser(add_help=False); gp.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    known, _ = gp.parse_known_args(argv)
    devices = entry.parse_gpu_list(known.gpus)
    writer = TensorTrace(output/"audio-trace", args.dump_budget_mib) if args.trace else None
    native_steps = {int(s) for s in args.native_steps.split(",") if s.strip()}
    capture_steps = {int(s) for s in args.capture_steps.split(",") if s.strip()}
    if any(s < 0 for s in native_steps | capture_steps):
        ap.error("Trace steps are zero based and nonnegative")
    tracer = RunnerTrace(writer, native_steps if writer else (), capture_steps if writer else ())
    restore_runner = tracer.install(entry.MiniMaxH3TransformerRunner, devices)
    original_install = entry.install_component_profilers
    original_report = entry.TimingReport.print_report
    handles, restores = [], []
    run = dict(complete=False, diagnostic_only=args.trace, entry_sha256=hashlib.sha256(entry_path.read_bytes()).hexdigest(),
               arguments=argv, torch=torch.__version__, hip=torch.version.hip, devices=devices,
               flags={k: v for k, v in os.environ.items() if k.startswith("H3_EXPERIMENT")},
               media_quality_accepted=False)
    plugin = os.environ.get("H3_FLASH_LIBRARY")
    if plugin:
        run["plugin"] = dict(path=plugin, sha256=hashlib.sha256(Path(plugin).read_bytes()).hexdigest())
    def wrap(owner, method, stage):
        original = getattr(owner, method)
        def traced(*a, **kw):
            writer.record(stage+".input", dict(args=a, kwargs=kw), dump="audio" in stage)
            if stage in ("video.decode", "audio.decode"):
                latent = a[0] if a else kw.get("z", kw.get("latents"))
                writer.tensor_snapshot(stage.replace(".", "-")+"-input", latent)
            value = original(*a, **kw)
            writer.record(stage+".output", value, dump="audio" in stage)
            return value
        setattr(owner, method, traced)
        def restore():
            if getattr(owner, method) is traced:
                setattr(owner, method, original)
        restores.append(restore)
    def install(report, **owners):
        profiler = original_install(report, **owners)
        if writer:
            wrap(owners["text_encoder"].model, "forward", "condition")
            wrap(owners["vae"], "encode", "video.encode")
            wrap(owners["vae"], "decode", "video.decode")
            wrap(owners["audio_vae"], "decode", "audio.decode")
            for name, module in owners["audio_vae"].named_modules():
                if name in ("dec_in_proj", "pre_block", "decoder.conv_pre", "decoder.conv_post",
                            "decoder.activation_post") or name.startswith(("decoder.ups.", "decoder.resblocks.")) and name.count(".") == 2:
                    def hook(owner, values, result, name=name):
                        writer.record("audio.layer."+name, result, dump=True)
                    handles.append(module.register_forward_hook(hook))
            run["audio_parameters"] = {n: dict(dtype=str(p.dtype), shape=list(p.shape))
                                       for n, p in owners["audio_vae"].named_parameters()}
        return profiler
    def report(owner, total):
        run["timings"] = {k: sum(v) for k, v in owner.values.items()}
        original_report(owner, total)
    entry.install_component_profilers = install
    entry.TimingReport.print_report = report
    previous_argv = sys.argv
    sys.argv = [str(entry_path), *argv, "--output", str(output/"output.mp4")]
    try:
        entry.main()
        if writer and set(native_steps) != set(tracer.native_seen):
            raise RuntimeError("Requested native trace steps were not executed")
        media = output/"output.mp4"
        run.update(complete=True, dit_seconds=tracer.latencies,
                   steady_dit_median=statistics.median(tracer.latencies[1:]) if len(tracer.latencies)>1 else None,
                   media_sha256=hashlib.sha256(media.read_bytes()).hexdigest())
    except BaseException as exc:
        run["error"] = repr(exc)
        raise
    finally:
        restore_runner()
        for handle in handles: handle.remove()
        for restore in reversed(restores): restore()
        entry.install_component_profilers = original_install
        entry.TimingReport.print_report = original_report
        sys.argv = previous_argv
        if writer: writer.close(run["complete"])
        (output/"run.json").write_bytes((json.dumps(run, indent=2)+"\n").encode())


if __name__ == "__main__":
    main()
