"""Plan locally; explicitly execute later on a private gfx936 node. No platform or AFS operations."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from phase8_trace import private_output
from phase9_freeze import FrozenLoop, generation_contract
from phase9_profile import summarize
from phase9_attention import verify_bundle


def parse_steps(text):
    result={int(x) for x in text.split(",") if x.strip()}
    if any(x<0 for x in result): raise ValueError("Steps are zero based and nonnegative")
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--verify-runtime",action="store_true",help="Require prepared private Python/native origins")
    ap.add_argument("--workspace-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--freeze-mode",choices=("capture","replay"))
    ap.add_argument("--freeze-dir",type=Path,help="Replay source directory; capture always uses output/upstream")
    ap.add_argument("--freeze-budget-mib",type=int,default=512)
    ap.add_argument("--profile-steps",default="",help="e.g. 2,3; separate diagnostic runs from timing runs")
    ap.add_argument("--capture-dit-step", type=int, help="Save one exact CPU DiT input for later standalone profiling")
    ap.add_argument("--time-steps",action="store_true",help="Synchronize every DiT call; perturbing diagnostic timer")
    ap.add_argument("--dispatch-bundle",type=Path)
    ap.add_argument("generation_args",nargs=argparse.REMAINDER)
    args=ap.parse_args()
    argv=args.generation_args
    if argv[:1]==["--"]: argv=argv[1:]
    if not argv or any(x=="--legacy" or x.split("=")[0]=="--output" for x in argv):
        ap.error("Pass native generation args after --; output is managed by wrapper")
    if (args.freeze_mode=="replay")!=(args.freeze_dir is not None):
        ap.error("--freeze-dir is required only for replay")
    if args.capture_dit_step is not None and args.capture_dit_step < 0: ap.error("Capture step must be nonnegative")
    if args.freeze_budget_mib<=0: ap.error("Positive freeze budget required")
    selected=parse_steps(args.profile_steps)
    output=private_output(args.workspace_root,args.output)
    replay=private_output(args.workspace_root,args.freeze_dir) if args.freeze_dir else None
    bundle=private_output(args.workspace_root,args.dispatch_bundle) if args.dispatch_bundle else None
    scripts=Path(__file__).resolve().parent
    source=scripts.parents[1]
    source_files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  sorted(scripts.glob("phase9_*")) if p.is_file()}
    if not args.execute:
        output.mkdir(parents=True,exist_ok=False)
        (output/"plan.json").write_text(json.dumps(dict(execute=False,arguments=argv,
            source_sha256=source_files,freeze_mode=args.freeze_mode,profile_steps=sorted(selected),
            capture_dit_step=args.capture_dit_step,
            synchronized_step_timing=args.time_steps,dispatch_bundle=str(bundle) if bundle else None,
            model_acceptance=False),indent=2)+"\n")
        print("PHASE9_PLAN_ONLY",output,flush=True);return
    if sys.platform!="linux": ap.error("Execution requires actual Linux HIP gfx936 node")
    import torch
    if not torch.cuda.is_available() or not torch.version.hip: ap.error("HIP required")
    if bundle: verify_bundle(bundle)
    entry_path=source/"InfiniLM/examples/minimax_h3.py"
    sys.path.insert(0,str(source/"scripts"))
    from h3_preflight import bf16_math
    bf16_math()
    spec=importlib.util.spec_from_file_location("phase9_h3_entry",entry_path)
    entry=importlib.util.module_from_spec(spec);sys.modules[spec.name]=entry;spec.loader.exec_module(entry)
    gp=argparse.ArgumentParser(add_help=False);gp.add_argument("--gpus",default="0,1,2,3")
    known,_=gp.parse_known_args(argv);devices=entry.parse_gpu_list(known.gpus)
    for d in devices:
        arch=getattr(torch.cuda.get_device_properties(d),"gcnArchName","").split(":")[0]
        if arch!="gfx936": ap.error("Actual gfx936 required on every selected device")
    output.mkdir(parents=True,exist_ok=False)
    run=dict(complete=False,arguments=argv,source_sha256=source_files,devices=devices,
             diagnostic=bool(selected or args.freeze_mode or args.time_steps or args.capture_dit_step is not None),
             profile_steps=sorted(selected),profiled_steps=[],step_records=[],
             torch=torch.__version__,hip=torch.version.hip,model_acceptance=False,
             synchronized_step_timing=args.time_steps,
             limitations=["Upstream freeze does not hash full model weights.",
                          "Profile and input hashing are diagnostics, not production timing."])
    freezer=FrozenLoop(args.freeze_mode,replay or output/"upstream",output,
                      budget_mib=args.freeze_budget_mib) if args.freeze_mode else None
    if args.verify_runtime:
        from phase10_runtime import origins
        run["runtime_origins"]=origins(args.workspace_root,Path(os.environ["INFINI_ROOT"]))
    original_prepare=entry.prepare_generation_inputs
    original_call=entry.MiniMaxH3TransformerRunner.__call__
    original_report=entry.TimingReport.print_report
    def prepare(generation_args,*a,**kw):
        value=original_prepare(generation_args,*a,**kw)
        if freezer: freezer.contract=generation_contract(generation_args,value[0])
        return value
    def report(owner,total):
        run["timings"]={k:sum(v) for k,v in owner.values.items()}
        original_report(owner,total)
    def sync():
        for d in devices: torch.cuda.synchronize(d)
    def called(owner,inputs,*a,**kw):
        step=len(run["step_records"])
        if freezer: freezer.observe(inputs)
        if step == args.capture_dit_step:
            from dataclasses import fields
            path = output / "dit-input.pt"
            if path.exists(): raise RuntimeError("Refusing to overwrite DiT snapshot")
            tensors = {f.name: getattr(inputs, f.name).detach().cpu().clone() for f in fields(inputs)}
            size = sum(t.numel() * t.element_size() for t in tensors.values())
            if size > args.freeze_budget_mib * (1 << 20):
                raise RuntimeError("DiT snapshot exceeds configured budget")
            torch.save(tensors, path)
            if path.stat().st_size > args.freeze_budget_mib * (1 << 20):
                raise RuntimeError("Serialized DiT snapshot exceeds configured budget")
            run["dit_snapshot"] = dict(step=step, path=str(path), bytes=path.stat().st_size,
                                      transformer_path=str(owner.transformer_path),
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if args.time_steps: sync()
        started=time.perf_counter()
        if step in selected:
            trace=output/("dit-step-%03d.trace.json"%step)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False,profile_memory=False,with_stack=False) as profiler:
                with torch.profiler.record_function("h3.dit.step.%d"%step):
                    result=original_call(owner,inputs,*a,**kw)
            profiler.export_chrome_trace(str(trace))
            summary=summarize(json.loads(trace.read_text()))
            summary["expected_device_ids"]=devices
            summary["missing_device_ids"]=[d for d in devices if not summary["devices"].get(str(d),{}).get("kernel_events")]
            (output/("dit-step-%03d.profile.json"%step)).write_text(json.dumps(summary,indent=2)+"\n")
            if not summary["gpu_events_present"] or summary["missing_device_ids"]:
                raise RuntimeError("Profiler missed native GPU work on one or more selected devices")
            run["profiled_steps"].append(step)
        else: result=original_call(owner,inputs,*a,**kw)
        if args.time_steps: sync()
        elapsed=time.perf_counter()-started
        run["step_records"].append(dict(step=step,host_call_seconds=elapsed,profiled=step in selected))
        print("PHASE9_DIT_STEP",step,elapsed,"profile" if step in selected else "host-call",flush=True)
        return result
    previous_argv=sys.argv
    previous_flags={key:os.environ.get(key) for key in ("H3_EXPERIMENT_FLASH_GFX936","H3_FLASH_LIBRARY")}
    try:
        with ExitStack() as stack:
            if bundle:
                os.environ["H3_EXPERIMENT_FLASH_GFX936"]="1"
                os.environ["H3_FLASH_LIBRARY"]=str(bundle/"libh3_flash_dispatch.so")
                run["dispatch"]=verify_bundle(bundle)
            run["flags"]={k:v for k,v in os.environ.items() if k.startswith("H3_EXPERIMENT") or k=="H3_FLASH_LIBRARY"}
            plugin=os.environ.get("H3_FLASH_LIBRARY")
            if plugin: run["plugin_sha256"]=hashlib.sha256(Path(plugin).read_bytes()).hexdigest()
            if freezer:
                from diffusers.modular_pipelines.minimax_h3 import denoise
                from diffusers.schedulers import scheduling_minimax_h3
                stack.callback(freezer.install(denoise.MiniMaxH3DenoiseLoopWrapper,dict(
                    denoise=denoise.__file__,scheduler=scheduling_minimax_h3.__file__,
                    entry=entry_path,freeze=scripts/"phase9_freeze.py")))
            entry.prepare_generation_inputs=prepare
            entry.TimingReport.print_report=report
            entry.MiniMaxH3TransformerRunner.__call__=called
            sys.argv=[str(entry_path),*argv,"--output",str(output/"output.mp4")]
            entry.main()
            if args.capture_dit_step is not None and "dit_snapshot" not in run:
                raise RuntimeError("Requested DiT snapshot step was not executed")
            if set(run["profiled_steps"])!=selected: raise RuntimeError("Requested profile steps were not executed")
            run["media_sha256"]=hashlib.sha256((output/"output.mp4").read_bytes()).hexdigest()
            if args.time_steps and not selected:
                values=[r["host_call_seconds"] for r in run["step_records"]][1:]
                run["steady_synchronized_dit_median_seconds"]=statistics.median(values) if values else None
            run["complete"]=True
    except BaseException as exc:
        run["error"]=repr(exc);raise
    finally:
        entry.prepare_generation_inputs=original_prepare
        entry.TimingReport.print_report=original_report
        entry.MiniMaxH3TransformerRunner.__call__=original_call
        sys.argv=previous_argv
        for key,value in previous_flags.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value
        try:
            if freezer: freezer.close(run["complete"])
        except BaseException as exc:
            run.update(complete=False,freeze_error=repr(exc));raise
        finally:
            (output/"run.json").write_text(json.dumps(run,indent=2)+"\n")


if __name__=="__main__": main()
