"""Private-node experiment matrix. Default only writes a CPU plan.

No allocation, cache restoration, downloads, AFS writes, or native rebuilds.
Execution requires a prepared gfx936 install and explicit --execute.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import itertools
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
import sys
import tarfile
import time
from phase8_trace import private_output
from phase9_attention import verify_bundle
from phase10_attention_worker import verified_baseline, BEST_BUNDLE_SHA256
from phase10_evidence import checked_attention
from phase10_jobs import Jobs, atomic, exclusive, files_in, lanes
from phase10_profile import device_ids, sha, stamp

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]
FLAGS = dict(H3_EXPERIMENT_MHA_QUERY_CHUNK="0", H3_EXPERIMENT_MHA_HEAD_BATCH="0",
    H3_EXPERIMENT_SWIGLU_ROWS="0", H3_EXPERIMENT_SNAKE_PREPARED="0", H3_EXPERIMENT_SNAKE_STRIDED="0",
    H3_EXPERIMENT_ADDCMUL_ROWS="1", H3_EXPERIMENT_ADALN_INPLACE="1", H3_EXPERIMENT_ADALN_BATCH="1",
    H3_EXPERIMENT_ROPE_INPLACE="1", H3_EXPERIMENT_ADALN_INDEXED="0", H3_EXPERIMENT_RMS_ADALN="1",
    H3_EXPERIMENT_GATE_RMS_ADALN="1", H3_EXPERIMENT_QK_NORM_ROPE="1",
    H3_EXPERIMENT_FUSED_RMS_BLOCK="0", H3_EXPERIMENT_VIDEO_POSTPROCESS="0",
    H3_EXPERIMENT_DETERMINISTIC_BLAS="0", H3_EXPERIMENT_GEMM_LT="0",
    H3_EXPERIMENT_TP_OVERLAP_ROWS="0", H3_EXPERIMENT_VAE_DEVICES="4",
    H3_EXPERIMENT_FLASH_GFX936="1", H3_GEMM_LT_TRACE="0", H3_GEMM_LT_TABLE="",
    H3_COMPARE_BF16_MATH="1")


def configurations():
    return [dict(block_m=m,block_n=n,num_warps=4,num_stages=s,prefetch_v=p)
            for p,m,n,s in itertools.product((False,True),(128,64),(32,64),(1,2))]


def config_id(row):
    return "m%d-n%d-w%d-s%d-p%d" % tuple(row[k] for k in
                ("block_m","block_n","num_warps","num_stages","prefetch_v"))


def read_complete(path):
    value = json.loads(path.read_text())
    if value.get("complete") is not True: raise ValueError("Incomplete evidence: "+str(path))
    return value


def environment(args):
    return dict(FLAGS, INFINI_ROOT=str(args.library_root),
        H3_FLASH_LIBRARY=str(args.baseline_bundle/"libh3_flash_dispatch.so"),
        LD_LIBRARY_PATH=str(args.library_root/"lib")+os.pathsep+os.environ.get("LD_LIBRARY_PATH",""),
        PYTHONPATH=os.environ.get("PYTHONPATH",""))


def plan(args):
    files = [*HERE.glob("phase10_*.py"),*HERE.glob("phase9_*.py"),
        HERE/"phase7_build_flash.py",HERE/"phase7_flash_plugin.cpp.in",HERE/"phase8_attention_worker.py",
        HERE/"phase8_trace.py",HERE/"phase6_gemm_probe.py",HERE/"phase6_gemm_select.py",HERE/"phase7_gemm_shapes.py",
        SOURCE/"InfiniCore/src/infinicore/ops/multi_head_attention/hygon/h3_flash_abi.h",
        SOURCE/"scripts/h3_preflight.py",SOURCE/"InfiniLM/examples/minimax_h3.py",SOURCE/"InfiniLM/examples/minimax_h3_compare.py"]
    assets = dict(baseline_manifest=args.baseline_bundle/"bundle.json",first_frame=args.first_frame)
    if args.freeze_dir:
        assets.update(frozen_metadata=args.freeze_dir/"freeze.json",frozen_state=args.freeze_dir/"upstream.pt")
    return dict(schema=1,configurations=configurations(),
        source_sha256={str(p.relative_to(SOURCE)):sha(p) for p in sorted(set(files))},
        assets={k:dict(path=str(p),sha256=sha(p) if p.is_file() else None) for k,p in assets.items()},
        paths=dict(workspace=str(args.workspace_root),baseline_bundle=str(args.baseline_bundle),
                   install=str(args.library_root),model=str(args.model_path),python=str(args.python)),
        generation=dict(mode="fl2va",prompt=args.prompt,num_frames=124,steps=50,seed=42,vae_devices=4),
        devices=args.devices,physical_trace_devices=args.trace_devices,
        baseline_bundle_sha256=BEST_BUNDLE_SHA256,baseline_step_seconds=4.394788252655417,
        matrix=dict(attention_variants=16,layouts=["mixed-v","legacy"],repeats=9,finalists=2,
                    finalist_same_device_runs=2,gemm_indices=list(range(16)),
                    gemm_workspace_mb=[0,32,128],gemm_reports=2),
        schedule=["best-full-control","parallel-attention","same-device-finalists","full-candidate-1",
                  "full-candidate-2","native-profile","quality","gemm-paired"],
        compiler=args.compiler,
        model_configs={str(p.relative_to(args.model_path)):sha(p) for p in
            [args.model_path/prefix/name for prefix in ("","diffusers") for name in
             ("model_index.json","modular_model_index.json","transformer/config.json",
              "transformer_ref/config.json","text_encoder/config.json","vae/config.json",
              "audio_vae/config.json","scheduler/scheduler_config.json",
              "audio_scheduler/scheduler_config.json")] if p.is_file()},
        model_weights_hashed=False,model_acceptance=False,afs_modified=False)


def pipeline_command(args,destination,bundle):
    command = [str(args.python),"-u",str(HERE/"phase9_pipeline.py"),"--execute",
        "--workspace-root",str(args.workspace_root),"--output",str(destination),"--time-steps",
        "--dispatch-bundle",str(bundle),"--capture-dit-step","2","--verify-runtime"]
    command += (["--freeze-mode","replay","--freeze-dir",str(args.freeze_dir)] if args.freeze_dir
                else ["--freeze-mode","capture"])
    return command+["--","fl2va","--model-path",str(args.model_path),"--first-frame",str(args.first_frame),
        "--text",args.prompt,"--gpus",",".join(map(str,args.devices)),
        "--num-frames","124","--steps","50","--seed","42"]


def check_full(path):
    run=read_complete(path/"run/run.json")
    receipt=read_complete(path/"run/freeze-receipt.json")
    if len(run["step_records"])!=49 or receipt["dit_calls"]!=49:
        raise ValueError("Full 50-step request must contain 49 DiT calls")
    if not (path/"run/output.mp4").is_file(): raise ValueError("Missing full media")
    if not run.get("steady_synchronized_dit_median_seconds",0)>0:
        raise ValueError("Missing synchronized full-model timings")
    return dict(steady_step=run["steady_synchronized_dit_median_seconds"],timings=run["timings"],
                media_sha256=sha(path/"run/output.mp4"))


@contextmanager
def phase_budget(jobs,seconds):
    old=jobs.phase_deadline
    jobs.phase_deadline=min(jobs.deadline-jobs.reserve,time.monotonic()+max(0,seconds))
    try: yield
    finally: jobs.phase_deadline=old


def build_command(args,row,path):
    command=[str(args.python),"-u",str(HERE/"phase10_build_flash.py"),"--execute-build",
             "--workspace-root",str(args.workspace_root),"--output",str(path/"aot"),"--compiler",args.compiler]
    for key in ("block_m","block_n","num_warps","num_stages"):
        command+=["--"+key.replace("_","-"),str(row[key])]
    if row["prefetch_v"]: command+=["--prefetch-v"]
    return command


def check_build(path):
    report=read_complete(path/"aot/build.json")
    if report.get("arch")!="gfx936" or sha(path/"aot/libh3_flash_gfx936.so")!=report["library_sha256"]:
        raise ValueError("AOT target/library changed")
    return dict(library_sha256=report["library_sha256"],build_id=report["build_id"],
                compiler_resources=report.get("compiler_resources",{}))


def attention(jobs,args,device,tag,aot,layout,env):
    digest=sha(aot/"libh3_flash_gfx936.so")
    baseline_sha=sha(args.baseline_bundle/"libh3_flash_dispatch.so")
    def command(path):
        return [str(args.python),"-u",str(HERE/"phase10_attention_worker.py"),
            "--workspace-root",str(args.workspace_root),"--baseline-bundle",str(args.baseline_bundle),
            "--candidate",str(aot/"libh3_flash_gfx936.so"),"--device",str(device),"--layout",layout,
            "--output",str(path/"worker.json"),"--repeats","9"]
    def validate(path):
        result=checked_attention(path/"worker.json",digest,baseline_sha)
        return dict(real_worst_speedup=result["real_worst_speedup"],device=device,layout=layout)
    return jobs.run(tag,command,validate,timeout=150,estimate=35,env=env)


def run_matrix(jobs,args):
    env=environment(args)
    jobs.data["stage_skips"]={}
    # Freeze the original CLI decision in the callback so resume uses the same command.
    original_freeze=args.freeze_dir
    def command(path):
        args.freeze_dir=original_freeze
        cmd=pipeline_command(args,path/"run",args.baseline_bundle)
        return cmd
    baseline=jobs.run("full-control",command,check_full,timeout=600,estimate=450,env=env)
    if baseline is None: raise RuntimeError("No complete model control; retain evidence and stop")
    control=baseline["directory"]/"run"
    args.freeze_dir=original_freeze or control/"upstream"
    def screen(device,row):
        label=config_id(row)
        build=jobs.run("build-"+label,lambda path:build_command(args,row,path),check_build,
                       timeout=180,estimate=60,env=env)
        if not build: return None
        aot=build["directory"]/"aot"
        mixed=attention(jobs,args,device,"screen-mixed-"+label,aot,"mixed-v",env)
        legacy=attention(jobs,args,device,"screen-legacy-"+label,aot,"legacy",env)
        if not (mixed and legacy): return None
        return dict(config=row,aot=str(aot),speedup=mixed["result"]["real_worst_speedup"])
    # Leave 25 minutes for finalists, full comparisons and the native diagnostic.
    with phase_budget(jobs,min(720,max(0,jobs.remaining()-1500))):
        screened=[r for r in lanes(args.devices,configurations(),screen) if r]
    screened.sort(key=lambda r:r["speedup"],reverse=True)
    atomic(jobs.output/"screen-ranking.json",dict(ranking=screened,parallel_screening_only=True,
                                                 model_acceptance=False))
    finalists=[]
    with phase_budget(jobs,300):
        for row in screened[:2]:
            label=config_id(row["config"])
            aot=Path(row["aot"])
            results=[attention(jobs,args,args.devices[0],"final-%d-"%i+label,aot,"mixed-v",env)
                     for i in range(2)]
            if not all(results): continue
            workers=[r["directory"]/"worker.json" for r in results]
            def promote_command(path,workers=workers,aot=aot):
                return [str(args.python),"-u",str(HERE/"phase10_promote.py"),"--execute-build",
                    "--workspace-root",str(args.workspace_root),"--baseline-bundle",str(args.baseline_bundle),
                    "--aot",str(aot),"--reports",*map(str,workers),"--output",str(path/"bundle"),
                    "--compiler",args.compiler]
            def validate(path):
                receipt=verify_bundle(path/"bundle")
                return dict(bundle_sha256=sha(path/"bundle/bundle.json"),files=receipt["files"])
            promoted=jobs.run("promote-"+label,promote_command,validate,timeout=60,estimate=15,env=env)
            if promoted: finalists.append(dict(config=row["config"],bundle=promoted["directory"]/"bundle"))
    full=[]
    for i,row in enumerate(finalists):
        result=jobs.run("full-candidate-%d"%i,
            lambda path,row=row:pipeline_command(args,path/"run",row["bundle"]),check_full,
            timeout=600,estimate=430,env=env)
        if result: full.append(result["directory"]/"run")
    # Separate process-level native diagnostic; does not contribute production timing.
    run=read_complete(control/"run.json")
    snapshot=run.get("dit_snapshot")
    if snapshot:
        def profile_command(path):
            return [str(args.python),"-u",str(HERE/"phase10_profile.py"),"collect","--execute-trace",
                "--workspace-root",str(args.workspace_root),"--output",str(path/"trace"),
                "--devices",",".join(map(str,args.trace_devices)),
                "--",str(args.python),"-u",str(HERE/"phase10_dit_probe.py"),
                "--workspace-root",str(args.workspace_root),"--output",str(path/"dit"),
                "--input-snapshot",str(control/"dit-input.pt"),"--input-sha256",snapshot["sha256"],
                "--model-path",snapshot["transformer_path"],"--dispatch-bundle",str(args.baseline_bundle),
                "--gpus",",".join(map(str,args.devices))]
        def validate_profile(path):
            trace=read_complete(path/"trace/profile.json")
            read_complete(path/"trace/run.json")
            read_complete(path/"dit/run.json")
            return dict(devices=trace["devices"],top_kernels=trace["top_kernels"],diagnostic_only=True)
        jobs.run("native-profile",profile_command,validate_profile,timeout=360,estimate=240,
                 env=dict(env,H3_GEMM_LT_TRACE="1"))
    else:
        jobs.data["stage_skips"]["native-profile"]="Control did not produce a DiT snapshot"
    if full:
        def quality_command(path):
            return [str(args.python),"-u",str(HERE/"phase10_quality.py"),"--baseline",str(control),
                    "--candidates",*map(str,full),"--output",str(path/"quality.json")]
        jobs.run("quality",quality_command,lambda p:read_complete(p/"quality.json"),
                 timeout=150,estimate=35,env=env)
    else:
        jobs.data["stage_skips"]["quality"]="No promoted full candidate completed"
    shape_rows=[]
    profile_task=jobs.data["tasks"].get("native-profile",{}).get("attempts",[])
    logs=[baseline["directory"]/"terminal.log"]
    if profile_task:
        logs.append(jobs.output/profile_task[-1]["directory"]/"trace/trace.log")
    for log in logs:
        if not log.exists(): continue
        for line in log.read_text(errors="replace").splitlines():
            if "H3_GEMM_SHAPE " in line:
                try: shape_rows.append(json.loads(line.split("H3_GEMM_SHAPE ",1)[1]))
                except json.JSONDecodeError: continue
    if shape_rows: gemm(jobs,args,shape_rows,env)
    else:
        jobs.data["stage_skips"]["gemm-paired"]="No actual H3_GEMM_SHAPE evidence; no synthetic substitute"
        jobs.save()
    jobs.data["matrix_finished"]=True
    jobs.data["complete"]=not jobs.data["stage_skips"] and not any(
        "deferred" in task or not task["attempts"] or task["attempts"][-1]["state"] != "passed"
        for task in jobs.data["tasks"].values())
    jobs.save()


def gemm(jobs,args,rows,env):
    from phase7_gemm_shapes import shapes_from_rows
    shapes,skipped=shapes_from_rows(rows)
    # Descriptor log counts are NOT execution frequencies. FLOPs are an explicit
    # proxy until raw native kernel events can be joined to dimensions.
    shapes.sort(key=lambda r:2*r["tokens"]*r["out_features"]*r["in_features"],reverse=True)
    selected=shapes[:4]
    if not selected:
        jobs.data["stage_skips"]["gemm-paired"]="Observed shapes contain no eligible Lt layout"
    manifest=dict(actual_unique_shapes=shapes,selected_top4_flop_proxy=selected,
                  skipped_rows=len(skipped),not_measured_per_shape_hotspot=True)
    shape_file=jobs.output/"gemm-observed-shapes.json"
    if shape_file.exists() and json.loads(shape_file.read_text())!=manifest:
        raise ValueError("Observed GEMM shapes changed during resume")
    if not shape_file.exists(): atomic(shape_file,manifest)
    units=list(itertools.product(range(len(selected)),(0,32,128)))
    def worker(device,unit):
        index,workspace=unit
        reports=[]
        for repeat in range(2):
            key="gemm-%d-w%d-r%d"%(index,workspace,repeat)
            shape=selected[index]
            def command(path):
                target=path/"shapes.json"
                if target.exists():
                    if json.loads(target.read_text())!=[shape]: raise ValueError("GEMM shape changed")
                else: atomic(target,[shape])
                return [str(args.python),"-u",str(HERE/"phase6_gemm_probe.py"),
                    "--workspace-root",str(args.workspace_root),"--library-root",str(args.library_root),
                    "--output",str(path/"probe.json"),"--shapes",str(target),"--device",str(device),
                    "--workspace-mb",str(workspace),"--timing-mode","paired","--repeats","7","--warmup","3"]
            result=jobs.run(key,command,lambda path:read_complete(path/"probe.json"),
                            timeout=180,estimate=75,env=env)
            if result: reports.append(result["directory"]/"probe.json")
        if len(reports)==2:
            from phase6_gemm_select import select
            digests=[sha(p) for p in reports]
            try:
                if len(set(digests))!=2: raise ValueError("Duplicate paired GEMM report")
                chosen=select([json.loads(p.read_text()) for p in reports],require_paired=True)
            except Exception as exc:
                return dict(shape=index,workspace_mb=workspace,device=device,error=repr(exc),
                            report_sha256=digests,incomplete=True)
            return dict(shape=index,workspace_mb=workspace,device=device,selected=chosen,
                        report_sha256=digests)
        return dict(shape=index,workspace_mb=workspace,device=device,incomplete=True)
    with phase_budget(jobs,min(600,max(0,jobs.remaining()-180))):
        results=lanes(args.devices,units,worker)
    if any(r.get("incomplete") for r in results):
        jobs.data["stage_skips"]["gemm-paired"]="One or more paired selections remain incomplete"
    atomic(jobs.output/"gemm-selection.json",dict(results=results,model_acceptance=False))
    lines=["# Paired evidence only; no full model Lt acceptance"]
    for result in results:
        for key,value in result.get("selected",{}).items():
            lines.append(key+" = "+str(value["index"]))
    (jobs.output/"gemm-candidate-table.txt").write_bytes(("\n".join(lines)+"\n").encode())


def export(output):
    manifest=files_in(output)
    manifest.pop("evidence-manifest.json",None)
    atomic(output/"evidence-manifest.json",dict(at=stamp(),files=manifest))
    archive=output.parent/(output.name+"-evidence.tar.gz")
    if archive.exists():
        archive=output.parent/(output.name+"-evidence-"+str(time.time_ns())+".tar.gz")
    # Export small metadata first. A large raw archive may outlive the reserve;
    # neither packing nor transfer is guaranteed to finish within three minutes.
    metadata=archive.with_name(archive.name.replace("-evidence","-metadata"))
    selected=[p for p in sorted(manifest) if (output/p).suffix in (".json",".jsonl",".log",".md",".txt",".py",".cpp",".h")
              and (output/p).stat().st_size <= (8<<20)]
    with tarfile.open(metadata,"w:gz",compresslevel=1) as stream:
        for relative in selected: stream.add(output/relative,arcname=relative,recursive=False)
        stream.add(output/"evidence-manifest.json",arcname="evidence-manifest.json",recursive=False)
    meta=dict(at=stamp(),path=str(metadata),bytes=metadata.stat().st_size,sha256=sha(metadata),
              contains_selected_metadata_only=True,member_count=len(selected)+1)
    atomic(metadata.with_suffix(".receipt.json"),meta)
    print("PHASE10_METADATA_READY",json.dumps(meta),flush=True)
    with tarfile.open(archive,"w:gz",compresslevel=1) as stream:
        for relative in sorted(manifest): stream.add(output/relative,arcname=relative,recursive=False)
        stream.add(output/"evidence-manifest.json",arcname="evidence-manifest.json",recursive=False)
    receipt=dict(at=stamp(),path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),metadata=meta)
    atomic(archive.with_suffix(".receipt.json"),receipt)
    print("PHASE10_EVIDENCE_READY",json.dumps(receipt),flush=True)
    return receipt


def summary(jobs):
    rows=[]
    parse_errors=[]
    def read(path):
        try: return json.loads(path.read_text())
        except (OSError,ValueError) as exc:
            parse_errors.append(dict(path=str(path),error=repr(exc)))
            return {}

    for name,task in sorted(jobs.data["tasks"].items()):
        last=task["attempts"][-1] if task["attempts"] else task.get("deferred",{})
        rows.append(dict(job=name,state=last.get("state","not_started"),attempts=len(task["attempts"]),
            outer_seconds=last.get("outer_seconds"),error=last.get("error"),result=last.get("result"),deferred=task.get("deferred")))
    details=dict(attention=[],full=[],profile=[],quality=[],gemm=None)
    for row in rows:
        task=jobs.data["tasks"][row["job"]]
        if not task["attempts"]: continue
        directory=jobs.output/task["attempts"][-1]["directory"]
        worker=directory/"worker.json"
        if worker.is_file():
            data=read(worker)
            details["attention"].append(dict(job=row["job"],state=row["state"],device=data.get("device"),
                layout=data.get("layout"),worst_speedup=data.get("real_worst_speedup"),
                cases=[{k:c.get(k) for k in ("query","key","padded","speedup","baseline","candidate","graph",
                    "versus_previous_best","graph_updated_inputs","peak_allocated_bytes","peak_reserved_bytes")}
                    for c in data.get("cases",[])]))
        if row["job"].startswith("full-"):
            details["full"].append(row)
        if row["job"]=="native-profile":
            details["profile"].append(row)
        if row["job"]=="quality" and (directory/"quality.json").is_file():
            details["quality"].append(read(directory/"quality.json"))
    selection=jobs.output/"gemm-selection.json"
    if selection.is_file(): details["gemm"]=read(selection)
    prefixes={"build-":16,"screen-":32,"final-":4,"promote-":2,"full-":3,"native-profile":1,"quality":1,"gemm-":24}
    coverage={key:dict(maximum=count,passed=sum(r["state"]=="passed" and r["job"].startswith(key) for r in rows),
                      attempted=sum(r["attempts"]>0 and r["job"].startswith(key) for r in rows))
              for key,count in prefixes.items()}
    report=dict(at=stamp(),jobs=rows,coverage=coverage,details=details,
        complete=jobs.data.get("complete",False),evidence_parse_errors=parse_errors,stage_skips=jobs.data.get("stage_skips",{}),
        matrix_finished=jobs.data.get("matrix_finished",False),
        failures=sum(r["state"] in ("failed","interrupted") for r in rows),
        budget_deferred=sum("deferred" in t for t in jobs.data["tasks"].values()),
        model_acceptance=False,baseline_step_seconds=4.394788252655417,
        limitations=["Parallel screening requires same-device finalist remeasurement.",
                     "Profiler and synchronized diagnostics are not production latency.",
                     "No official same-round PyTorch speedup or audio quality acceptance."])
    atomic(jobs.output/"summary.json",report)
    table=["# Phase10 experiment summary","","Generated: "+report["at"],"",
           "| Job | State | Outer seconds |","|---|---|---:|"]
    table+=["| %s | %s | %s |"%(r["job"],r["state"],
             "" if r["outer_seconds"] is None else "%.3f"%r["outer_seconds"]) for r in rows]
    table+=["","Maximum coverage is conditional on numerical qualification and remaining budget.","",
            "| Stage | Passed | Maximum |","|---|---:|---:|"]
    table+=["| %s | %s | %s |"%(name,c["passed"],c["maximum"]) for name,c in coverage.items()]
    table+=["","Skipped stages: "+json.dumps(report["stage_skips"]),
            "Detailed errors, memory, compiler resources, timings and comparisons: summary.json"]
    (jobs.output/"summary.md").write_bytes(("\n".join(table)+"\n").encode())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--baseline-bundle",type=Path,required=True)
    ap.add_argument("--library-root",type=Path,required=True)
    ap.add_argument("--model-path",type=Path,required=True)
    ap.add_argument("--first-frame",type=Path,default=SOURCE/"inputs/first_frame.png")
    ap.add_argument("--freeze-dir",type=Path)
    ap.add_argument("--prompt",default="A dog falls asleep")
    ap.add_argument("--devices",type=device_ids,default=device_ids("0,1,2,3,4,5,6,7"))
    ap.add_argument("--trace-devices",type=device_ids,default=device_ids("0,1,2,3,4,5,6,7"))
    ap.add_argument("--python",type=Path,default=Path(sys.executable))
    ap.add_argument("--compiler",default="c++")
    ap.add_argument("--minutes",type=float,default=55)
    ap.add_argument("--reserve-seconds",type=int,default=180)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--retry-failed",action="store_true")
    args=ap.parse_args()
    args.workspace_root=args.workspace_root.resolve()
    for name in ("output","baseline_bundle","library_root","first_frame"):
        setattr(args,name,private_output(args.workspace_root,getattr(args,name)))
    if args.freeze_dir: args.freeze_dir=private_output(args.workspace_root,args.freeze_dir)
    args.model_path=args.model_path.resolve()
    if len(args.devices)!=8 or len(args.trace_devices)!=8: ap.error("Matrix requires eight distinct device IDs")
    if not math.isfinite(args.minutes) or args.minutes*60<=args.reserve_seconds or args.reserve_seconds<120:
        ap.error("Budget must leave at least 120 seconds reserved for evidence recovery")
    current_plan=plan(args)
    if not args.execute:
        args.output.mkdir(parents=True,exist_ok=False)
        atomic(args.output/"plan.json",dict(at=stamp(),execute=False,plan=current_plan,
            budget_seconds=args.minutes*60,reserve_seconds=args.reserve_seconds))
        print("PHASE10_MATRIX_PLAN_ONLY",len(configurations()),args.output,flush=True)
        return
    if sys.platform!="linux": ap.error("Execution requires the authorized actual Linux gfx936 node")
    verified_baseline(args.baseline_bundle)
    if not args.first_frame.is_file() or not args.model_path.is_dir() or not args.library_root.is_dir():
        ap.error("Prepared model/install/input assets required; no download or native rebuild here")
    # Query the exact child interpreter and prepared environment. Do not prepend
    # incomplete source Python packages over an installed native extension.
    query=[str(args.python),"-u",str(HERE/"phase10_runtime.py"),
        "--workspace-root",str(args.workspace_root),"--library-root",str(args.library_root),
        "--devices",",".join(map(str,args.devices))]
    print("PHASE10_RUNTIME_PREFLIGHT",json.dumps(query),flush=True)
    raw=subprocess.check_output(query,env={**os.environ,**environment(args)},text=True,timeout=60)
    print(raw,end="",flush=True)
    platform=json.loads(next(line[len("PHASE10_RUNTIME "):] for line in reversed(raw.splitlines())
                             if line.startswith("PHASE10_RUNTIME ")))
    platform["compiler_version"]=subprocess.check_output([args.compiler,"--version"],text=True,timeout=15)
    platform["inherited_runtime_environment"]={k:v for k,v in os.environ.items() if
        k.startswith(("HIP","ROCR","CUDA","HSA","INFINI","H3_")) or
        k in ("PATH","PYTHONPATH","LD_LIBRARY_PATH")}
    current_plan["platform"]=platform
    args.output.mkdir(parents=True,exist_ok=True)
    with exclusive(args.output):
        jobs=Jobs(args.output,current_plan,args.minutes*60,args.reserve_seconds,args.resume,args.retry_failed)
        try:
            # Snapshot only SHA-bound sources and the small immutable baseline
            # bundle, never the full native install or model weights.
            for relative,digest in current_plan["source_sha256"].items():
                source=SOURCE/relative
                destination=args.output/"sources"/relative
                if destination.exists():
                    if sha(destination)!=digest: raise ValueError("Archived source changed")
                else:
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(source,destination)
            base_manifest=verified_baseline(args.baseline_bundle)
            for name,digest in {**base_manifest["files"],"bundle.json":BEST_BUNDLE_SHA256}.items():
                destination=args.output/"baseline-bundle"/name
                if destination.exists():
                    if sha(destination)!=digest: raise ValueError("Archived baseline changed")
                else:
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(args.baseline_bundle/name,destination)
            run_matrix(jobs,args)
        except BaseException as exc:
            jobs.data["coordinator_error"]=repr(exc)
            raise
        finally:
            jobs.save()
            try:
                summary(jobs)
            finally:
                export(args.output)


if __name__=="__main__":
    main()
