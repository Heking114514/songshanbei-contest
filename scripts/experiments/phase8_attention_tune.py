"""Plan or later execute a gfx936 AOT sweep. No API, resource request, or AFS write."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import itertools
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time


def configs():
    rows = [dict(block_m=m, block_n=n, num_warps=w, num_stages=s)
            for m, n, w, s in itertools.product((32, 64, 128), (32, 64, 128), (4, 8), (1, 2))]
    baseline = dict(block_m=64, block_n=64, num_warps=4, num_stages=1)
    rows.sort(key=lambda r: (r != baseline, r["num_stages"], r["num_warps"], -r["block_m"], -r["block_n"]))
    return rows


def identity(row):
    return "m{block_m}-n{block_n}-w{num_warps}-s{num_stages}".format(**row)


def run_logged(command, log, timeout):
    print("PHASE8_COMMAND", json.dumps(command), flush=True)
    events = queue.Queue()
    start = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1,
                               start_new_session=True)
    def read():
        try:
            for line in process.stdout:
                events.put(line)
        finally:
            events.put(None)
    thread = threading.Thread(target=read, daemon=True); thread.start()
    try:
        with log.open("x", encoding="utf-8", newline="\n") as output:
            while True:
                elapsed = time.monotonic()-start
                if elapsed > timeout:
                    raise TimeoutError(f"Child exceeded {timeout}s")
                try:
                    line = events.get(timeout=min(15, max(0.01, timeout-elapsed)))
                except queue.Empty:
                    print("PHASE8_HEARTBEAT", round(time.monotonic()-start, 1), flush=True)
                    continue
                if line is None:
                    break
                output.write(line); output.flush(); print(line, end="", flush=True)
        return process.wait(timeout=max(1, timeout-(time.monotonic()-start)))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        thread.join(timeout=5)


def checked_worker(directory, baseline_sha, expected_worker_sha=None):
    """Bind a complete six-case result to both libraries before reuse or ranking."""
    directory = Path(directory)
    path = directory/"worker.json"
    worker_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_worker_sha is not None and worker_sha != expected_worker_sha:
        raise RuntimeError("Cached worker result changed")
    result = json.loads(path.read_text())
    if not result.get("complete") or not result.get("screening_pass"):
        raise RuntimeError("Incomplete worker result")
    cases = result["cases"]
    expected = {(q, k, padded) for q, k in ((65, 33), (2048, 2048), (27570, 27570))
                for padded in (False, True)}
    actual = [(r["query"], r["key"], r["padded"]) for r in cases]
    if len(actual) != len(expected) or set(actual) != expected or not all(
            r["repeat_bitwise"] and r["versus_v4"]["finite"]
            and r["graph_updated_inputs"]["finite"] and r["speedup"] > 0 for r in cases):
        raise RuntimeError("Incomplete shape/numerical coverage")
    library = directory/"aot/libh3_flash_gfx936.so"
    if hashlib.sha256(library.read_bytes()).hexdigest() != result["libraries"]["candidate"]["sha256"]:
        raise RuntimeError("Cached candidate library changed")
    if result["libraries"]["baseline"]["sha256"] != baseline_sha:
        raise RuntimeError("Worker used a different baseline")
    worst = min(r["speedup"] for r in cases if r["query"] == 27570)
    if result["real_worst_speedup"] != worst:
        raise RuntimeError("Worker ranking disagrees with measured cases")
    return result, worker_sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--execute", action="store_true", help="Explicit future node execution; default only emits plan")
    ap.add_argument("--workspace-root", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--baseline-sha256", default="05d78dddba8d4c8a0a8e90e99cceee19654cf44056e7639aedb76c5a3310eefe")
    ap.add_argument("--limit", type=int, default=36)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--compiler", default="c++")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.limit <= 36 or args.timeout < 1:
        ap.error("Invalid limit/timeout")
    selected = configs()[:args.limit]
    folder = Path(__file__).resolve().parent
    names = ["phase7_build_flash.py", "phase7_flash_kernel.py", "phase7_flash_plugin.cpp.in",
             "phase8_attention_worker.py", "phase8_attention_tune.py"]
    source = {n: hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in names}
    abi = folder.parents[1]/"InfiniCore/src/infinicore/ops/multi_head_attention/hygon/h3_flash_abi.h"
    source["h3_flash_abi.h"] = hashlib.sha256(abi.read_bytes()).hexdigest()
    plan = dict(arch="gfx936", wave_size=64, configurations=selected, source_sha256=source,
                execute=args.execute, model_acceptance=False,
                ranking="Worst speedup versus v4 across real compact+padded shapes, after numerical/repeat/graph checks")
    if not args.execute:
        afs = os.environ.get("IFAB_AFS_HOME")
        if afs and args.output.resolve().is_relative_to(Path(afs).resolve()):
            ap.error("Plan output must remain outside AFS")
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output/"plan.json"
        if path.exists():
            ap.error("Choose a fresh plan output directory")
        path.write_bytes((json.dumps(plan, indent=2)+"\n").encode())
        print("PHASE8_PLAN_ONLY", len(selected), path, flush=True)
        return
    if sys.platform != "linux" or not args.workspace_root or not args.baseline:
        ap.error("Execution needs Linux, a private workspace and the validated v4 baseline library")
    work, output, baseline = args.workspace_root.resolve(), args.output.resolve(), args.baseline.resolve()
    afs = os.environ.get("IFAB_AFS_HOME")
    if (afs and any(p.is_relative_to(Path(afs).resolve()) for p in (work, output, baseline))) or not all(
            p.is_relative_to(work) for p in (output, baseline)) or not baseline.is_file():
        ap.error("Workspace, artifacts and baseline must be outside AFS in private node storage")
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != args.baseline_sha256:
        ap.error("Baseline SHA mismatch; expected validated round2 v4")
    import torch, triton
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if not torch.version.hip or getattr(props, "gcnArchName", "").split(":")[0] != "gfx936":
        ap.error("Actual HIP gfx936 required")
    plan["platform"] = dict(torch=torch.__version__, hip=torch.version.hip, triton=triton.__version__,
                            compiler=args.compiler, device=args.device,
                            compiler_version=subprocess.check_output([args.compiler, "--version"], text=True),
                            device_name=props.name, total_memory=props.total_memory)
    plan["baseline_sha256"] = hashlib.sha256(baseline.read_bytes()).hexdigest()
    path = output/"sweep.json"
    if path.exists():
        if not args.resume:
            ap.error("Existing sweep requires --resume")
        report = json.loads(path.read_text())
        if report["plan"] != plan:
            ap.error("Source/platform/baseline/config changed; choose a fresh sweep directory")
    else:
        output.mkdir(parents=True, exist_ok=True)
        report = dict(plan=plan, cases={}, complete=False)
    def save():
        report["updated_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        tmp = output/"sweep.json.tmp"; tmp.write_bytes((json.dumps(report, indent=2)+"\n").encode())
        tmp.replace(path)
    report["complete"] = False
    report.pop("ranking", None)
    save()
    for row in selected:
        key = identity(row); dest = output/key; old = report["cases"].get(key)
        if old and old.get("state") == "passed":
            if not Path(old["directory"]).resolve().is_relative_to(output):
                raise RuntimeError("Cached attempt escaped its sweep directory")
            checked_worker(old["directory"], plan["baseline_sha256"], old["worker_sha256"])
            print("PHASE8_REUSE_VERIFIED", key, flush=True)
            continue
        if old:
            # Failed/partial variants get a new immutable attempt, never overwrite artifacts.
            attempt = old.get("attempt", 0)+1
            dest = output/(key+"-retry"+str(attempt))
        else:
            attempt = 0
        dest.mkdir(exist_ok=False)
        status = dict(state="building", config=row, attempt=attempt, directory=str(dest))
        report["cases"][key] = status; save()
        command = [sys.executable, "-u", str(folder/"phase7_build_flash.py"),
                   "--workspace-root", str(work), "--output", str(dest/"aot"), "--compiler", args.compiler]
        for name, value in row.items():
            command += ["--"+name.replace("_", "-"), str(value)]
        try:
            rc = run_logged(command, dest/"build.log", args.timeout)
            if rc: raise RuntimeError(f"Build exit {rc}")
            status["state"] = "screening"; save()
            command = [sys.executable, "-u", str(folder/"phase8_attention_worker.py"),
                       "--baseline", str(baseline), "--candidate", str(dest/"aot/libh3_flash_gfx936.so"),
                       "--device", str(args.device), "--output", str(dest/"worker.json")]
            rc = run_logged(command, dest/"worker.log", args.timeout)
            if rc: raise RuntimeError(f"Worker exit {rc}")
            result, worker_sha = checked_worker(dest, plan["baseline_sha256"])
            status.update(state="passed", real_worst_speedup=result["real_worst_speedup"],
                          worker_sha256=worker_sha)
        except Exception as exc:
            status.update(state="failed", error=repr(exc))
            print("PHASE8_VARIANT_FAILED", key, repr(exc), flush=True)
        save()
    report["complete"] = True
    report["ranking"] = sorted([dict(id=k, **v) for k, v in report["cases"].items()
                               if v["state"] == "passed"],
                              key=lambda r: r["real_worst_speedup"], reverse=True)
    save()
    print("PHASE8_SWEEP_COMPLETE", json.dumps(report["ranking"][:3]), flush=True)


if __name__ == "__main__":
    main()
