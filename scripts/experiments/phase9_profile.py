"""Analyze Chrome/Kineto GPU intervals without adding overlapping kernel times into wall time."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re


def merge(intervals):
    result=[]
    for a,b in sorted(intervals):
        if not (math.isfinite(a) and math.isfinite(b) and b>=a): raise ValueError("Invalid trace interval")
        if result and a<=result[-1][1]: result[-1][1]=max(result[-1][1],b)
        else: result.append([a,b])
    return result


def duration(intervals):
    return sum(b-a for a,b in merge(intervals))


def intersection(a,b):
    a,b=merge(a),merge(b);i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]: i+=1
        else: j+=1
    return total


def category(name):
    text=name.lower()
    if re.search(r"nccl|rccl|all.?reduce|reduce.?scatter|all.?gather|broadcast",text): return "communication"
    if re.search(r"flash|attention|fmha",text): return "attention"
    if re.search(r"gemm|matmul|rocblas|hipblas|cublas",text): return "gemm"
    return "other_compute"


def summarize(trace):
    events=trace["traceEvents"] if isinstance(trace,dict) else trace
    devices=defaultdict(lambda:defaultdict(list)); names=defaultdict(lambda:[0,0.0])
    kernel_counts=defaultdict(int)
    for event in events:
        if event.get("ph")!="X" or "dur" not in event: continue
        cat=event.get("cat","").lower()
        if cat not in ("kernel","gpu_kernel","gpu_memcpy","gpu_memset"): continue
        arguments=event.get("args",{})
        device=arguments.get("device",arguments.get("Device Id"))
        if isinstance(device,bool) or not str(device).isdigit():
            raise ValueError("GPU event needs an explicit nonnegative device ID; pid is not a device")
        device=str(int(device))
        kind="memory" if cat in ("gpu_memcpy","gpu_memset") else category(event.get("name",""))
        start=float(event["ts"]); end=start+float(event["dur"])
        if not math.isfinite(start) or not math.isfinite(end) or end<start: raise ValueError("Invalid GPU interval")
        devices[device][kind].append((start,end))
        if cat in ("kernel","gpu_kernel"): kernel_counts[device]+=1
        key=(device,kind,event.get("name",""));names[key][0]+=1;names[key][1]+=end-start
    reports={}
    for device,groups in devices.items():
        all_events=[v for rows in groups.values() for v in rows]
        compute=[v for key,rows in groups.items() if key in ("attention","gemm","other_compute") for v in rows]
        communication=groups.get("communication",[])
        union=duration(all_events)
        span=max(b for _,b in all_events)-min(a for a,_ in all_events)
        overlap=intersection(compute,communication)
        reports[device]=dict(kernel_events=kernel_counts[device],observed_span_ms=span/1000,busy_union_ms=union/1000,
                             gap_within_observed_span_ms=(span-union)/1000,
                             category_union_ms={k:duration(v)/1000 for k,v in groups.items()},
                             communication_compute_overlap_ms=overlap/1000,
                             exposed_communication_ms=(duration(communication)-overlap)/1000)
    top=sorted((dict(device=d,category=k,name=n,calls=v[0],summed_duration_ms=v[1]/1000)
                for (d,k,n),v in names.items()),key=lambda r:-r["summed_duration_ms"])[:50]
    return dict(schema=1,gpu_events_present=bool(sum(kernel_counts.values())),devices=reports,top_kernels=top,
                model_acceptance=False,wall_clock_speedup=None,
                limitations=["Kernel-name classification is heuristic; inspect raw traces.",
                             "Category unions overlap; never sum them as pipeline wall time.",
                             "Unobserved GPU work and profiler overhead prevent causal wall-time claims.",
                             "No GPU events means the native path was not measured, not zero device time."])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists(): ap.error("Choose a fresh output")
    from os import environ
    afs=environ.get("IFAB_AFS_HOME")
    if afs and args.output.resolve().is_relative_to(Path(afs).resolve()): ap.error("Output must remain outside AFS")
    args.output.write_text(json.dumps(summarize(json.loads(args.trace.read_text())),indent=2)+"\n")


if __name__=="__main__": main()
