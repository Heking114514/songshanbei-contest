"""Build only a host dispatcher after two independent same-device screenings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from phase8_trace import private_output
from phase9_attention import render, verify_bundle
from phase10_evidence import promoted_policy
from phase10_profile import sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace-root", required=True, type=Path)
    ap.add_argument("--baseline-bundle", required=True, type=Path)
    ap.add_argument("--aot", required=True, type=Path)
    ap.add_argument("--reports", required=True, type=Path, nargs="+")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--compiler", default="c++")
    ap.add_argument("--execute-build", action="store_true")
    args = ap.parse_args()
    output = private_output(args.workspace_root, args.output)
    args.baseline_bundle=private_output(args.workspace_root,args.baseline_bundle)
    args.aot=private_output(args.workspace_root,args.aot)
    args.reports=[private_output(args.workspace_root,p) for p in args.reports]
    policy = promoted_policy(args.baseline_bundle, args.aot, args.reports)
    output.mkdir(parents=True, exist_ok=False)
    (output/"policy.json").write_bytes((json.dumps(policy, indent=2)+"\n").encode())
    (output/"dispatcher.cpp").write_bytes(render(policy).encode())
    if not args.execute_build:
        print("PHASE10_DISPATCH_PLAN_ONLY", output, flush=True)
        return
    for digest, record in policy["libraries"].items():
        path = (args.aot/"libh3_flash_gfx936.so" if digest == policy["phase10_candidate"]
                else args.baseline_bundle/record["file"])
        if sha(path) != digest: raise ValueError("Selected child SHA changed")
        shutil.copyfile(path, output/record["file"])
    abi = Path(__file__).resolve().parents[2]/"InfiniCore/src/infinicore/ops/multi_head_attention/hygon/h3_flash_abi.h"
    shutil.copyfile(abi, output/"h3_flash_abi.h")
    command = [args.compiler, "-std=c++17", "-O3", "-fPIC", "-shared",
               str(output/"dispatcher.cpp"), "-I", str(output), "-ldl", "-pthread",
               "-o", str(output/"libh3_flash_dispatch.so")]
    print("PHASE10_DISPATCH_BUILD", json.dumps(command), flush=True)
    subprocess.run(command, check=True)
    files = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
    (output/"bundle.json").write_bytes((json.dumps(dict(schema=1, model_acceptance=False, files=files),
                                                  indent=2)+"\n").encode())
    verify_bundle(output)


if __name__ == "__main__":
    main()
