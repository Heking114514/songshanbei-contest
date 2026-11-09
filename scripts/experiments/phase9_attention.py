"""Offline exact-layout policy and optional host dispatcher build; never requests a node."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics
import subprocess
import sys
import tarfile

BASELINE_SHA = "05d78dddba8d4c8a0a8e90e99cceee19654cf44056e7639aedb76c5a3310eefe"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def shape_spec(q, k, padded):
    h, d = 7, 128
    def strides(n, multiple):
        return [n*h*d*multiple, h*d*multiple, d, 1]
    return dict(batch=1, query_tokens=q, key_tokens=k, heads=h,
                strides=[strides(n, 3 if padded else 1) for n in (q,k,k)]
                        + [strides(q, 2 if padded else 1)])


def checked_cases(result):
    expected = {(q,k,p) for q,k in ((65,33),(2048,2048),(27570,27570)) for p in (False,True)}
    rows = result["cases"]
    if not result.get("complete") or not result.get("screening_pass") or result.get("arch") != "gfx936":
        raise ValueError("Worker is incomplete or for another architecture")
    if len(rows) != 6 or {(r["query"],r["key"],r["padded"]) for r in rows} != expected:
        raise ValueError("Incomplete or duplicate shape coverage")
    for r in rows:
        if r["heads"] != 7 or not r["repeat_bitwise"]:
            raise ValueError("Unsupported head count or nondeterministic worker")
        for key in ("versus_v4", "graph_updated_inputs"):
            e = r[key]
            if not e["finite"] or not math.isfinite(e["relative_l2"]) or not 0 <= e["relative_l2"] <= .005:
                raise ValueError("Numerical or graph screening failed")
        if r["query"] == 65:
            e = r["versus_math_small"]
            if not e["finite"] or not 0 <= e["relative_l2"] < .02:
                raise ValueError("Small MATH screening failed")
        for key in ("baseline","candidate","graph"):
            values = r[key]["samples_ms"]
            if len(values) < 3 or not all(math.isfinite(x) and x > 0 for x in values):
                raise ValueError("Invalid timing samples")
            if not math.isclose(statistics.median(values), r[key]["median_ms"], rel_tol=1e-9):
                raise ValueError("Timing median disagrees with raw samples")
        if not math.isclose(r["speedup"], r["baseline"]["median_ms"]/r["candidate"]["median_ms"], rel_tol=1e-9):
            raise ValueError("Speedup disagrees with timings")
    return rows


def from_archive(path, minimum_gain=.03, tie_fraction=.02):
    """Read selected members only; never extract an archive with arbitrary paths."""
    if not 0 <= minimum_gain < 1 or not 0 <= tie_fraction < 1:
        raise ValueError("Invalid gain/tie policy")
    with tarfile.open(path) as archive:
        members = {m.name:m for m in archive.getmembers()}
        if len(members) != len(archive.getmembers()):
            raise ValueError("Duplicate archive members")
        def read(name, limit=32*1024*1024):
            m = members[name]
            if not m.isfile() or m.size > limit:
                raise ValueError("Invalid evidence member")
            return archive.extractfile(m).read()
        sweeps = [n for n in members if n.endswith("/sweep.json")]
        if len(sweeps) != 1:
            raise ValueError("Exactly one sweep receipt required")
        sweep_name = sweeps[0]; prefix = sweep_name.rsplit("/",1)[0]
        sweep = json.loads(read(sweep_name))
        if not sweep["complete"] or len(sweep["cases"]) != 36:
            raise ValueError("Complete 36-worker sweep required")
        workers, libraries = [], {}
        worker_names = sorted(n for n in members if n.startswith(prefix+"/") and n.endswith("/worker.json"))
        if len(worker_names) != 36:
            raise ValueError("Missing worker receipts")
        # Sweep binds worker JSON hashes; relocated directories are resolved inside the archive.
        receipts = sweep["cases"]
        if len(receipts) != 36:
            raise ValueError("Duplicate sweep IDs")
        for name in worker_names:
            ident = PurePosixPath(name).parent.name
            raw = read(name); result = json.loads(raw)
            receipt = receipts[ident]
            if receipt["state"] != "passed": raise ValueError("Unpassed sweep worker")
            if receipt["worker_sha256"] != sha(raw):
                raise ValueError("Worker receipt hash mismatch")
            checked_cases(result)
            if result["libraries"]["baseline"]["sha256"] != BASELINE_SHA:
                raise ValueError("Wrong baseline")
            lib = result["libraries"]["candidate"]
            binary = read(str(PurePosixPath(name).parent/"aot/libh3_flash_gfx936.so"))
            if sha(binary) != lib["sha256"]:
                raise ValueError("Candidate binary hash mismatch")
            libraries[lib["sha256"]] = dict(sha256=lib["sha256"], build_id=lib["build_id"],
                                           file=lib["sha256"]+".so")
            workers.append((ident, result, sha(raw)))
        base = workers[0][1]["libraries"]["baseline"]
        libraries[BASELINE_SHA] = dict(sha256=BASELINE_SHA, build_id=base["build_id"], file=BASELINE_SHA+".so")
        rules = []
        for index in range(6):
            candidates = [(ident,r["cases"][index],r["libraries"]["candidate"]["sha256"],digest)
                          for ident,r,digest in workers]
            # Require actual matching layout; do not silently extrapolate from token count.
            q,k,p = [(r["query"],r["key"],r["padded"]) for _,r,_,_ in candidates][0]
            if any((r["query"],r["key"],r["padded"]) != (q,k,p) for _,r,_,_ in candidates):
                raise ValueError("Worker case order mismatch")
            fastest = max(r["speedup"] for _,r,_,_ in candidates)
            tied = [c for c in candidates if c[1]["speedup"] >= fastest/(1+tie_fraction)]
            winner = min(tied, key=lambda c:(max(c[1][x]["relative_l2"] for x in
                                                  ("versus_v4","graph_updated_inputs")),c[0]))
            ident,row,digest,worker_hash = winner
            selected = row["speedup"] >= 1+minimum_gain
            rules.append(dict(spec=shape_spec(q,k,p), library=digest if selected else BASELINE_SHA,
                              candidate_id=ident, worker_sha256=worker_hash,
                              measured_speedup=row["speedup"], candidate_selected=selected,
                              versus_v4_relative_l2=row["versus_v4"]["relative_l2"],
                              graph_relative_l2=row["graph_updated_inputs"]["relative_l2"]))
        used = {BASELINE_SHA,*(r["library"] for r in rules)}
        return dict(schema=1, arch="gfx936", abi=1, head_dim=128, dtype="bfloat16",
                    baseline=BASELINE_SHA, unknown_shape="baseline", model_acceptance=False,
                    source_archive_sha256=sha(Path(path).read_bytes()),
                    sweep_sha256=sha(read(sweep_name)),
                    evidence_platform=sweep["plan"]["platform"],
                    worker_source_sha256=sweep["plan"]["source_sha256"],
                    minimum_gain=minimum_gain, tie_fraction=tie_fraction,
                    libraries={k:v for k,v in sorted(libraries.items()) if k in used}, rules=rules)


def select(policy, spec):
    for row in policy["rules"]:
        if row["spec"] == spec:
            return row["library"]
    return policy["baseline"]


def render(policy):
    libs = list(policy["libraries"])
    records = [policy["libraries"][k] for k in libs]
    for r in records:
        if r["file"] != r["sha256"]+".so" or len(r["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in r["sha256"]):
            raise ValueError("Unsafe library name")
    def literal(s):
        return json.dumps(s, ensure_ascii=True)
    library_rows = ",\n".join("{"+literal(r["file"])+","+literal(r["build_id"])+"}" for r in records)
    rule_rows=[]
    for r in policy["rules"]:
        v=r["spec"]
        values=[v[k] for k in ("batch","query_tokens","key_tokens","heads")]
        strides="{"+",".join("{"+",".join(map(str,s))+"}" for s in v["strides"])+"}"
        rule_rows.append("{{"+",".join(map(str,values))+","+strides+"},"+str(libs.index(r["library"]))+"}")
    build_id="phase9-dispatch-"+sha(json.dumps(policy,sort_keys=True).encode())[:24]
    template=Path(__file__).with_name("phase9_dispatch.cpp.in").read_text()
    return (template.replace("@LIBRARIES@",library_rows).replace("@RULES@",",\n".join(rule_rows))
            .replace("@BASELINE@",str(libs.index(policy["baseline"]))).replace("@BUILD_ID@",build_id))


def verify_bundle(root):
    root=Path(root)
    manifest=json.loads((root/"bundle.json").read_text())
    policy=json.loads((root/"policy.json").read_text())
    required={"policy.json","dispatcher.cpp","h3_flash_abi.h","libh3_flash_dispatch.so",
              *(r["file"] for r in policy["libraries"].values())}
    if manifest.get("schema")!=1 or set(manifest["files"])!=required:
        raise ValueError("Incomplete bundle manifest")
    for name,digest in manifest["files"].items():
        if Path(name).name != name or sha((root/name).read_bytes()) != digest:
            raise ValueError("Bundle file SHA mismatch: "+name)
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--verify-bundle",type=Path)
    ap.add_argument("--baseline",type=Path)
    ap.add_argument("--execute-build",action="store_true")
    ap.add_argument("--workspace-root",type=Path)
    ap.add_argument("--compiler",default="c++")
    args=ap.parse_args()
    if args.verify_bundle:
        verify_bundle(args.verify_bundle); print("PHASE9_BUNDLE_SHA_OK"); return
    if not args.archive or not args.output:
        ap.error("--archive and --output required")
    from phase8_trace import private_output
    out=private_output(args.workspace_root,args.output) if args.workspace_root else args.output.resolve()
    afs=os.environ.get("IFAB_AFS_HOME")
    if afs and out.is_relative_to(Path(afs).resolve()):
        ap.error("Output must remain outside AFS")
    if args.execute_build and (sys.platform != "linux" or not args.workspace_root or not args.baseline):
        ap.error("Build requires Linux private workspace and v4 baseline")
    policy=from_archive(args.archive)
    code=render(policy)
    out.mkdir(parents=True,exist_ok=False)
    (out/"policy.json").write_text(json.dumps(policy,indent=2)+"\n")
    (out/"dispatcher.cpp").write_text(code)
    print("PHASE9_POLICY_ONLY",len(policy["rules"]),"rules",flush=True)
    if not args.execute_build: return
    # No HIP or Triton compilation: reuse the SHA-bound, previously built children.
    if sha(args.baseline.read_bytes()) != BASELINE_SHA: raise ValueError("Baseline changed")
    shutil.copyfile(args.baseline,out/(BASELINE_SHA+".so"))
    with tarfile.open(args.archive) as archive:
        wanted=set(policy["libraries"])-{BASELINE_SHA}
        for member in archive.getmembers():
            if not member.name.endswith("/aot/libh3_flash_gfx936.so"): continue
            binary=archive.extractfile(member).read(); digest=sha(binary)
            if digest in wanted: (out/(digest+".so")).write_bytes(binary); wanted.remove(digest)
        if wanted: raise ValueError("Missing selected libraries")
    abi=Path(__file__).resolve().parents[2]/"InfiniCore/src/infinicore/ops/multi_head_attention/hygon/h3_flash_abi.h"
    shutil.copyfile(abi,out/"h3_flash_abi.h")
    command=[args.compiler,"-std=c++17","-O3","-fPIC","-shared",str(out/"dispatcher.cpp"),
             "-I",str(out),"-o",str(out/"libh3_flash_dispatch.so"),"-ldl","-pthread"]
    print("PHASE9_HOST_BUILD",json.dumps(command),flush=True)
    subprocess.run(command,check=True)
    files={p.name:sha(p.read_bytes()) for p in out.iterdir() if p.is_file()}
    (out/"bundle.json").write_text(json.dumps(dict(schema=1,model_acceptance=False,files=files),indent=2)+"\n")
    verify_bundle(out)


if __name__=="__main__": main()
