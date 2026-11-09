"""Inspect the exact prepared Python/native runtime; no build or platform I/O."""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from phase10_profile import sha, device_ids


def check_paths(workspace, root, modules, mapped):
    workspace,root=Path(workspace).resolve(),Path(root).resolve()
    if not root.is_relative_to(workspace): raise ValueError("Install must be private")
    rows={}
    for name,path in modules.items():
        path=Path(path).resolve()
        if not path.is_relative_to(workspace): raise ValueError("Python origin outside private workspace: "+str(path))
        rows[name]=dict(path=str(path),sha256=sha(path))
    libraries={}
    for path in sorted(set(map(Path,mapped))):
        path=path.resolve()
        allowed=root if path.name.startswith("libinfini") else workspace
        if not path.is_relative_to(allowed): raise ValueError("Native library origin mismatch: "+str(path))
        libraries[str(path)]=sha(path)
    names={Path(p).name for p in libraries}
    if not {"libinfiniop.so","libinfinirt.so"} <= names or not any(
            name.startswith("_infinilm") for name in names):
        raise ValueError("Required Core/LM native libraries are not mapped")
    return dict(python_modules=rows,mapped_native_sha256=libraries)


def origins(workspace,root):
    names=("infinicore","infinilm","infinilm.multimodal.minimax_h3")
    modules={name:importlib.import_module(name).__file__ for name in names}
    mapped=[]
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6 or not fields[5].startswith("/"): continue
        path=fields[5]
        if ".so" in path and any(t in Path(path).name for t in ("libinfini","_infinilm","_infinicore")):
            mapped.append(path)
    return check_paths(workspace,root,modules,mapped)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace-root",type=Path,required=True)
    ap.add_argument("--library-root",type=Path,required=True)
    ap.add_argument("--devices",type=device_ids,required=True)
    args=ap.parse_args()
    if sys.platform!="linux": ap.error("Actual Linux runtime required")
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from h3_preflight import bf16_math
    bf16_math()
    import torch
    report=origins(args.workspace_root,args.library_root)
    devices=[]
    for device in args.devices:
        props=torch.cuda.get_device_properties(device)
        if not torch.version.hip or getattr(props,"gcnArchName","").split(":")[0]!="gfx936":
            raise ValueError("Actual gfx936 required on every selected GPU")
        devices.append(dict(id=device,name=props.name,memory=props.total_memory,
                            arch=getattr(props,"gcnArchName","")))
    report.update(python=str(Path(sys.executable).resolve()),torch=torch.__version__,
                  hip=torch.version.hip,devices=devices)
    print("PHASE10_RUNTIME",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
