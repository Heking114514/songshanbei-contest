"""Plan a structural attention AOT variant; only --execute-build compiles.

Private storage, exact gfx936/wave64 target, and the verified plugin ABI.
Does not enable a model flag or change the phase9 dispatcher.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo


from phase7_build_flash import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-build", action="store_true")
    parser.add_argument("--prefetch-v", action="store_true")
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-m", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--block-n", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--num-stages", type=int, choices=(1, 2), default=1)
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--hip-root", type=Path, default=Path("/opt/dtk"))
    args = parser.parse_args()
    work, output = args.workspace_root.resolve(), args.output.resolve()
    afs = os.environ.get("IFAB_AFS_HOME")
    if (afs and work.is_relative_to(Path(afs).resolve())) or not output.is_relative_to(work):
        parser.error("Output and workspace must resolve to private storage outside AFS")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new empty directory per AOT variant")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(output / "triton-cache")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    source_dir = Path(__file__).resolve().parent
    abi = source_dir.parents[1] / "InfiniCore/src/infinicore/ops/multi_head_attention/hygon"
    config = {"block_m": args.block_m, "block_n": args.block_n,
              "num_warps": args.num_warps, "num_stages": args.num_stages,
              "prefetch_v": args.prefetch_v}
    report = {"at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "arch": "gfx936", "wave_size": 64, "config": config,
              "complete": False, "gpu_tested": False, "afs_modified": False,
              "execute_build": args.execute_build, "variant": "split_tail",
              "model_acceptance": False}

    def save():
        (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    if not args.execute_build:
        report["planned_sources"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source_dir / "phase10_flash_kernel.py", Path(__file__),
                      source_dir / "phase7_flash_plugin.cpp.in", abi / "h3_flash_abi.h")}
        save()
        print("PHASE10_AOT_PLAN_ONLY", json.dumps(report), flush=True)
        return
    if sys.platform != "linux":
        raise RuntimeError("AOT execution requires the Linux gfx936 toolchain")
    try:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        kernel_path = source_dir / "phase10_flash_kernel.py"
        spec = importlib.util.spec_from_file_location("phase10_flash_kernel", kernel_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        signature = {name: "*bf16" for name in ("Q", "K", "V", "O")}
        signature.update({name: "i32" for name in ("SQ", "SK", "H")})
        signature["SCALE"] = "fp32"
        signature.update({prefix + axis: "i64" for prefix in "QKVO" for axis in "BSH"})
        print("PHASE10_AOT_COMPILE_START", json.dumps(report), flush=True)
        compiled = triton.compile(
            ASTSource(fn=module.h3_flash_gfx936, signature=signature,
                      constexprs={"BM": args.block_m, "BN": args.block_n,
                                  "PREFETCH_V": args.prefetch_v}),
            target=GPUTarget("hip", "gfx936", 64),
            options={"num_warps": args.num_warps, "num_stages": args.num_stages})
        if any(getattr(compiled.metadata, name, 0) for name in
               ("global_scratch_size", "profile_scratch_size")):
            raise ValueError("Kernel requires an unimplemented scratch ABI")
        code = compiled.asm["hsaco"]
        if not isinstance(code, bytes) or not code:
            raise ValueError("Compiler did not return a gfx936 code object")
        (output / "kernel.hsaco").write_bytes(code)
        template = (source_dir / "phase7_flash_plugin.cpp.in").read_text()
        source_identity = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (kernel_path, source_dir / "phase7_flash_plugin.cpp.in",
                                        abi / "h3_flash_abi.h")}
        build_id = hashlib.sha256(json.dumps({
            "arch": report["arch"], "config": config, "sources": source_identity,
            "triton": triton.__version__, "hsaco": hashlib.sha256(code).hexdigest()
        }, sort_keys=True).encode()).hexdigest()
        cpp = output / "plugin.cpp"
        cpp.write_text(render(template, code, compiled.metadata, config, build_id))
        library = output / "libh3_flash_gfx936.so"
        command = [args.compiler, "-std=c++17", "-O3", "-fPIC", "-shared",
                   "-D__HIP_PLATFORM_AMD__", "-I"+str(abi),
                   "-I"+str(args.hip_root / "include"), str(cpp),
                   "-L"+str(args.hip_root / "lib"), "-lamdhip64", "-o", str(library)]
        metadata = compiled.metadata._asdict() if hasattr(compiled.metadata, "_asdict") else getattr(compiled.metadata, "__dict__", {})
        resources = {key: value for key, value in metadata.items()
                     if isinstance(value, (str, int, float, bool, type(None)))}
        report["compiler_resources"] = resources
        report.update(triton=triton.__version__, build_id=build_id,
                      kernel_entry=compiled.metadata.name, shared_bytes=compiled.metadata.shared,
                      hsaco_sha256=hashlib.sha256(code).hexdigest(), sources=source_identity,
                      command=command)
        save()
        print("PHASE10_PLUGIN_LINK_START", json.dumps(command), flush=True)
        subprocess.run(command, check=True)
        report.update(complete=True, library=str(library),
                      library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save()
        print("PHASE10_AOT_END", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
