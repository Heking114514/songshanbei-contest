"""Build one gfx936/wave64 AOT flash variant in private node storage.

No resource submission or model execution. Invoked only after user authorizes
server work. Target compilation must succeed; no gfx928/gfx90a substitution.
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


def render(template, code, metadata, config, build_id):
    name = metadata.name
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("Invalid compiled entry-point")
    if metadata.num_warps != config["num_warps"] or metadata.shared < 0:
        raise ValueError("Unexpected compile metadata")
    # Verify wave64 thread geometry against an exposed compiler field if present.
    warp_size = getattr(metadata, "warp_size", 64)
    if warp_size != 64:
        raise ValueError("Compiled target is not wave64")
    replacements = {
        "__CODE_OBJECT_BYTES__": ",".join(str(byte) for byte in code),
        "__BLOCK_M__": str(config["block_m"]), "__BLOCK_N__": str(config["block_n"]),
        "__THREADS__": str(config["num_warps"] * 64),
        "__SHARED__": str(metadata.shared), "__KERNEL_NAME__": name,
        "__BUILD_ID__": build_id}
    for key, value in replacements.items():
        template = template.replace(key, value)
    if re.search(r"__[A-Z][A-Z_]+__", template):
        raise ValueError("Unresolved template field")
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-m", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--block-n", type=int, choices=(32, 64, 128), default=64)
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
              "num_warps": args.num_warps, "num_stages": args.num_stages}
    report = {"at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "arch": "gfx936", "wave_size": 64, "config": config,
              "complete": False, "gpu_tested": False, "afs_modified": False}

    def save():
        (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        kernel_path = source_dir / "phase7_flash_kernel.py"
        if not kernel_path.is_file():
            raise FileNotFoundError("Missing required phase7 AOT kernel: " + str(kernel_path))
        spec = importlib.util.spec_from_file_location("phase7_flash_kernel", kernel_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        signature = {name: "*bf16" for name in ("Q", "K", "V", "O")}
        signature.update({name: "i32" for name in ("SQ", "SK", "H")})
        signature["SCALE"] = "fp32"
        signature.update({prefix + axis: "i64" for prefix in "QKVO" for axis in "BSH"})
        print("PHASE7_AOT_COMPILE_START", json.dumps(report), flush=True)
        compiled = triton.compile(
            ASTSource(fn=module.h3_flash_gfx936, signature=signature,
                      constexprs={"BM": args.block_m, "BN": args.block_n}),
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
        print("PHASE7_PLUGIN_LINK_START", json.dumps(command), flush=True)
        subprocess.run(command, check=True)
        report.update(complete=True, library=str(library),
                      library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save()
        print("PHASE7_AOT_END", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
