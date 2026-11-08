"""Prepare a clean, node-local H3 environment from pinned submission assets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "requirements/h3_linux_cp311.lock.json"
ASSETS = REPO / "third_party/h3/manifest.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)


def check_node_path(path, afs=None):
    path = Path(path).resolve()
    if afs and path.is_relative_to(Path(afs).resolve()):
        raise RuntimeError("Reproduction work must be on node storage, outside personal AFS")
    if (path == Path(path.anchor) or path == REPO or REPO.is_relative_to(path)
            or any(path.is_relative_to(REPO / part) for part in
                   ("InfiniCore", "InfiniLM", "ninetoothed", "scripts", "third_party", "requirements", "inputs", "results", ".git"))):
        raise RuntimeError("Use a dedicated reproduction work directory")
    return path


def verified(path, item):
    path = Path(path)
    return (path.is_file() and path.stat().st_size == item["size"]
            and digest(path) == item["sha256"])


def verify_assets(repo=REPO):
    manifest = json.loads((repo / "third_party/h3/manifest.json").read_text())
    for item in manifest["files"]:
        path = repo / item["path"]
        if not path.resolve().is_relative_to(repo.resolve()):
            raise RuntimeError("Asset path escapes source checkout")
        if not verified(path, item):
            raise RuntimeError("Missing or modified bundled dependency: " + item["path"])
    print("H3_BUNDLED_ASSETS_VERIFIED", flush=True)
    return manifest


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_versions(directory):
    return {normalize(d.metadata["Name"]): d.version
            for d in importlib.metadata.distributions(path=[str(directory)])}


def verify_packages(directory, lock):
    versions = installed_versions(directory)
    for row in lock["wheels"] + [dict(name="flagtree", version="0.7.0+hcu3.6")]:
        if versions.get(normalize(row["name"])) != row["version"]:
            raise RuntimeError("Installed lock mismatch: " + row["name"])


def fetch(item, directory):
    name = item["filename"]
    if Path(name).name != name:
        raise RuntimeError("Invalid wheel filename")
    path = Path(directory) / name
    if verified(path, item):
        print("WHEEL_REUSED", name, flush=True)
        return path
    partial = path.with_suffix(path.suffix + ".partial")
    last_error = None
    for attempt in range(3):
        print("WHEEL_BEGIN", name, item["size"], "attempt", attempt + 1, flush=True)
        try:
            count = 0
            last = time.monotonic()
            with urlopen(item["url"], timeout=30) as response, partial.open("wb") as out:
                while block := response.read(256 * 1024):
                    count += len(block)
                    if count > item["size"]:
                        raise RuntimeError("Wheel exceeded locked size")
                    out.write(block)
                    if time.monotonic() - last >= 10:
                        print("WHEEL_BYTES", name, count, item["size"], flush=True)
                        last = time.monotonic()
            if not verified(partial, item):
                raise RuntimeError("Wheel size/SHA mismatch: " + name)
            os.replace(partial, path)
            print("WHEEL_SHA256_OK", name, flush=True)
            return path
        except (OSError, RuntimeError) as error:
            last_error = error
            print("WHEEL_RETRY", name, type(error).__name__, flush=True)
    raise RuntimeError("Could not fetch locked wheel " + name) from last_error


def run(argv, **kwargs):
    print("RUN", " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True, **kwargs)


def platform_info():
    if sys.version_info[:2] != (3, 11) or sys.platform != "linux":
        raise RuntimeError("Requires platform CPython 3.11 on Linux")
    if platform.machine() != "x86_64":
        raise RuntimeError("Requires Linux x86_64")
    import torch
    if torch.__version__.split("+")[0] != "2.9.0" or not str(torch.version.hip).startswith("6.3"):
        raise RuntimeError("Requires preinstalled Hygon Torch 2.9.0 / HIP 6.3; do not install PyPI Torch")
    origin = Path(torch.__file__).resolve()
    if not origin.is_relative_to(Path("/usr/local")):
        raise RuntimeError("Base Torch must come from platform /usr/local, not an old private venv")
    return dict(python=sys.version, base_python=sys.executable, torch=torch.__version__,
                torch_path=str(origin), hip=torch.version.hip,
                abi=int(torch._C._GLIBCXX_USE_CXX11_ABI))


def prepare(work):
    work = check_node_path(work, os.environ.get("IFAB_AFS_HOME"))
    afs = os.environ.get("IFAB_AFS_HOME")
    if afs and REPO.is_relative_to(Path(afs).resolve()):
        raise RuntimeError("Source checkout must be on node storage, outside personal AFS")
    info = platform_info()
    manifest = verify_assets()
    lock = json.loads(LOCK.read_text())
    key = hashlib.sha256(LOCK.read_bytes() + ASSETS.read_bytes()
                         + json.dumps(info, sort_keys=True).encode()).hexdigest()
    ready = work / "prepared.json"
    if ready.exists() and json.loads(ready.read_text())["key"] != key:
        raise RuntimeError("Environment lock/platform changed; use a new dedicated work directory")
    work.mkdir(parents=True, exist_ok=True)
    downloads = work / "downloads"
    downloads.mkdir(exist_ok=True)
    venv = work / "venv"
    if not (venv / "bin/python3").exists():
        run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", venv])
    python = venv / "bin/python3"
    marker = work / "python/.h3-lock"
    if not marker.exists() or marker.read_text().strip() != key:
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda row: fetch(row, downloads), lock["wheels"]))
        # A validated install is promoted atomically, including its marker.
        # An interrupted staging install is retained for diagnosis and never reused.
        if (work / "python").exists():
            raise RuntimeError("Existing Python install has a missing/different lock; use a new work directory")
        stage = work / "python.stage"
        if stage.exists():
            os.replace(stage, work / ("python.failed-" + str(time.time_ns())))
        stage.mkdir()
        requirements = work / "local-wheels.lock.txt"
        lines = [row["name"] + " @ " + path.resolve().as_uri()
                 + " --hash=sha256:" + row["sha256"]
                 for row, path in zip(lock["wheels"], paths)]
        wheel = next(row for row in manifest["files"] if row["path"].endswith(".whl"))
        lines.append("flagtree @ " + (REPO / wheel["path"]).as_uri()
                     + " --hash=sha256:" + wheel["sha256"])
        requirements.write_text("\n".join(lines) + "\n")
        run([python, REPO / "scripts/h3_pip.py", "install", "--no-index", "--no-deps",
             "--require-hashes", "--no-compile", "--upgrade", "--target", stage,
             "-r", requirements])
        verify_packages(stage, lock)
        (stage / ".h3-lock").write_text(key + "\n")
        os.replace(stage, work / "python")
    verify_packages(work / "python", lock)
    wrapper = work / "clang-hcu"
    shutil.copyfile(REPO / "scripts/h3_clang.sh", wrapper)
    wrapper.chmod(0o755)
    toolchain = work / "toolchain"
    archive = REPO / "third_party/h3/hcu-jit-clang18-minimal.tar.gz"
    if not (toolchain / ".verified").exists():
        toolchain.mkdir(exist_ok=True)
        with tarfile.open(archive) as bundle:
            bundle.extractall(toolchain, filter="data")
    for name, sha in manifest["jit_toolchain"]["binaries"].items():
        if digest(toolchain / "bin" / name) != sha:
            raise RuntimeError("JIT binary mismatch: " + name)
    (toolchain / ".verified").write_text(digest(archive))
    xmake = work / "xmake"
    if not (xmake / "bin/xmake").exists():
        source = work / "xmake-src"
        if not (source / "configure").exists():
            run(["bash", "--noprofile", "--norc",
                 REPO / "third_party/h3/xmake-v3.0.3.gz.run",
                 "--noexec", "--quiet", "--target", source])
        run(["bash", "./configure", "--prefix=" + str(xmake)], cwd=source)
        run(["make", "-j" + os.environ.get("BUILD_JOBS", "120")], cwd=source)
        run(["make", "install", "PREFIX=" + str(xmake)], cwd=source)
    version = subprocess.check_output([str(xmake / "bin/xmake"), "--version"], text=True)
    if "3.0.3" not in version:
        raise RuntimeError("Unexpected xmake version")
    atomic_json(ready, dict(key=key, platform=info, wheel_count=len(lock["wheels"]),
                           prepared_at=time.time(), lock_sha256=digest(LOCK),
                           assets_sha256=digest(ASSETS), xmake_version=version,
                           native_source_build_validated=False))
    print("H3_REPRO_PREPARED", ready, flush=True)


def config_key(component):
    source = REPO / component
    names = ("INFINI_ROOT", "PYTHON_INCLUDE", "PYBIND11_INCLUDE", "INFINILM_CXX11_ABI",
             "INFINICORE_TORCH_DIR", "HYGON_CUDA_HOME", "HYGON_ARCH", "DTK_ROOT")
    prepared = Path(os.environ.get("H3_REPRO_WORK", str(REPO / ".h3-reproduction"))) / "prepared.json"
    data = {"source": str(source), "python": sys.executable,
            "platform_lock": json.loads(prepared.read_text())["key"] if prepared.is_file() else None,
            "compiler_env": {n: os.environ.get(n) for n in
                             ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "XMAKE_PROGRAM_DIR")},
            "xmake_sha256": digest(shutil.which("xmake")) if shutil.which("xmake") else None,
            "settings": {n: os.environ.get(n) for n in names},
            "xmake": shutil.which("xmake"),
            "build_driver": digest(REPO / "scripts/build_hygon.sh"),
            "lua": {str(p.relative_to(source)): digest(p)
                    for p in [source / "xmake.lua", *sorted((source / "xmake").glob("**/*.lua"))]
                    if p.is_file()}}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def audit(work, output):
    work = Path(work).resolve()
    lock = json.loads(LOCK.read_text())
    verify_packages(work / "python", lock)
    import importlib
    modules = {}
    for name in ("torch", "transformers", "diffusers", "torchvision", "av", "triton",
                 "ninetoothed", "infinicore", "infinilm"):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if name == "torch":
            allowed = Path("/usr/local")
        elif name in ("ninetoothed", "infinicore", "infinilm"):
            allowed = REPO / {"ninetoothed": "ninetoothed", "infinicore": "InfiniCore",
                              "infinilm": "InfiniLM"}[name]
        else:
            allowed = work / "python"
        if not origin.is_relative_to(allowed):
            raise RuntimeError("Unexpected package origin: " + name + " " + str(origin))
        modules[name] = dict(path=str(origin), version=getattr(module, "__version__", None))
    from triton.backends import backends
    if "hcu" not in backends:
        raise RuntimeError("FlagTree HCU backend absent")
    # Native extension paths must come from the new source or its new install.
    import sysconfig
    native = {}
    for name in ("infinicore", "infinilm"):
        paths = {Path(m.__file__).resolve() for m in list(sys.modules.values())
                 if getattr(m, "__file__", None)
                 and Path(m.__file__).name.startswith("_" + name)
                 and str(m.__file__).endswith(".so")}
        if len(paths) != 1:
            raise RuntimeError("Expected one loaded native extension: " + name)
        path = paths.pop()
        if not (path.is_relative_to(REPO) or path.is_relative_to(work / "install")):
            raise RuntimeError("Native extension came from another installation: " + str(path))
        native[name] = dict(path=str(path), sha256=digest(path))
    # Check the libraries actually mapped by this process, not just Python files.
    libraries = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5]).resolve()
        if path.name.startswith(("libinfini", "_infinicore", "_infinilm")):
            if not (path.is_relative_to(work / "install") or path.is_relative_to(REPO)):
                raise RuntimeError("Loaded an older native library: " + str(path))
            libraries[str(path)] = digest(path)
    if not libraries:
        raise RuntimeError("No native H3 libraries found in process mappings")
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain",
                                       "--untracked-files=normal"], text=True)
    provenance = {str(p.relative_to(REPO)): digest(p) for p in
                  [*sorted((REPO / "scripts").glob("h3_*")), REPO / "scripts/reproduce_h3.sh",
                   REPO / "scripts/build_hygon.sh", REPO / "scripts/run_h3_compare.sh",
                   REPO / "InfiniLM/examples/minimax_h3_compare.py",
                   REPO / "inputs/first_frame.png"] if p.is_file()}
    atomic_json(output, dict(passed=True, time=time.time(), source_root=str(REPO),
                             head=head, worktree_status=changed, source_sha256=provenance,
                             modules=modules, native=native, loaded_libraries=libraries,
                             lock_sha256=digest(LOCK), assets_sha256=digest(ASSETS),
                             python_include=sysconfig.get_path("include")))
    print("H3_REPRO_ORIGINS_OK", output, flush=True)


def rebase_cache(work):
    work = check_node_path(work, os.environ.get("IFAB_AFS_HOME"))
    updates = []
    for path in (work / "jit").glob("*/__grp__*"):
        data = json.loads(path.read_text())
        children = data.get("child_paths")
        if not isinstance(children, dict) or not children:
            raise RuntimeError("Invalid JIT cache group: " + str(path))
        for name, previous in children.items():
            child = path.parent / Path(previous).name
            if not child.is_file():
                raise RuntimeError("Restored JIT group missing child: " + str(child))
            children[name] = str(child.resolve())
        updates.append((path, data))
    if not updates:
        raise RuntimeError("No JIT cache groups; complete single-GPU jit/smoke first")
    for path, data in updates:
        atomic_json(path, data)
    print("H3_JIT_GROUPS_REBASED", len(updates), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-assets")
    for name in ("validate-work", "rebase-cache"):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True, type=Path)
    p = sub.add_parser("prepare")
    p.add_argument("--work", required=True, type=Path)
    p = sub.add_parser("config-key")
    p.add_argument("component", choices=["InfiniCore", "InfiniLM"])
    p = sub.add_parser("audit")
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "validate-work":
        check_node_path(args.work, os.environ.get("IFAB_AFS_HOME"))
    elif args.command == "rebase-cache":
        rebase_cache(args.work)
    elif args.command == "verify-assets":
        verify_assets()
    elif args.command == "prepare":
        prepare(args.work)
    elif args.command == "config-key":
        print(config_key(args.component))
    elif args.command == "audit":
        audit(args.work, args.output)


if __name__ == "__main__":
    main()
