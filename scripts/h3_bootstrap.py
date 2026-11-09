"""Prepare a clean, node-local H3 environment from pinned submission assets."""
from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
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
import zipfile
from urllib.request import urlopen

from h3_source_identity import source_identity

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "requirements/h3_linux_cp311.lock.json"
ASSETS = REPO / "third_party/h3/manifest.json"


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
    return path.is_file() and path.stat().st_size == item["size"]


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
                raise RuntimeError("Wheel size mismatch: " + name)
            os.replace(partial, path)
            print("WHEEL_COMPLETE", name, flush=True)
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


def prepare_xmake(work, manifest):
    """Prefer the size-checked official self-contained tool; retain source fallback."""
    xmake = work / "xmake"
    executable = xmake / "bin/xmake"
    receipt = xmake / "tool.json"
    environment = dict(os.environ)
    environment.pop("XMAKE_PROGRAM_DIR", None)
    bundle_name = manifest["xmake"].get("bundle_path")
    kind = None
    if receipt.is_file() and executable.is_file():
        previous = json.loads(receipt.read_text())
        if ("executable_bytes" in previous
                and executable.stat().st_size != previous["executable_bytes"]):
            raise RuntimeError("Installed xmake changed")
        kind = previous["kind"]
    if not executable.exists() and bundle_name and os.environ.get("H3_XMAKE_FROM_SOURCE") != "1":
        bundle = REPO / bundle_name
        item = next(row for row in manifest["files"] if row["path"] == bundle_name)
        if not verified(bundle, item):
            raise RuntimeError("Bundled xmake size mismatch")
        executable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle, executable)
        executable.chmod(0o755)
        try:
            version = subprocess.check_output([str(executable), "--version"],
                                              env=environment, text=True, stderr=subprocess.STDOUT)
            if "3.0.3" not in version:
                raise RuntimeError("Unexpected xmake bundle version")
            subprocess.run([str(executable), "show", "--help"], env=environment,
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            kind = "official_bundle"
            print("H3_XMAKE_BUNDLE_READY", flush=True)
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            print("H3_XMAKE_SOURCE_FALLBACK", type(error).__name__, str(error)[:300], flush=True)
            executable.unlink(missing_ok=True)
    if not executable.exists():
        source = work / "xmake-src"
        if not (source / "configure").exists():
            run(["bash", "--noprofile", "--norc",
                 REPO / "third_party/h3/xmake-v3.0.3.gz.run",
                 "--noexec", "--quiet", "--target", source])
        run(["bash", "./configure", "--prefix=" + str(xmake)], cwd=source)
        run(["make", "-j" + os.environ.get("BUILD_JOBS", "120")], cwd=source)
        run(["make", "install", "PREFIX=" + str(xmake)], cwd=source)
        kind = "source"
    version = subprocess.check_output([str(executable), "--version"], env=environment, text=True)
    if "3.0.3" not in version:
        raise RuntimeError("Unexpected xmake version")
    kind = kind or "source"
    atomic_json(receipt, dict(kind=kind, executable_bytes=executable.stat().st_size))
    return version, kind


def prepare(work):
    work = check_node_path(work, os.environ.get("IFAB_AFS_HOME"))
    afs = os.environ.get("IFAB_AFS_HOME")
    if afs and REPO.is_relative_to(Path(afs).resolve()):
        raise RuntimeError("Source checkout must be on node storage, outside personal AFS")
    info = platform_info()
    manifest = verify_assets()
    lock = json.loads(LOCK.read_text())
    key = json.dumps(dict(platform=info,
        wheels=[{k: row[k] for k in ("name", "version", "filename", "size")} for row in lock["wheels"]],
        assets=[{k: row[k] for k in ("path", "size")} for row in manifest["files"]]),
        sort_keys=True, separators=(",", ":"))
    ready = work / "prepared.json"
    if ready.exists():
        previous = json.loads(ready.read_text())
        if previous["key"] != key:
            if previous.get("integrity") == "presence_and_size" or previous["platform"] != info:
                raise RuntimeError("Environment lock/platform changed; use a new dedicated work directory")
            # Migrate an old hash receipt using platform and installed version checks.
            # This keeps the already installed environment for incremental retries.
            verify_packages(work / "python", lock)
            (work / "python/.h3-lock").write_text(key + "\n")
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
                 for row, path in zip(lock["wheels"], paths)]
        wheel = next(row for row in manifest["files"] if row["path"].endswith(".whl"))
        lines.append("flagtree @ " + (REPO / wheel["path"]).as_uri())
        requirements.write_text("\n".join(lines) + "\n")
        run([python, REPO / "scripts/h3_pip.py", "install", "--no-index", "--no-deps",
             "--no-compile", "--upgrade", "--target", stage,
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
    for name, size in manifest["jit_toolchain"]["binary_sizes"].items():
        binary = toolchain / "bin" / name
        if not binary.is_file() or binary.stat().st_size != size:
            raise RuntimeError("Missing or wrong-size JIT binary: " + str(binary))
    (toolchain / ".verified").write_text(str(archive.stat().st_size))
    version, tool_kind = prepare_xmake(work, manifest)
    atomic_json(ready, dict(key=key, platform=info, wheel_count=len(lock["wheels"]),
                           prepared_at=time.time(), integrity="presence_and_size", lock_bytes=LOCK.stat().st_size,
                           assets_bytes=ASSETS.stat().st_size, xmake_version=version, xmake_kind=tool_kind,
                           native_source_build_validated=False))
    print("H3_REPRO_PREPARED", ready, flush=True)


def file_metadata(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def config_key(component):
    source = REPO / component
    names = ("INFINI_ROOT", "PYTHON_INCLUDE", "PYBIND11_INCLUDE", "INFINILM_CXX11_ABI",
             "INFINICORE_TORCH_DIR", "HYGON_CUDA_HOME", "HYGON_ARCH", "DTK_ROOT")
    prepared = Path(os.environ.get("H3_REPRO_WORK", str(REPO / ".h3-reproduction"))) / "prepared.json"
    data = {"source": str(source), "python": sys.executable,
            "platform_lock": json.loads(prepared.read_text())["key"] if prepared.is_file() else None,
            "compiler_env": {n: os.environ.get(n) for n in
                             ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "XMAKE_PROGRAM_DIR")},
            "xmake_metadata": file_metadata(shutil.which("xmake")) if shutil.which("xmake") else None,
            "settings": {n: os.environ.get(n) for n in names},
            "xmake": shutil.which("xmake"),
            "build_driver": (REPO / "scripts/build_hygon.sh").read_text(),
            "lua": {str(p.relative_to(source)): p.read_text()
                    for p in [source / "xmake.lua", *sorted((source / "xmake").glob("**/*.lua"))]
                    if p.is_file()}}
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


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
    from infinicore.nn.functional import snake_beta_ninetoothed
    if not callable(snake_beta_ninetoothed):
        raise RuntimeError("Snake beta Python JIT function is unavailable")
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
        native[name] = dict(path=str(path), bytes=path.stat().st_size)
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
            libraries[str(path)] = path.stat().st_size
    if not libraries:
        raise RuntimeError("No native H3 libraries found in process mappings")
    identity = source_identity(REPO)
    provenance = {str(p.relative_to(REPO)): p.stat().st_size for p in
                  [*sorted((REPO / "scripts").glob("h3_*")), REPO / "scripts/reproduce_h3.sh",
                   REPO / "scripts/build_hygon.sh", REPO / "scripts/run_h3_compare.sh",
                   REPO / "InfiniLM/examples/minimax_h3_compare.py",
                   REPO / "inputs/first_frame.png"] if p.is_file()}
    atomic_json(output, dict(passed=True, time=time.time(), source_root=str(REPO),
                             head=identity["git_head"], worktree_status=identity["git_status"],
                             source_identity=identity, source_sizes=provenance,
                             modules=modules, native=native, loaded_libraries=libraries,
                             lock_bytes=LOCK.stat().st_size, assets_bytes=ASSETS.stat().st_size,
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



WHEEL_ZIP = "h3-offline-wheels-linux-cp311.zip"


def wheel_input(wheelhouse, rows):
    """Validate an unpacked directory or the release ZIP without content hashes."""
    path = Path(wheelhouse).resolve()
    for row in rows:
        if Path(row["filename"]).name != row["filename"]:
            raise RuntimeError("Unsafe wheel filename")
    if not path.exists() and path.name == "offline-wheels":
        path = path.parent / WHEEL_ZIP
    if path.is_dir() and not any(path.glob("*.whl")) and (path.parent / WHEEL_ZIP).is_file():
        path = path.parent / WHEEL_ZIP
    if path.is_dir():
        for row in rows:
            if not verified(path / row["filename"], row):
                raise RuntimeError("Missing/wrong-size offline wheel: " + row["filename"])
        return path, False
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise RuntimeError("Put " + WHEEL_ZIP + " beside run_submission.sh, "
                           "or provide a complete offline-wheels directory")
    expected = {"offline-wheels/" + row["filename"]: row for row in rows}
    seen = set()
    with zipfile.ZipFile(path) as bundle:
        for member in bundle.infolist():
            if member.is_dir() and member.filename == "offline-wheels/":
                continue
            if (member.filename not in expected or member.filename in seen
                    or member.is_dir() or member.flag_bits & 1
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise RuntimeError("Unexpected/unsafe wheel ZIP member: " + member.filename)
            if member.file_size != expected[member.filename]["size"]:
                raise RuntimeError("Wrong-size wheel ZIP member: " + member.filename)
            seen.add(member.filename)
    if seen != expected.keys():
        raise RuntimeError("Wheel ZIP is missing locked files: " + ", ".join(sorted(expected.keys() - seen)))
    return path, True


def stage_wheels(work, wheelhouse, repo=None):
    """Install ZIP members directly on node storage, retaining size-complete files."""
    repo = REPO if repo is None else Path(repo)
    work = check_node_path(work, os.environ.get("IFAB_AFS_HOME"))
    rows = json.loads((repo / "requirements/h3_linux_cp311.lock.json").read_text())["wheels"]
    dest = work / "downloads"
    if all(verified(dest / row["filename"], row) for row in rows):
        print("WHEEL_PRESTAGED_ALL", len(rows), flush=True)
        return
    source, zipped = wheel_input(wheelhouse, rows)
    dest.mkdir(parents=True, exist_ok=True)
    with (zipfile.ZipFile(source) if zipped else contextlib.nullcontext()) as bundle:
        for index, row in enumerate(rows, 1):
            target = dest / row["filename"]
            if not verified(target, row):
                partial = target.with_suffix(target.suffix + ".offline-partial")
                print("WHEEL_EXTRACT_BEGIN" if zipped else "WHEEL_COPY_BEGIN",
                      index, len(rows), target.name, row["size"], flush=True)
                if zipped:
                    with bundle.open("offline-wheels/" + row["filename"]) as incoming, partial.open("wb") as out:
                        shutil.copyfileobj(incoming, out, 1024 * 1024)
                else:
                    shutil.copy2(source / row["filename"], partial)
                if not verified(partial, row):
                    raise RuntimeError("Offline wheel copy size mismatch: " + row["filename"])
                partial.replace(target)
            print("WHEEL_OFFLINE_COMPLETE", index, len(rows), target.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-assets")
    for name in ("validate-work", "rebase-cache"):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True, type=Path)
    p = sub.add_parser("stage-wheels")
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--wheelhouse", required=True, type=Path)
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
    elif args.command == "stage-wheels":
        stage_wheels(args.work, args.wheelhouse)
    elif args.command == "prepare":
        prepare(args.work)
    elif args.command == "config-key":
        print(config_key(args.component))
    elif args.command == "audit":
        audit(args.work, args.output)


if __name__ == "__main__":
    main()
