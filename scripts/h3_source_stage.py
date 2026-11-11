"""Stage uploaded source and locked wheels for node-local H3 execution."""
import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

from h3_bootstrap import verify_assets, wheel_input, stage_wheels
from h3_delivery import file_row, safe_relative, verify, write_json
from h3_source_identity import source_identity

DIRECTORIES = ("InfiniCore", "InfiniLM", "ninetoothed", "scripts", "requirements",
               "third_party", "inputs", "reference", "tests")
ROOT_FILES = ("run_submission.sh", "README.md", "REPRODUCE.md", "ENVIRONMENT.md",
              "PACKAGING.md", "VERSIONS.md", "BASELINE.md", "复现与提交说明.md",
              ".gitattributes", ".gitignore")
REQUIRED = ("run_submission.sh", "InfiniCore/xmake.lua", "InfiniLM/xmake.lua",
            "scripts/h3_source_stage.py", "scripts/h3_bootstrap.py",
            "scripts/h3_delivery.py", "scripts/h3_source_identity.py",
            "scripts/h3_submission_common.sh", "scripts/h3_submission_record.py",
            "scripts/h3_repro_env.sh", "scripts/reproduce_h3.sh",
            "requirements/h3_linux_cp311.lock.json", "third_party/h3/manifest.json",
            "inputs/first_frame.png", "reference/case.json",
            "reference/reference_fl2va.mp4")
IGNORED_DIRS = {".git", ".xmake", ".venv", ".pytest_cache", "__pycache__"}
KIND = "h3_source_directory"


def parallel_rows(items, action, label):
    workers = max(1, min(32, int(os.environ.get("H3_STAGE_WORKERS", "16"))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = []
        for index, value in enumerate(pool.map(action, items), 1):
            result.append(value)
            if index % 250 == 0 or index == len(items):
                print(label, index, len(items), file=sys.stderr, flush=True)
    return result


def source_rows(root):
    root = Path(root).resolve()
    rows = []
    def describe(path, info):
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("Unsafe source path: " + str(path))
        rows.append(dict(path=path.relative_to(root).as_posix(), size=info.st_size,
                         source_mtime_ns=info.st_mtime_ns))
    for name in ROOT_FILES:
        path = root / name
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        describe(path, info)
    def visit(base, current):
        # Check each directory before entering it, then reuse scandir metadata
        # without resolving every ancestor separately for every source file.
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink():
                    raise RuntimeError("Unsafe source symlink: " + str(path))
                if entry.is_dir(follow_symlinks=False):
                    # Locked wheels go straight to work/downloads, not source/.
                    if path == root / "third_party/offline-wheels":
                        continue
                    if entry.name in IGNORED_DIRS or (current == base and entry.name == "build"):
                        continue
                    visit(base, path)
                elif (not entry.name.endswith((".pyc", ".pyo", ".so", ".o", ".obj", ".pyd", ".a"))
                      and ".so." not in entry.name):
                    describe(path, entry.stat(follow_symlinks=False))
    for name in DIRECTORIES:
        base = root / name
        try:
            info = base.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("Unsafe source directory: " + name)
        visit(base, base)
    rows.sort(key=lambda row: row["path"])
    names = {row["path"] for row in rows}
    for name in REQUIRED:
        if name not in names:
            raise RuntimeError("Missing source file: " + name)
    path = root / "ninetoothed/src"
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError("Missing source directory: ninetoothed/src")
    print("H3_SOURCE_INPUTS_COMPLETE", len(rows), len(rows), file=sys.stderr, flush=True)
    return rows


def git_metadata(root):
    if not (root / ".git").exists():
        return None, None
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0")
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       env=env, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                          "--untracked-files=normal"], env=env, text=True,
                                         stderr=subprocess.DEVNULL)
        return head, status
    except (OSError, subprocess.SubprocessError):
        return None, None


def node_directory(source, explicit=None):
    if explicit is not None:
        return Path(explicit).resolve()
    afs = Path(os.environ["IFAB_AFS_HOME"]).resolve()
    candidates = [Path.cwd()]
    if os.environ.get("OLDPWD"):
        candidates.append(Path(os.environ["OLDPWD"]))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_relative_to(afs):
            continue
        for workspace in (candidate, *candidate.parents):
            if workspace.name == "workspace" and workspace.is_dir():
                parent = workspace.parent
                if parent == Path(parent.anchor) or parent.is_relative_to(afs):
                    continue
                # Hash the path string only to preserve stable node directory names.
                # This never reads or verifies any file contents.
                key = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:10]
                return parent / ("h3-native-" + key)
    raise RuntimeError(
        'Cannot locate node workspace. In a NEW eight-GPU terminal, before changing '
        'directory, run: bash "$IFAB_AFS_HOME/songshanbei-contest/run_submission.sh" '
        '(or set H3_NODE_ROOT to node storage outside AFS/workspace).')


def stage(source, wheels, node=None):
    source, wheels = (Path(p).resolve() for p in (source, wheels))
    node = node_directory(source, node)
    afs = Path(os.environ["IFAB_AFS_HOME"]).resolve()
    if source == afs or not source.is_relative_to(afs):
        raise RuntimeError("Uploaded source must be a project directory inside personal AFS")
    if (node == Path(node.anchor) or node.is_relative_to(afs) or afs.is_relative_to(node)
            or "workspace" in node.parts):
        raise RuntimeError("Use node storage outside AFS and the limited workspace")

    # All offline inputs are checked before creating or changing the node tree.
    with contextlib.redirect_stdout(sys.stderr):
        verify_assets(source)
    locked = json.loads((source / "requirements/h3_linux_cp311.lock.json").read_text())["wheels"]
    wheels = wheel_input(wheels, locked)
    print("H3_OFFLINE_WHEELS_COMPLETE", len(locked), "directory",
          file=sys.stderr, flush=True)
    rows = source_rows(source)

    marker = node / "source-stage.json"
    previous = {}
    previous_checked_ns = 0
    pending = {}
    if node.exists():
        if not marker.is_file():
            raise RuntimeError("Node directory has no source-stage receipt; choose a dedicated directory")
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("kind") != KIND or data.get("source_root") != str(source):
            raise RuntimeError("Node directory belongs to a different source")
        previous = {str(safe_relative(row["path"])): row for row in data["files"]}
        previous_checked_ns = int(data.get("checked_at", 0) * 1_000_000_000)
        pending = {str(safe_relative(row["path"])): row for row in data.get("pending_files", [])
                   if "target_mtime_ns" in row}

    target = node / "source"
    if target.is_symlink():
        raise RuntimeError("Unsafe node source directory")
    current = {row["path"]: row for row in rows}
    # Metadata preserves incremental files without reading their contents.
    # A local edit changes size/mtime and is rejected before any update.
    def check_previous(item):
        name, old = item
        path = target / str(safe_relative(name))
        if not path.resolve().is_relative_to(target) or path.is_symlink():
            raise RuntimeError("Unsafe node source: " + name)
        if path.exists():
            stat = path.stat()
            candidates = [old]
            if name in pending:
                candidates.append(pending[name])
            if not any(stat.st_size == candidate["size"]
                    and ("target_mtime_ns" not in candidate
                         or stat.st_mtime_ns == candidate["target_mtime_ns"])
                    for candidate in candidates):
                raise RuntimeError("Node source has independent edits: " + name)
    parallel_rows(list(previous.items()), check_previous, "H3_NODE_SOURCE_METADATA")
    for name, recovery in pending.items():
        path = target / str(safe_relative(name))
        if path.is_file():
            stat = path.stat()
            if stat.st_size == recovery["size"] and stat.st_mtime_ns == recovery["target_mtime_ns"]:
                previous[name] = recovery
    for name in current.keys() - previous.keys():
        path = target / str(safe_relative(name))
        if path.is_symlink() or not path.resolve().is_relative_to(target):
            raise RuntimeError("Unsafe node source: " + name)
        if path.exists() and path.stat().st_size != current[name]["size"]:
            raise RuntimeError("New source path conflicts with node files: " + name)
    head, status = git_metadata(source)
    receipt = dict(kind=KIND, source_root=str(source), wheelhouse=str(wheels),
                   files=list(previous.values()), pending_files=rows,
                   wheel_count=len(locked), state="preparing")
    node.mkdir(parents=True, exist_ok=True)
    write_json(marker, receipt)
    with contextlib.redirect_stdout(sys.stderr):
        stage_wheels(node / "work", wheels, repo=source)
    def copy_row(row):
        origin = source / row["path"]
        path = target / row["path"]
        old = previous.get(row["path"], {})
        recovery = pending.get(row["path"])
        if recovery and path.is_file():
            stat = path.stat()
            if stat.st_size == recovery["size"] and stat.st_mtime_ns == recovery["target_mtime_ns"]:
                old = recovery
        same_source_time = (old.get("source_mtime_ns") == row["source_mtime_ns"]
            or ("source_mtime_ns" not in old and bool(old)
                and row["source_mtime_ns"] <= previous_checked_ns))
        if not path.exists() or path.stat().st_size != row["size"] or not same_source_time:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".h3-partial")
            shutil.copyfile(origin, partial)
            verify(partial, row)
            partial.chmod(0o755 if path.suffix == ".sh" else 0o644)
            row["target_mtime_ns"] = partial.stat().st_mtime_ns
            return row
        row["target_mtime_ns"] = path.stat().st_mtime_ns
        return None
    staged = parallel_rows(rows, copy_row, "H3_SOURCE_FILES")
    # Persist the exact metadata of every pending rename before committing.
    # A retry can distinguish a completed rename from an independent node edit.
    write_json(marker, dict(receipt, pending_files=rows))
    def promote(row):
        if row is not None:
            path = target / row["path"]
            path.with_name(path.name + ".h3-partial").replace(path)
    parallel_rows(staged, promote, "H3_SOURCE_COMMITTED")
    changed = sum(row is not None for row in staged)
    for name in previous.keys() - current.keys():
        (target / name).unlink(missing_ok=True)
    # Record sizes and timestamps without rereading source contents.
    def verify_row(row):
        origin = (source / row["path"]).stat()
        if origin.st_size != row["size"] or origin.st_mtime_ns != row["source_mtime_ns"]:
            raise RuntimeError("Source metadata changed while staging: " + row["path"])
        verify(target / row["path"], row)
        row["target_mtime_ns"] = (target / row["path"]).stat().st_mtime_ns
    parallel_rows(rows, verify_row, "H3_SOURCE_COPY_COMPLETE")
    write_json(target / ".h3-source.json", dict(schema=1, kind="source_directory_snapshot",
               git_head=head, git_status=status, source_root=str(source), files=rows))
    # Every target member was just checked above; later audits revalidate identity.
    write_json(marker, dict(receipt, files=rows, pending_files=[], state="ready",
                            changed_files=changed, checked_at=time.time()))
    print("H3_SOURCE_STAGE_READY", len(rows), "changed", changed,
          file=sys.stderr, flush=True)
    print(node)
    return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--node", type=Path, help="Optional node directory; default: session workspace sibling")
    args = parser.parse_args()
    stage(args.source, args.wheelhouse, args.node)


if __name__ == "__main__":
    main()
