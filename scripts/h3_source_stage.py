"""Stage uploaded source and locked wheels for node-local H3 execution."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from h3_bootstrap import verify_assets
from h3_delivery import file_row, safe_relative, verify, write_json
from h3_source_identity import digest, source_identity

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


def source_rows(root):
    for name in REQUIRED:
        if not (root / name).is_file():
            raise RuntimeError("Missing source file: " + name)
    if not (root / "ninetoothed/src").is_dir():
        raise RuntimeError("Missing source directory: ninetoothed/src")
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in DIRECTORIES:
        base = root / name
        if not base.exists():
            continue
        if base.is_symlink():
            raise RuntimeError("Source symlinks are not supported: " + name)
        for current, dirs, names in os.walk(base, followlinks=False):
            current = Path(current)
            kept = []
            for child in dirs:
                path = current / child
                if child in IGNORED_DIRS or (current == base and child == "build"):
                    continue
                if path.is_symlink():
                    raise RuntimeError("Source symlinks are not supported: " + str(path))
                kept.append(child)
            dirs[:] = kept
            for child in names:
                if child.endswith((".pyc", ".pyo", ".so", ".o", ".obj", ".pyd", ".a")) or ".so." in child:
                    continue
                files.append(current / child)
    rows = []
    for index, path in enumerate(sorted(files), 1):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RuntimeError("Unsafe source path: " + str(path))
        rows.append(file_row(path, path.relative_to(root).as_posix()))
        if index % 250 == 0 or index == len(files):
            print("H3_SOURCE_INPUTS_VERIFIED", index, len(files), file=sys.stderr, flush=True)
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


def stage(source, wheels, node):
    source, wheels, node = (Path(p).resolve() for p in (source, wheels, node))
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
    for row in locked:
        if Path(row["filename"]).name != row["filename"]:
            raise RuntimeError("Unsafe wheel filename")
        verify(wheels / row["filename"], row)
    print("H3_OFFLINE_WHEELS_VERIFIED", len(locked), file=sys.stderr, flush=True)
    rows = source_rows(source)

    marker = node / "source-stage.json"
    previous = {}
    if node.exists():
        if not marker.is_file():
            raise RuntimeError("Node directory has no source-stage receipt; choose a dedicated directory")
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("kind") != KIND or data.get("source_root") != str(source):
            raise RuntimeError("Node directory belongs to a different source")
        previous = {str(safe_relative(row["path"])): row for row in data["files"]}

    target = node / "source"
    if target.is_symlink():
        raise RuntimeError("Unsafe node source directory")
    current = {row["path"]: row for row in rows}
    # Refuse to overwrite independent edits in an existing node checkout.
    for name, old in previous.items():
        path = target / str(safe_relative(name))
        if not path.resolve().is_relative_to(target) or path.is_symlink():
            raise RuntimeError("Unsafe node source: " + name)
        if path.exists() and digest(path) not in {
                old["sha256"], current.get(name, old)["sha256"]}:
            raise RuntimeError("Node source has independent edits: " + name)
    if target.exists():
        for name in current.keys() - previous.keys():
            path = target / str(safe_relative(name))
            if (not path.resolve().is_relative_to(target) or path.is_symlink()
                    or (path.exists() and digest(path) != current[name]["sha256"])):
                raise RuntimeError("New source path conflicts with node files: " + name)
    head, status = git_metadata(source)
    receipt = dict(kind=KIND, source_root=str(source), wheelhouse=str(wheels),
                   files=list(previous.values()), pending_files=rows,
                   wheel_count=len(locked), state="preparing")
    node.mkdir(parents=True, exist_ok=True)
    write_json(marker, receipt)
    changed = 0
    for index, row in enumerate(rows, 1):
        origin = source / row["path"]
        path = target / row["path"]
        if not path.exists() or digest(path) != row["sha256"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".h3-partial")
            shutil.copyfile(origin, partial)
            verify(partial, row)
            partial.chmod(0o755 if path.suffix == ".sh" else 0o644)
            partial.replace(path)
            changed += 1
        if index % 250 == 0 or index == len(rows):
            print("H3_SOURCE_FILES", index, len(rows), "changed", changed,
                  file=sys.stderr, flush=True)
    for name in previous.keys() - current.keys():
        (target / name).unlink(missing_ok=True)
    # Bind the snapshot to the uploaded bytes and catch concurrent source edits.
    for index, row in enumerate(rows, 1):
        verify(source / row["path"], row)
        verify(target / row["path"], row)
        if index % 250 == 0 or index == len(rows):
            print("H3_SOURCE_COPY_VERIFIED", index, len(rows), file=sys.stderr, flush=True)
    write_json(target / ".h3-source.json", dict(schema=1, kind="source_directory_snapshot",
               git_head=head, git_status=status, source_root=str(source), files=rows))
    source_identity(target)
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
    parser.add_argument("--node", type=Path, required=True)
    args = parser.parse_args()
    stage(args.source, args.wheelhouse, args.node)


if __name__ == "__main__":
    main()
