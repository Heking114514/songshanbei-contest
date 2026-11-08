"""Identify a real checkout or a SHA-verified exported working tree."""
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_identity(root):
    root = Path(root).resolve()
    manifest = root / ".h3-source.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for row in data["files"]:
            p = root / row["path"]
            if not p.resolve().is_relative_to(root) or not p.is_file():
                raise RuntimeError("Exported source missing/unsafe: " + row["path"])
            if p.stat().st_size != row["size"] or digest(p) != row["sha256"]:
                raise RuntimeError("Exported source changed: " + row["path"])
        return dict(kind="exported_worktree", git_head=data["git_head"],
                    git_status=data["git_status"], manifest_sha256=digest(manifest),
                    file_count=len(data["files"]))
    return dict(kind="git_checkout",
        git_head=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True))
