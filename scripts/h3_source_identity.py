"""Identify a real checkout or a size-checked exported working tree."""
import json
from pathlib import Path
import subprocess


def source_identity(root):
    root = Path(root).resolve()
    manifest = root / ".h3-source.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for row in data["files"]:
            p = root / row["path"]
            if not p.resolve().is_relative_to(root) or not p.is_file():
                raise RuntimeError("Exported source missing/unsafe: " + row["path"])
            if p.stat().st_size != row["size"]:
                raise RuntimeError("Exported source changed: " + row["path"])
        return dict(kind=data.get("kind", "exported_worktree"), git_head=data["git_head"],
                    git_status=data["git_status"], integrity="presence_and_size",
                    file_count=len(data["files"]))
    return dict(kind="git_checkout",
        git_head=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True))
