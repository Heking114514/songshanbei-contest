"""Package offline inputs, stage on node storage, and save bounded outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

from h3_source_identity import digest, source_identity

MiB = 1024 ** 2
LIMIT = 4 * 1024 ** 3


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def file_row(path, name):
    return dict(path=name, size=path.stat().st_size, sha256=digest(path))


def verify(path, row):
    if not path.is_file() or path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
        raise RuntimeError("Missing/corrupt delivery file: " + str(path))


def safe_relative(name):
    p = PurePosixPath(name)
    if not p.parts or p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise RuntimeError("Unsafe delivery path: " + name)
    return p


def package(repo, wheelhouse, output):
    repo, wheelhouse, output = map(lambda p: Path(p).resolve(), (repo, wheelhouse, output))
    if output.exists():
        raise RuntimeError("Delivery output must be new")
    # Git supplies the source boundary, including current nonignored new files.
    raw = subprocess.check_output(["git", "-C", str(repo), "ls-files", "--cached", "--others",
                                   "--exclude-standard", "-z"])
    names = sorted(set(x.decode("utf-8") for x in raw.split(b"\0") if x))
    rows = []
    for name in names:
        if name.startswith("reference/"):
            continue
        safe_relative(name)
        p = repo / name
        if not p.is_file() or p.is_symlink():
            raise RuntimeError("Unsupported/missing source item: " + name)
        rows.append(file_row(p, name))
    wheels = json.loads((repo / "requirements/h3_linux_cp311.lock.json").read_text())["wheels"]
    for row in wheels:
        safe_relative(row["filename"])
        verify(wheelhouse / row["filename"], row)
    for row in json.loads((repo / "third_party/h3/manifest.json").read_text())["files"]:
        verify(repo / row["path"], row)
    identity = source_identity(repo)
    source_manifest = dict(schema=1, git_head=identity["git_head"], git_status=identity["git_status"],
                           created_at=datetime.now(timezone.utc).isoformat(), files=rows)
    output.mkdir(parents=True)
    (output / "scripts").mkdir()
    (output / "reference").mkdir()
    for name in ("run_submission.sh", "REPRODUCE.md",
                 "reference/reference_fl2va.mp4", "reference/case.json", "reference/README.md",
                 "scripts/h3_delivery.py", "scripts/h3_source_identity.py"):
        shutil.copyfile(repo / name, output / name)
    launchers = [file_row(output / name, name) for name in
                 ("run_submission.sh", "reference/reference_fl2va.mp4", "reference/case.json",
                  "scripts/h3_delivery.py", "scripts/h3_source_identity.py")]
    # AFS receives the split archive alone; no unpacked copy or old build cache.
    with tempfile.TemporaryDirectory(prefix="h3-package-") as scratch:
        scratch = Path(scratch)
        identity_file = scratch / ".h3-source.json"
        write_json(identity_file, source_manifest)
        archive = scratch / "payload.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
            for row in rows:
                p = repo / row["path"]
                verify(p, row)
                info = bundle.gettarinfo(str(p), "source/" + row["path"])
                info.mode = 0o755 if row["path"].endswith(".sh") else 0o644
                with p.open("rb") as stream:
                    bundle.addfile(info, stream)
            bundle.add(identity_file, arcname="source/.h3-source.json")
            for row in wheels:
                p = wheelhouse / row["filename"]
                verify(p, row)
                bundle.add(p, arcname="wheelhouse/" + row["filename"], recursive=False)
        for row in rows:
            verify(repo / row["path"], row)
        parts = []
        with archive.open("rb") as stream:
            index = 0
            while block := stream.read(96 * MiB):
                name = "payload.part%03d" % index
                (output / name).write_bytes(block)
                parts.append(file_row(output / name, name))
                index += 1
        manifest = dict(schema=1, archive_sha256=digest(archive), archive_size=archive.stat().st_size,
            parts=parts, launchers=launchers, git_head=identity["git_head"],
            dirty_worktree=bool(identity["git_status"]), source_manifest_sha256=digest(identity_file),
            source_files=len(rows), source_bytes=sum(x["size"] for x in rows),
            wheel_count=len(wheels), wheel_bytes=sum(x["size"] for x in wheels))
        write_json(output / "delivery-manifest.json", manifest)
    lines = ["# AFS 上传清单", "", "上传本目录全部文件，保留分片名称；不要在 AFS 解压。", "",
             "| 文件 | 字节 | SHA256 |", "|---|---:|---|"]
    for row in manifest["launchers"] + manifest["parts"]:
        lines.append("| %s | %s | %s |" % (row["path"], row["size"], row["sha256"]))
    lines += ["", "另传 delivery-manifest.json、REPRODUCE.md、UPLOAD.md。上传后另取容量统计 JSON。",
              "平台自带 Python3.11、Torch2.9.0/HIP6.3、DTK/gfx936；共享模型权重不随包上传。",
              "导出的是当前工作树；git_head 是基准提交，dirty_worktree 如实记录未提交修改。"]
    (output / "UPLOAD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def stage(package_dir, node=None):
    if not node:
        raise RuntimeError("Set H3_NODE_ROOT to node storage outside AFS and workspace; /tmp is not a supported default")
    package_dir = Path(package_dir).resolve()
    manifest = json.loads((package_dir / "delivery-manifest.json").read_text())
    for row in manifest["launchers"]:
        verify(package_dir / str(safe_relative(row["path"])), row)
    afs = Path(os.environ["IFAB_AFS_HOME"]).resolve()
    if not package_dir.is_relative_to(afs):
        raise RuntimeError("Run the uploaded delivery from IFAB_AFS_HOME")
    node = Path(node).resolve()
    if node.is_relative_to(afs) or node == Path(node.anchor) or afs.is_relative_to(node):
        raise RuntimeError("Node staging must be outside AFS")
    target = node / "source"
    if (node / "delivery-manifest.json").exists():
        if json.loads((node / "delivery-manifest.json").read_text()) != manifest:
            raise RuntimeError("Node directory belongs to a different delivery")
        if digest(target / ".h3-source.json") != manifest["source_manifest_sha256"]:
            raise RuntimeError("Source identity SHA mismatch")
        source_identity(target)
        print("H3_DELIVERY_REUSED " + str(node), file=sys.stderr, flush=True)
        print(node)
        return
    if node.exists():
        raise RuntimeError("Node staging path exists without a verified delivery")
    node.mkdir(parents=True)
    archive = node / "payload.tar.gz.partial"
    h = hashlib.sha256()
    with archive.open("wb") as out:
        for row in manifest["parts"]:
            p = package_dir / str(safe_relative(row["path"]))
            verify(p, row)
            print("H3_DELIVERY_PART_SHA_OK " + p.name, file=sys.stderr, flush=True)
            with p.open("rb") as stream:
                for block in iter(lambda: stream.read(MiB), b""):
                    out.write(block)
                    h.update(block)
    if h.hexdigest() != manifest["archive_sha256"] or archive.stat().st_size != manifest["archive_size"]:
        raise RuntimeError("Joined delivery SHA mismatch")
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            rel = safe_relative(member.name)
            if rel.parts[0] not in ("source", "wheelhouse") or not member.isfile():
                raise RuntimeError("Unexpected delivery member: " + member.name)
        bundle.extractall(node, filter="data")
    if digest(target / ".h3-source.json") != manifest["source_manifest_sha256"]:
        raise RuntimeError("Source identity SHA mismatch")
    source_identity(target)
    write_json(node / "delivery-manifest.json", manifest)
    archive.unlink()
    print("H3_DELIVERY_STAGED " + str(node), file=sys.stderr, flush=True)
    print(node)


def storage_snapshot(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    storage = data.get("storage", data)
    usage = storage["usage"]
    sampled = usage["sampled_at"]
    if isinstance(sampled, str):
        try:
            sampled = float(sampled)
        except ValueError:
            sampled = datetime.fromisoformat(sampled.replace("Z", "+00:00")).timestamp()
    if usage.get("complete") is not True or usage.get("over_quota") or not 0 <= time.time() - sampled <= 900:
        raise RuntimeError("AFS API snapshot must be complete, below quota, and at most 15 minutes old")
    return dict(used=int(usage["bytes"]), limit=min(int(storage["bytes"]), LIMIT),
                sampled_at=sampled, pending=sum(int(x["size"]) for x in storage.get("pending_uploads", [])))


def destination():
    afs = Path(os.environ["IFAB_AFS_HOME"]).resolve()
    path = Path(os.environ["H3_AFS_OUTPUT_DIR"]).resolve()
    if path == afs or not path.is_relative_to(afs) or path.exists():
        raise RuntimeError("H3_AFS_OUTPUT_DIR must be a NEW directory inside personal AFS")
    package_dir = Path(os.environ["H3_DELIVERY_DIR"]).resolve()
    if path.is_relative_to(package_dir):
        raise RuntimeError("Keep results outside the uploaded delivery")
    return path


def reserve(run):
    dest = destination()
    baseline = storage_snapshot(os.environ["H3_AFS_INFO_JSON"])
    cap = int(os.environ.get("H3_AFS_OUTPUT_CAP_MIB", "128")) * MiB
    if not MiB <= cap <= 512 * MiB or min(baseline["used"], baseline["pending"]) < 0:
        raise RuntimeError("Invalid AFS budget")
    peak = baseline["used"] + baseline["pending"] + cap + 32 * MiB
    if peak > baseline["limit"]:
        raise RuntimeError("AFS 4 GiB budget exceeded; free space before starting")
    write_json(Path(run) / "afs-reservation.json",
               dict(destination=str(dest), baseline=baseline, cap=cap, peak=peak,
                    note="Bounded reservation from API baseline; coordinate other AFS writers"))
    print("H3_AFS_BUDGET_RESERVED", peak, baseline["limit"], "destination", dest, flush=True)


def persist(run, rc):
    run = Path(run).resolve()
    reservation = json.loads((run / "afs-reservation.json").read_text())
    dest = destination()
    if str(dest) != reservation["destination"]:
        raise RuntimeError("AFS destination changed after reservation")
    files = sorted(p for p in run.rglob("*") if p.is_file())
    if any(p.is_symlink() or not p.resolve().is_relative_to(run) for p in files):
        raise RuntimeError("Unsafe output file")
    rows = [file_row(p, p.relative_to(run).as_posix()) for p in files]
    total = sum(x["size"] for x in rows)
    if total + MiB > reservation["cap"]:
        raise RuntimeError("Output exceeds reserved AFS budget; retained on node: " + str(run))
    dest.mkdir(parents=True, exist_ok=False)
    for row in rows:
        target = dest / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        shutil.copyfile(run / row["path"], partial)
        verify(partial, row)
        partial.replace(target)
    write_json(dest / "saved-manifest.json", dict(state="saved", pipeline_rc=rc,
               saved_at=datetime.now(timezone.utc).isoformat(), files=rows, bytes=total))
    print("H3_AFS_SAVED", dest, "pipeline_rc", rc, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("package")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--wheelhouse", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("stage")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--node", type=Path, required=True)
    for name in ("reserve", "persist"):
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path, required=True)
        if name == "persist":
            p.add_argument("--rc", type=int, required=True)
    a = parser.parse_args()
    if a.command == "package":
        package(a.repo, a.wheelhouse, a.output)
    elif a.command == "stage":
        stage(a.package, a.node)
    elif a.command == "reserve":
        reserve(a.run)
    elif a.command == "persist":
        persist(a.run, a.rc)


if __name__ == "__main__":
    main()
