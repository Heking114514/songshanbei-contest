"""Verify and deploy the pinned best H3 attention bundle for formal runs.

The archive is a fixed delivery asset. Validation happens before extraction and
every extracted file is checked against the bundle manifest before publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "third_party/h3/phase10-best-bundle.tar.gz"
ARCHIVE_SHA256 = "a9621f048e1bdcd692d4e41c7604d3fc5ba35ad1c4be14dd1f78a6b2444a4443"
MANIFEST_SHA256 = "d167348cc76e845439c554c007d48154b34414cec9769b9e82547cf2f187b120"
DISPATCHER_NAME = "libh3_flash_dispatch.so"
DISPATCHER_SHA256 = "5a5e40df8c40d02460f9f478387c693bac69a64863b20ded9d0f5e92965d2230"
ABI_NAME = "h3_flash_abi.h"
ABI_SHA256 = "212bdebc17b24ef96ee8e3ae2de6c81203a59ed26b07dc1997d20707ffdc1635"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name: str) -> bool:
    path = Path(name)
    return path.name == name and name not in ("", ".", "..") and "\\" not in name


def verify_archive(path: Path = ARCHIVE):
    if not path.is_file() or sha256(path) != ARCHIVE_SHA256:
        raise RuntimeError("Pinned H3 attention bundle archive changed")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or any(
                not member.isfile() or not safe_name(member.name) for member in members):
            raise RuntimeError("Unsafe or duplicate H3 attention bundle member")
        by_name = {member.name: member for member in members}
        if "bundle.json" not in by_name:
            raise RuntimeError("H3 attention bundle manifest is missing")
        raw_manifest = archive.extractfile(by_name["bundle.json"]).read()
        if hashlib.sha256(raw_manifest).hexdigest() != MANIFEST_SHA256:
            raise RuntimeError("H3 attention bundle manifest identity changed")
        manifest = json.loads(raw_manifest)
        files = manifest.get("files")
        if manifest.get("schema") != 1 or not isinstance(files, dict):
            raise RuntimeError("Invalid H3 attention bundle manifest")
        if set(names) != set(files) | {"bundle.json"}:
            raise RuntimeError("H3 attention bundle member set changed")
        if files.get(DISPATCHER_NAME) != DISPATCHER_SHA256 or files.get(ABI_NAME) != ABI_SHA256:
            raise RuntimeError("Unexpected H3 attention dispatcher or ABI")
        for member in members:
            if member.name == "bundle.json":
                continue
            digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            if files.get(member.name) != digest:
                raise RuntimeError("H3 attention bundle member changed: " + member.name)
    return manifest


def deploy(archive: Path, output: Path):
    output = output.resolve()
    afs = os.environ.get("IFAB_AFS_HOME")
    if afs and output.is_relative_to(Path(afs).resolve()):
        raise RuntimeError("H3 attention deployment must stay outside AFS")
    if output.exists():
        raise RuntimeError("H3 attention deployment destination must be new")
    manifest = verify_archive(archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".h3-attention-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(temporary, filter="data")
        for name, digest in manifest["files"].items():
            path = temporary / name
            if not path.is_file() or path.is_symlink() or sha256(path) != digest:
                raise RuntimeError("Deployed H3 attention file changed: " + name)
        (temporary / "bundle.json").chmod(0o644)
        for path in temporary.iterdir():
            path.chmod(0o755 if path.suffix == ".so" else 0o644)
        temporary.replace(output)
    receipt = dict(
        schema=1,
        archive=str(archive.resolve()),
        archive_sha256=ARCHIVE_SHA256,
        bundle_json_sha256=MANIFEST_SHA256,
        dispatcher_sha256=DISPATCHER_SHA256,
        abi_sha256=ABI_SHA256,
        files=manifest["files"],
        path=str(output),
        model_acceptance=False,
    )
    receipt_path = output / "deployment.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("H3_ATTENTION_BUNDLE_READY", output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "deploy"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        verify_archive(args.archive.resolve())
        print("H3_ATTENTION_BUNDLE_VERIFIED", flush=True)
        return
    if args.output is None:
        parser.error("deploy requires --output")
    deploy(args.archive.resolve(), args.output)


if __name__ == "__main__":
    main()
