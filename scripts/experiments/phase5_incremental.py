"""Node-local phase45 source overlay and object guard preparation.

Requires an already restored, zero-change-checked phase23 workspace. Does not
allocate, restore environments, configure xmake, build, install, or access AFS.
Use the existing isolated xmake compile guard with the emitted allowlist.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("overlay", "verify"))
    parser.add_argument("--live-root", type=Path, required=True, help="Restored private src directory")
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="phase45-combined-manifest.json")
    parser.add_argument("--state-dir", type=Path, required=True, help="Node-private audit directory")
    args = parser.parse_args()
    live, candidate, state = (p.resolve() for p in (args.live_root, args.candidate_root, args.state_dir))
    assert live != candidate and live.is_dir() and candidate.is_dir()
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["files"]
    state.mkdir(parents=True, exist_ok=True)

    def emit(event, **data):
        print("PHASE5_INCREMENTAL", event, json.dumps(data), flush=True)

    def write(name, data):
        (state / name).write_text(json.dumps(data, indent=2) + "\n")

    def safe_path(root, relative):
        path = root / relative
        assert path.resolve().is_relative_to(root), relative
        assert not Path(relative).is_absolute(), relative
        return path

    def snapshot():
        objects = [p for component in ("InfiniCore", "InfiniLM")
                   for p in (live / component / "build/.objs").rglob("*.o")]
        result = {}
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(sha, p): p for p in objects}
            for n, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                result[path.relative_to(live).as_posix()] = {
                    "sha256": future.result(), "mtime_ns": path.stat().st_mtime_ns}
                if n % 128 == 0 or n == len(objects):
                    emit("HASH_PROGRESS", done=n, total=len(objects))
        return result

    def object_source(key):
        component, rest = key.split("/", 1)
        assert "/release/" in rest and rest.endswith(".o"), key
        return component, rest.split("/release/", 1)[1][:-2]

    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    if args.mode == "overlay":
        assert not (state / "objects-before.json").exists(), "Use a new state directory"
        # Validate the entire overlay before modifying any source.
        for row in rows:
            source = safe_path(candidate, row["path"])
            target = safe_path(live, row["path"])
            assert source.is_file() and sha(source) == row["sha256"], row["path"]
            expected = row["before_sha256"]
            if expected is None:
                assert not target.exists(), ("unexpected preexisting new file", row["path"])
            else:
                assert target.is_file() and sha(target) == expected, ("base mismatch", row["path"])
        before = snapshot()
        assert len(before) >= 1194, ("expected validated phase23 cache", len(before))
        write("objects-before.json", before)
        changed = [row for row in rows if row["change"] != "unchanged"]
        allowed = {("InfiniCore", row["path"].split("/", 1)[1]) for row in changed
                   if row["path"].startswith("InfiniCore/") and row["path"].endswith((".cc", ".cu", ".cpp"))}
        allowed |= {("InfiniLM", row["path"].split("/", 1)[1]) for row in changed
                    if row["path"].startswith("InfiniLM/") and row["path"].endswith((".cc", ".cu", ".cpp"))}
        headers = [(row["path"].split("/", 1)[0], row["path"].split("/", 1)[1])
                   for row in changed if row["path"].endswith((".h", ".hpp", ".cuh"))]
        evidence = {}
        for component in ("InfiniCore", "InfiniLM"):
            for dep in (live / component / "build/.deps").rglob("*.o.d"):
                # xmake's Lua dependency files contain quoted relative/absolute
                # paths. Match full component-relative header paths, not basenames.
                text = dep.read_text(errors="replace").replace("\\\\", "/").replace("\\", "/")
                hits = [comp + "/" + header for comp, header in headers
                        if re.search(r"(?:^|[/\"'])" + re.escape(header) + r"(?=[\"'\s,]|$)", text)]
                if hits:
                    key = component + "/" + dep.relative_to(live / component / "build/.deps").as_posix()
                    source = object_source(key.removesuffix(".d"))
                    allowed.add(source)
                    evidence[component + "/" + source[1]] = hits
        for row in changed:
            target = safe_path(live, row["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(safe_path(candidate, row["path"]).read_bytes())
            emit("OVERLAY", path=row["path"])
        # Do not write unchanged infiniop.h: preserve its cache-relevant mtime.
        raw_allowed = sorted({source for _, source in allowed})
        (state / "allow-candidate.txt").write_text("\n".join(raw_allowed) + "\n")
        write("build-plan.json", {
            "created_at": now, "manifest_sha256": sha(args.manifest),
            "objects_before": len(before), "allowed": sorted([list(x) for x in allowed]),
            "header_dependencies": evidence, "overlaid": [row["path"] for row in changed],
            "untouched": [row["path"] for row in rows if row["change"] == "unchanged"]})
        emit("GUARD_READY", objects=len(before), source_allowlist=len(allowed),
             allow_file=str(state / "allow-candidate.txt"))
    else:
        plan = json.loads((state / "build-plan.json").read_text())
        assert plan["manifest_sha256"] == sha(args.manifest), "Manifest changed during build"
        for row in rows:
            assert sha(safe_path(live, row["path"])) == row["sha256"], row["path"]
        before = json.loads((state / "objects-before.json").read_text())
        after = snapshot()
        assert set(before) <= set(after), "Build removed cached objects"
        allowed = {tuple(x) for x in plan["allowed"]}
        changed = [key for key in after if after[key] != before.get(key)]
        assert all(object_source(key) in allowed for key in changed), changed
        new_units = {tuple(x) for x in plan["allowed"] if x[1].startswith((
            "src/infiniop/ops/h3_", "src/infinicore/ops/h3_"))}
        actual_units = {object_source(key) for key in after}
        assert new_units <= actual_units, ("Missing new xmake source units", sorted(new_units - actual_units))
        write("incremental-result.json", {"verified_at": now, "objects_before": len(before),
              "objects_after": len(after), "changed": changed,
              "unaffected_preserved": len(before) - sum(k in before for k in changed),
              "scope": "objects/source only; install/library symbols and GPU probes still required"})
        emit("VERIFIED", before=len(before), after=len(after), changed=len(changed))


if __name__ == "__main__":
    main()
