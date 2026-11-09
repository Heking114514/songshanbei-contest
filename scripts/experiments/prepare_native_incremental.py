"""Clone verified sources/cache onto node disk and forbid unintended compilation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

work = Path(os.environ["WORK"])
stable = Path(os.environ["IFAB_AFS_HOME"]) / "handover/h3-stable"
root = stable.parent
project = work / "src/InfiniCore"
mode = sys.argv[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def snapshot():
    return {str(p.relative_to(project)): [p.stat().st_mtime_ns, sha(p)]
            for p in (project/"build/.objs").rglob("*.o")}


if mode == "prepare":
    project.parent.mkdir()
    # Old-lease .xmake files can be mode 0600; restore them from the
    # verified archive with current ownership instead of reading them in AFS.
    futures = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        def enqueue(source, destination):
            futures.append(pool.submit(shutil.copy2, source, destination))
            return destination
        shutil.copytree(stable/"src/InfiniCore", project, symlinks=True,
                        ignore=shutil.ignore_patterns("build", ".git", ".xmake", "__pycache__"),
                        copy_function=enqueue)
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 500 == 0:
                print("PHASE1_SOURCE_COPY", index, len(futures), flush=True)
    print("PHASE1_SOURCE_COPY_COMPLETE", len(futures), flush=True)
    cache = root/"current-batch/h3-stable-build-cache.tar.gz"
    manifest = json.loads((root/"current-batch/h3-stable-cache.json").read_text())
    assert sha(cache) == manifest["sha256"], "Cache SHA mismatch"
    restored = work/"restored"
    print("PHASE1_CACHE_RESTORE", cache.name, manifest["sha256"], flush=True)
    with tarfile.open(cache) as tf:
        # Restore only Core objects/configuration; LM does not need rebuilding.
        tf.extractall(restored, members=[x for x in tf.getmembers()
                                        if x.name.startswith("InfiniCore/")],
                      filter="data")
    shutil.copytree(restored/"InfiniCore/.xmake", project/".xmake")
    build = restored/"InfiniCore/build"
    assert build.is_dir()
    (project/"build").symlink_to(build)
    # Node migration changes source absolute paths, never toolchain/ABI/options.
    old = str(stable/"src/InfiniCore").encode()
    new = str(project).encode()
    changed = []
    for base in [project/".xmake", build/".deps"]:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and not p.is_symlink():
                data = p.read_bytes()
                if old not in data:
                    continue
                assert bytes([0]) not in data, "Unexpected binary dependency record"
                stat = p.stat()
                p.write_bytes(data.replace(old, new))
                os.utime(p, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                changed.append(str(p.relative_to(base)))
    data = snapshot()
    assert len(data) > 400, len(data)
    (work/"objects-restored.json").write_text(json.dumps(data, sort_keys=True))
    print("PHASE1_NODE_CLONE", "objects", len(data), "rebased_text_records", len(changed), flush=True)
elif mode == "guard":
    p = Path(os.environ["XMAKE_PROGRAM_DIR"])/"modules/private/action/build/object.lua"
    s = p.read_text()
    anchor = ("    if not dryrun and not depend.is_changed(dependinfo, "
              "{lastmtime = lastmtime, values = depvalues, timecache = true}) then\\n"
              "        return\\n    end\\n").replace("\\n","\n")
    assert s.count(anchor) == 1
    if "H3_ALLOWED_SOURCE" not in s:
        guard = """
    local allowed = os.getenv("H3_ALLOWED_SOURCE")
    if allowed and path.translate(sourcefile) ~= path.translate(allowed) then
        print("PHASE1_BLOCKED_SOURCE " .. sourcefile)
        print("PHASE1_BLOCKED_DEPEND_FILE " .. dependfile)
        raise("PHASE1_UNEXPECTED_COMPILE: " .. sourcefile)
    end
"""
        p.write_text(s.replace(anchor, anchor+guard))
    print("PHASE1_BUILD_GUARD_READY", flush=True)
elif mode == "noop":
    before = json.loads((work/"objects-restored.json").read_text())
    after = snapshot()
    assert before == after, "Zero-change objects mutated"
    print("PHASE1_NOOP_ZERO_COMPILES", len(after), flush=True)
elif mode == "patch":
    p = project/"src/infinicore/ops/multi_head_attention/hygon/mha_flashattn_hygon.cc"
    (work/"original-mha.cc").write_bytes(p.read_bytes())
    shutil.copyfile(work/"mha_flashattn_hygon.cc", p)
    print("PHASE1_NATIVE_SOURCE", sha(p), flush=True)
elif mode == "verify":
    before = json.loads((work/"objects-restored.json").read_text())
    after = snapshot()
    changed = [p for p in set(before)|set(after) if before.get(p) != after.get(p)]
    assert len(changed) == 1 and changed[0].endswith("/mha_flashattn_hygon.cc.o"), changed
    report = dict(changed_objects=changed, baseline_objects=len(before), after_objects=len(after))
    (work/"incremental-result.json").write_text(json.dumps(report, indent=2))
    print("PHASE1_INCREMENTAL_VERIFIED", json.dumps(report), flush=True)
elif mode == "diagnostic":
    print("PHASE1_INCREMENTAL_DIAGNOSTIC", flush=True)
    for p in (project/".xmake").rglob("config"):
        print("CONFIG", str(p), p.read_text(errors="replace")[:12000], flush=True)
    p = project/"build/.deps/infinicore_cpp_api/linux/x86_64/release"
    shown = 0
    for q in p.rglob("*.d"):
        data = q.read_text(errors="replace")
        if shown == 0 or "mha_flashattn_hygon" in q.name:
            print("DEPEND", str(q), data[:14000], flush=True)
            shown += 1
        if shown >= 2:
            break
else:
    raise ValueError(mode)
