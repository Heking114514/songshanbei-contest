"""Preserve only the experimental incremental delta, never replace stable caches."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from h3_afs_budget import budget

work = Path(os.environ["WORK"])
project = work/"src/InfiniCore"
archive = work/"native-incremental-delta.tar.gz"
result = json.loads((work/"incremental-result.json").read_text())
assert len(result["changed_objects"]) == 1
members = [project/p for p in result["changed_objects"]]
members += [project/"build/linux/x86_64/release/libinfinicore_cpp_api.so"]
deps = project/"build/.deps/infinicore_cpp_api"
members += [p for p in deps.rglob("*") if p.is_file() and
            ("mha_flashattn_hygon" in p.name or "libinfinicore_cpp_api" in p.name)]
members += [project/".xmake"]
with tarfile.open(archive, "w:gz", compresslevel=6) as tf:
    for p in members:
        tf.add(p.resolve(), arcname="InfiniCore/"+str(p.relative_to(project)))
    for name in ["incremental-result.json", "native-results.json", "noop.log", "build.log", "native.log"]:
        p = work/name
        if p.is_file():
            tf.add(p, arcname="evidence/"+name)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*1024**2), b""):
            h.update(b)
    return h.hexdigest()

size = archive.stat().st_size
assert size < 100*1024**2, "Unexpectedly large delta; retain on node for selected download"
manifest = dict(size=size, sha256=sha(archive), source_project=str(project),
                source_sha256=sha(work/"mha_flashattn_hygon.cc"),
                base_cache=json.loads((Path(os.environ["ROOT"])/"current-batch/h3-stable-cache.json").read_text()),
                restore_note="Overlay onto the verified base Core cache, rebase source paths and verify zero-compiles before use.",
                native_probe_complete=False, created_at=time.time(), **result)
p=work/"native-results.json"
if p.exists():
    manifest["native_probe_complete"] = json.loads(p.read_text()).get("complete", False)
budget(size+1024**2,reserve=True,reason="experimental native incremental delta and manifest")
dest=Path(os.environ["ROOT"])/"experiments"/os.environ["PHASE1_SAVE_NAME"]
dest.mkdir(parents=True, exist_ok=False)
target=dest/archive.name
shutil.copyfile(archive,target)
assert sha(target)==manifest["sha256"]
(dest/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
(work/"delta-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("PHASE1_DELTA_SAVED",json.dumps(dict(path=str(target),**manifest)),flush=True)
