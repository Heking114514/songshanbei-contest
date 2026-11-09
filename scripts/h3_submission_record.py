"""Record submission inputs, official comparison text and output fingerprints."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
from h3_source_identity import source_identity


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")

def parse_report(text):
    final = text.rsplit("[result] MiniMax-H3 video comparison", 1)
    if len(final) != 2:
        raise ValueError("Official final comparison report is missing")
    text = final[1]
    size = re.search(r"^\s+Size:\s+(PASS|FAIL)\b", text, re.M)
    length = re.search(r"^\s+Length:\s+(PASS|FAIL)\b", text, re.M)
    ssim = re.search(r"^\s+Average SSIM:\s+(-?\d+(?:\.\d+)?)%", text, re.M)
    if not re.search(r"^\s+FL2VA\s*$", text, re.M) or not all((size, length, ssim)):
        raise ValueError("Official FL2VA Size/Length/SSIM result is incomplete")
    return dict(size_pass=size[1] == "PASS", length_pass=length[1] == "PASS",
                average_ssim_percent=float(ssim[1]), ssim_threshold=None)

def verify_reference():
    directory = Path(os.environ["H3_DELIVERY_DIR"]) / "reference"
    case = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    expected = dict(first_frame_bytes=fingerprint(os.environ["FIRST_FRAME"])["bytes"],
                    prompt="A dog falls asleep", seed=42, requested_steps=50, num_frames=124)
    for key, value in expected.items():
        if case.get(key) != value:
            raise RuntimeError("Reference belongs to a different test case: " + key)
    reference = fingerprint(directory / "reference_fl2va.mp4")
    if reference["bytes"] != case["reference_bytes"]:
        raise RuntimeError("Bundled reference size mismatch")
    return dict(video=reference, case=case)

def attention_runtime():
    library = Path(os.environ["H3_FLASH_LIBRARY"]).resolve(strict=True)
    bundle = Path(os.environ["H3_ATTENTION_BUNDLE_DIR"]).resolve(strict=True)
    if not library.is_relative_to(bundle):
        raise RuntimeError("H3 attention dispatcher escaped the deployed bundle")
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    expected = manifest["files"].get("libh3_flash_dispatch.so")
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    if expected != digest:
        raise RuntimeError("H3 attention dispatcher changed after deployment")
    flags = {key: value for key, value in os.environ.items()
             if key.startswith("H3_EXPERIMENT_") or key == "H3_FLASH_LIBRARY"}
    return dict(backend="triton_aot_dispatch", dispatcher=fingerprint(library),
                bundle=fingerprint(bundle / "bundle.json"), dispatcher_sha256=digest,
                abi=fingerprint(bundle / "h3_flash_abi.h"), flags=flags)


def begin():
    root = Path(os.environ["H3_SOURCE_ROOT"]).resolve()
    image = fingerprint(os.environ["FIRST_FRAME"])
    compare = root / "InfiniLM/examples/minimax_h3_compare.py"
    if not compare.is_file() or compare.stat().st_size == 0:
        raise RuntimeError("Missing or empty official comparator: " + str(compare))
    model = Path(os.environ["MODEL_PATH"]).resolve()
    attention = attention_runtime()
    result = dict(started_at=now(), mode=os.environ["H3_SUBMISSION_MODE"], first_frame=image,
        prompt="A dog falls asleep", seed=42, requested_steps=50, num_frames=124,
        full_model=True, smoke_executed=False, preliminary_warmup_executed=False,
        first_use_jit_allowed=True, attention=attention, reference_max_heads=4,
        source_root=str(root), model_path=str(model),
        model_configs=[fingerprint(model / name) for name in (
            "diffusers/model_index.json", "diffusers/transformer/config.json",
            "diffusers/text_encoder/config.json", "FL2VA/model_index.json")
            if (model / name).is_file()],
        source_identity=source_identity(root), integrity="presence_size_and_media_decode",
        provided_reference=verify_reference() if os.environ["H3_SUBMISSION_MODE"] == "check" else None,
        native_path=os.environ["H3_NATIVE_VIDEO"], compare_dir=os.environ["OUTPUT_DIR"],
        source_files=[fingerprint(root / name) for name in (
            "run_submission.sh", "scripts/h3_source_stage.py", "scripts/h3_submission_common.sh",
            "scripts/run_h3_compare.sh", "scripts/h3_preflight.py",
            "scripts/h3_submission_record.py", "scripts/h3_delivery.py",
            "scripts/h3_compare_hook/sitecustomize.py", "scripts/h3_reference_sdpa.py",
            "InfiniLM/examples/minimax_h3.py")])
    write_json(os.environ["H3_SUBMISSION_INVOCATION"], result)
    print("H3_SUBMISSION_INVOCATION", json.dumps(result, ensure_ascii=False), flush=True)

def video_metadata(path):
    import av
    with av.open(str(path)) as container:
        fps = str(container.streams.video[0].average_rate)
        audio_streams = len(container.streams.audio)
        shapes = set()
        frames = 0
        for frame in container.decode(video=0):
            shapes.add((frame.width, frame.height))
            frames += 1
    if frames != 124 or len(shapes) != 1:
        raise RuntimeError("Formal video must decode to 124 frames at one resolution: " + str(path))
    if fps != "24" or audio_streams != 1:
        raise RuntimeError("Formal video must have 24 fps and one audio stream: " + str(path))
    width, height = shapes.pop()
    if width % 32 or height % 32:
        raise RuntimeError("Formal video dimensions must be divisible by 32")
    return dict(frames=frames, width=width, height=height, fps=fps, audio_streams=audio_streams)

def invocation():
    return json.loads(Path(os.environ["H3_SUBMISSION_INVOCATION"]).read_text(encoding="utf-8"))

def check_input(data):
    if fingerprint(data["first_frame"]["path"]) != data["first_frame"]:
        raise RuntimeError("Input changed during this run")

def native_check():
    data = invocation()
    check_input(data)
    media = video_metadata(data["native_path"])
    result = dict(state="complete", at=now(), video=fingerprint(data["native_path"]), media=media)
    write_json(Path(os.environ["H3_RUN_DIR"]) / "native-check.json", result)
    print("H3_NATIVE_CHECK", json.dumps(result, ensure_ascii=False), flush=True)

def check():
    data = invocation()
    check_input(data)
    run = Path(os.environ["H3_RUN_DIR"])
    native = json.loads((run / "native-check.json").read_text(encoding="utf-8"))
    if fingerprint(data["native_path"]) != native["video"]:
        raise RuntimeError("Native video changed after generation")
    output = Path(data["compare_dir"])
    report = parse_report((output / "compare.log").read_text(encoding="utf-8"))
    if not report["size_pass"] or not report["length_pass"]:
        raise RuntimeError("Official Size/Length check failed")
    reference = Path(data["provided_reference"]["video"]["path"])
    if fingerprint(reference) != data["provided_reference"]["video"]:
        raise RuntimeError("Reference changed during comparison")
    ref_media = video_metadata(reference)
    if any(ref_media[k] != native["media"][k] for k in ("frames", "width", "height", "fps")):
        raise RuntimeError("Decoded video specifications differ")
    result = dict(state="complete", checked_at=now(), official_result=report,
        native=native, reference=dict(fingerprint(reference), media=ref_media),
        generated_native_this_workflow=True, generated_reference_this_workflow=False,
        official_native_provided=True, official_same_run_speedup_available=False)
    write_json(run / "evaluation.json", result)
    print("H3_FORMAL_CHECK", json.dumps(result, ensure_ascii=False), flush=True)

def status(rc):
    run = Path(os.environ["H3_RUN_DIR"])
    required = "evaluation.json" if os.environ["H3_SUBMISSION_MODE"] == "check" else "native-check.json"
    if rc == 0 and not (run / required).is_file():
        rc = 1
    write_json(run / "submission-status.json", dict(
        state="complete" if rc == 0 else "failed", pipeline_rc=rc, at=now(),
        mode=os.environ["H3_SUBMISSION_MODE"], afs_destination=os.environ["H3_AFS_OUTPUT_DIR"],
        note="AFS persistence is confirmed separately by saved-manifest.json"))
    return rc

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("begin", "native-check", "check", "status", "verify-reference"))
    parser.add_argument("--rc", type=int, default=0)
    args = parser.parse_args()
    if args.command == "begin": begin()
    elif args.command == "native-check": native_check()
    elif args.command == "check": check()
    elif args.command == "verify-reference": verify_reference()
    elif args.command == "status": return status(args.rc)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
