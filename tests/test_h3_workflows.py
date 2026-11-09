"""CPU-only behavioral checks for developer/submission entry separation."""
import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import h3_bootstrap as bootstrap
import h3_submission_record as record

import time
import h3_delivery as delivery
from h3_source_identity import source_identity
class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3-workflow-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "repo"
        self.tools = self.repo / "scripts"
        self.tools.mkdir(parents=True)
        self.trace = self.root / "trace.jsonl"


    def test_offline_corrupt_wheel_rejected_before_creating_work(self):
        (self.repo / "requirements").mkdir()
        row = dict(filename="a.whl", size=4, sha256=hashlib.sha256(b"good").hexdigest())
        (self.repo / "requirements/h3_linux_cp311.lock.json").write_text(json.dumps({"wheels": [row]}))
        wheels = self.root / "wheels";wheels.mkdir();(wheels / "a.whl").write_bytes(b"bad")
        work = self.root / "work"
        with patch.object(bootstrap, "REPO", self.repo), patch.dict(os.environ, {"IFAB_AFS_HOME": str(self.root / "afs")}):
            with self.assertRaisesRegex(RuntimeError, "wrong-size offline wheel"):
                bootstrap.stage_wheels(work, wheels)
        self.assertFalse(work.exists())


    def test_report_does_not_treat_size_fail_or_missing_ssim_as_success(self):
        text = "[result] MiniMax-H3 video comparison\n  FL2VA\n    Size: PASS\n    Length: PASS\n    Average SSIM: 94.277%\n"
        self.assertTrue(record.parse_report(text)["size_pass"])
        self.assertFalse(record.parse_report(text.replace("Size: PASS", "Size: FAIL"))["size_pass"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            record.parse_report(text.replace("94.277%", "n/a"))


    def test_formal_gpu_inventory_never_launches_compute_probe(self):
        import contextlib
        import types
        from unittest.mock import Mock
        import h3_preflight
        hip = types.SimpleNamespace(hipGetDeviceCount=Mock(), hipGetErrorString=Mock(return_value=b"ok"))
        def count(pointer):
            pointer._obj.value = 8
            return 0
        hip.hipGetDeviceCount.side_effect = count
        torch = types.SimpleNamespace(version=types.SimpleNamespace(hip="6.3"),
            __version__="2.9.0", ones=Mock(side_effect=AssertionError("unexpected warmup")),
            cuda=types.SimpleNamespace(device_count=lambda: 8,
                device=lambda i: contextlib.nullcontext(), mem_get_info=lambda i: (100, 200),
                get_device_name=lambda i: "gfx936"))
        with patch.dict(os.environ, {"H3_GPU_CHECK_MODE": "inventory"}), \
             patch.object(h3_preflight.ctypes, "CDLL", return_value=hip), \
             patch.dict(sys.modules, {"torch": torch}):
            result = h3_preflight.gpu_check(8)
        self.assertEqual(len(result["devices"]), 8)
        torch.ones.assert_not_called()


    def driver(self, mode="native", extra=None, development_gpus=None):
        shell=os.environ.get("H3_TEST_GIT_BASH")
        if not shell or not Path(shell).is_file() or "system32" in shell.lower():
            self.skipTest("Explicit native Git Bash required")
        shutil.copyfile(ROOT/"scripts/h3_submission_common.sh",self.tools/"h3_submission_common.sh")
        shutil.copyfile(ROOT/"scripts/run_h3_development.sh",self.tools/"run_h3_development.sh")
        program = """import os,sys,json
kind=KIND
phase=sys.argv[1]
if kind=='generation' and os.environ.get('H3_PREFLIGHT_OUTPUT'):
    with open(os.environ['TRACE'],'a') as f:
        f.write(json.dumps(dict(kind='preflight',phase='in-generator',native=None,gpu_check=os.environ['H3_GPU_CHECK_MODE']))+'\\n')
    if os.environ.get('FAIL_PREFLIGHT'):raise SystemExit(int(os.environ['FAIL_PREFLIGHT']))
with open(os.environ['TRACE'],'a') as f:
    f.write(json.dumps(dict(kind=kind,phase=phase,native=os.environ.get('H3_INFINILM_VIDEO'),gpu_check=os.environ['H3_GPU_CHECK_MODE']))+'\\n')
key='FAIL_'+dict(phase=phase,bootstrap=phase,generation='generate',compare='compare',record=phase,delivery=phase).get(kind,phase).upper()
if os.environ.get(key):raise SystemExit(int(os.environ[key]))
if kind=='phase' and phase=='build':
    with open(os.environ['TRACE'],'a') as f:
        f.write(json.dumps(dict(kind='bootstrap',phase='audit',native=None,gpu_check=os.environ['H3_GPU_CHECK_MODE']))+'\\n')
    if os.environ.get('FAIL_AUDIT'):raise SystemExit(int(os.environ['FAIL_AUDIT']))
if kind=='record' and phase=='status':raise SystemExit(int(sys.argv[-1]))
"""
        for name,kind in (("h3_bootstrap.py","bootstrap"),("h3_delivery.py","delivery"),
                ("h3_submission_record.py","record"),("h3_preflight.py","preflight"),
                ("reproduce_h3.sh.py","phase"),("run_h3_compare.sh.py","compare")):
            (self.tools/name).write_text(program.replace("KIND",repr(kind)),encoding="utf-8")
        for name in ("reproduce_h3.sh","run_h3_compare.sh"):
            (self.tools/name).write_text('#!/usr/bin/env bash\nset -eu\n"$H3_BASE_PYTHON" "$0.py" "${1:-compare}"\n')
        (self.tools/"h3_repro_env.sh").write_text('python() { "$H3_BASE_PYTHON" "$@"; }\n')
        examples=self.repo/"InfiniLM/examples";examples.mkdir(parents=True,exist_ok=True)
        (examples/"minimax_h3.py").write_text(program.replace("KIND",repr("generation")))
        env={k:v for k,v in os.environ.items() if not k.startswith("H3_")}
        env.update(H3_BASE_PYTHON=Path(sys.executable).as_posix(),H3_NODE_ROOT=self.root.as_posix(),
            TRACE=str(self.trace),MODEL_PATH=(self.root/"model").as_posix(),
            IFAB_AFS_HOME=(self.root/"afs").as_posix(),H3_AFS_OUTPUT_DIR=(self.root/"afs/result").as_posix(),H3_DELIVERY_DIR=(self.root/"afs/package").as_posix())
        if extra:env.update(extra)
        name="h3_submission_common.sh"
        if development_gpus is not None:
            env["H3_DEV_GPUS"]=development_gpus
            name="run_h3_development.sh"
            mode="all"
        r=subprocess.run([shell,"--noprofile","--norc",(self.tools/name).as_posix(),mode],
            env=env,text=True,capture_output=True,timeout=25)
        rows=[json.loads(x) for x in self.trace.read_text().splitlines()] if self.trace.exists() else []
        return r,rows

    def test_development_single_gpu_still_runs_smoke(self):
        r,rows=self.driver(development_gpus="1")
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual([x["phase"] for x in rows if x["kind"]=="phase"],["prepare","build","jit","smoke"])
        self.assertFalse(any(x["kind"]=="compare" for x in rows))

    def test_development_eight_gpu_still_uses_short_path(self):
        r,rows=self.driver(development_gpus="8")
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual([x["phase"] for x in rows if x["kind"]=="phase"],["prepare","build","eight-smoke"])

    def test_native_only_never_compares(self):
        r,rows=self.driver()
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual([x["phase"] for x in rows if x["kind"]=="phase"],["prepare","build"])
        self.assertEqual(len([x for x in rows if x["kind"]=="generation"]),1)
        self.assertFalse(any(x["kind"]=="compare" for x in rows))
        self.assertEqual(len([x for x in rows if x["kind"]=="bootstrap" and x["phase"]=="audit"]),1)
        self.assertEqual([x["phase"] for x in rows if x["kind"]=="record"],["begin","native-check","status"])
        self.assertTrue(all(x["gpu_check"]=="inventory" for x in rows))
        self.assertEqual(rows[-1]["phase"],"persist")

    def test_native_preflight_runs_before_generation_and_offline_stage_is_not_repeated(self):
        r, rows = self.driver()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        actions = [(x["kind"], x["phase"]) for x in rows]
        self.assertLess(actions.index(("preflight", "in-generator")), actions.index(("generation", "fl2va")))
        self.assertEqual(sum(x["kind"] == "preflight" for x in rows), 1)
        self.assertNotIn(("bootstrap", "stage-wheels"), actions)

    def test_generation_hook_checks_once_and_fails_before_running_script(self):
        import runpy
        import types
        from unittest.mock import Mock
        hook = ROOT / "scripts/h3_compare_hook/sitecustomize.py"
        preflight = types.SimpleNamespace(run_preflight=Mock(), gpu_check=Mock(), bf16_math=Mock())
        settings = dict(H3_COMPARE_BF16_MATH="1", H3_PREFLIGHT_OUTPUT="preflight.json",
                        H3_SOURCE_ROOT="source", MODEL_PATH="model", FIRST_FRAME="image")
        with patch.dict(os.environ, settings), patch.object(sys, "argv", ["minimax_h3.py"]), \
             patch.dict(sys.modules, {"h3_preflight": preflight}):
            exec(compile(hook.read_text(), str(hook), "exec"), {})
            preflight.run_preflight.assert_called_once()
            preflight.gpu_check.assert_not_called()
            preflight.bf16_math.assert_not_called()
            preflight.run_preflight.side_effect = RuntimeError("bad input")
            with patch.object(os, "_exit", side_effect=SystemExit(78)), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    exec(compile(hook.read_text(), str(hook), "exec"), {})
            self.assertEqual(error.exception.code, 78)

    def test_check_generates_native_once_then_compares_existing_reference(self):
        r,rows=self.driver("check")
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(len([x for x in rows if x["kind"]=="generation"]),1)
        compare=next(x for x in rows if x["kind"]=="compare")
        self.assertTrue(compare["native"].endswith("/result.mp4"))
        actions=[(x["kind"],x["phase"]) for x in rows]
        self.assertLess(actions.index(("generation","fl2va")),actions.index(("compare","compare")))
        self.assertLess(actions.index(("compare","compare")),actions.index(("record","check")))

    def test_failures_stop_and_save_evidence_without_masking_rc(self):
        for key,code,no_generation in (("FAIL_BUILD",29,True),("FAIL_AUDIT",37,True),("FAIL_PREFLIGHT",78,True),
                ("FAIL_GENERATE",43,False),("FAIL_COMPARE",47,False),("FAIL_CHECK",41,False)):
            with self.subTest(key=key):
                self.trace.unlink(missing_ok=True)
                r,rows=self.driver("check",{key:str(code)})
                self.assertEqual(r.returncode,code,r.stdout+r.stderr)
                self.assertEqual(rows[-1]["phase"],"persist")
                if no_generation:self.assertFalse(any(x["kind"]=="generation" for x in rows))
                if key=="FAIL_GENERATE":self.assertFalse(any(x["kind"]=="compare" for x in rows))

    def test_save_failure_is_not_success(self):
        r,rows=self.driver(extra={"FAIL_PERSIST":"51"})
        self.assertEqual(r.returncode,51,r.stdout+r.stderr)

    def test_rejects_short_mode_and_old_videos(self):
        for env in ({"H3_SHORT_SMOKE":"1"},{"H3_INFINILM_VIDEO":"old"},{"H3_REFERENCE_VIDEO":"old"}):
            with self.subTest(env=env):
                r,rows=self.driver(extra=env)
                self.assertEqual(r.returncode,64,r.stdout+r.stderr)
                self.assertEqual(rows,[])

    def test_native_changed_after_generation_is_rejected(self):
        image=self.root/"image";image.write_bytes(b"image")
        native=self.root/"result.mp4";native.write_bytes(b"native")
        data=dict(first_frame=record.fingerprint(image),native_path=str(native),compare_dir=str(self.root/"compare"))
        inv=self.root/"inv.json";inv.write_text(json.dumps(data))
        (self.root/"native-check.json").write_text(json.dumps(dict(video=record.fingerprint(native))))
        native.write_bytes(b"changed")
        with patch.dict(os.environ,dict(H3_RUN_DIR=str(self.root),H3_SUBMISSION_INVOCATION=str(inv))):
            with self.assertRaisesRegex(RuntimeError,"Native video changed"):record.check()

    def storage_env(self):
        afs=self.root/"afs";afs.mkdir()
        run=self.root/"run";run.mkdir()
        snapshot=self.root/"storage.json"
        snapshot.write_text(json.dumps(dict(storage=dict(bytes=4*1024**3,
            usage=dict(bytes=1000,complete=True,over_quota=False,sampled_at=time.time())))))
        return run,dict(IFAB_AFS_HOME=str(afs),H3_DELIVERY_DIR=str(afs/"package"),
            H3_AFS_OUTPUT_DIR=str(afs/"outputs/result"),H3_AFS_INFO_JSON=str(snapshot))

    def test_afs_budget_and_persistence_preserve_failure_receipt(self):
        run,env=self.storage_env()
        (run/"result.mp4").write_bytes(b"video")
        with patch.dict(os.environ,env),contextlib.redirect_stdout(io.StringIO()):
            with patch.object(hashlib, "sha256", side_effect=AssertionError("SHA256 called")):
                delivery.reserve(run);delivery.persist(run,43)
        dest=Path(env["H3_AFS_OUTPUT_DIR"])
        receipt=json.loads((dest/"saved-manifest.json").read_text())
        self.assertEqual(receipt["pipeline_rc"],43)
        self.assertEqual((dest/"result.mp4").read_bytes(),b"video")
        with patch.dict(os.environ,env):
            with self.assertRaisesRegex(RuntimeError,"NEW directory"):delivery.persist(run,0)

    def test_afs_without_snapshot_reserves_bounded_output_and_persists(self):
        run, env = self.storage_env()
        env.pop("H3_AFS_INFO_JSON")
        (run / "result.mp4").write_bytes(b"video")
        clean = {k: v for k, v in os.environ.items() if k != "H3_AFS_INFO_JSON"}
        with patch.dict(os.environ, dict(clean, **env), clear=True), contextlib.redirect_stdout(io.StringIO()):
            delivery.reserve(run)
            reservation = json.loads((run / "afs-reservation.json").read_text(encoding="utf-8"))
            self.assertFalse(reservation["quota_checked"])
            self.assertIsNone(reservation["baseline"])
            self.assertIsNone(reservation["peak"])
            self.assertEqual(reservation["cap"], 128 * 1024**2)
            delivery.persist(run, 0)
        self.assertEqual((Path(env["H3_AFS_OUTPUT_DIR"]) / "result.mp4").read_bytes(), b"video")

    def test_afs_without_snapshot_still_refuses_oversize_output(self):
        run, env = self.storage_env()
        env.pop("H3_AFS_INFO_JSON")
        env["H3_AFS_OUTPUT_CAP_MIB"] = "1"
        clean = {k: v for k, v in os.environ.items() if k != "H3_AFS_INFO_JSON"}
        with patch.dict(os.environ, dict(clean, **env), clear=True), contextlib.redirect_stdout(io.StringIO()):
            delivery.reserve(run)
            (run / "huge.bin").write_bytes(b"x" * 1024**2)
            with self.assertRaisesRegex(RuntimeError, "exceeds reserved"):
                delivery.persist(run, 0)
        self.assertFalse(Path(env["H3_AFS_OUTPUT_DIR"]).exists())

    def test_afs_over_quota_or_stale_snapshot_never_creates_destination(self):
        run,env=self.storage_env()
        path=Path(env["H3_AFS_INFO_JSON"])
        for used,sampled in ((4*1024**3,time.time()),(1000,0)):
            path.write_text(json.dumps(dict(bytes=4*1024**3,usage=dict(bytes=used,complete=True,
                over_quota=False,sampled_at=sampled))))
            with patch.dict(os.environ,env):
                with self.assertRaises(RuntimeError):delivery.reserve(run)
            self.assertFalse(Path(env["H3_AFS_OUTPUT_DIR"]).exists())

    def test_afs_output_escape_and_oversize_refused(self):
        run,env=self.storage_env()
        with patch.dict(os.environ,dict(env,H3_AFS_OUTPUT_DIR=str(self.root/"outside"))):
            with self.assertRaisesRegex(RuntimeError,"inside personal AFS"):delivery.reserve(run)
        with patch.dict(os.environ,dict(env,H3_AFS_OUTPUT_CAP_MIB="1")),contextlib.redirect_stdout(io.StringIO()):
            delivery.reserve(run)
            (run/"huge").write_bytes(b"x"*1024**2)
            with self.assertRaisesRegex(RuntimeError,"exceeds reserved"):delivery.persist(run,0)
        self.assertFalse(Path(env["H3_AFS_OUTPUT_DIR"]).exists())

    def test_exported_worktree_detects_changed_code(self):
        source=self.root/"source";source.mkdir()
        path=source/"code.py";path.write_bytes(b"original")
        delivery.write_json(source/".h3-source.json",dict(git_head="real-base",git_status=" M code.py",
            files=[delivery.file_row(path,"code.py")]))
        self.assertEqual(source_identity(source)["kind"],"exported_worktree")
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError,"Exported source changed"):source_identity(source)

    def test_bundled_reference_matches_registered_image_and_rejects_other_case(self):
        env=dict(H3_DELIVERY_DIR=str(ROOT),FIRST_FRAME=str(ROOT/"inputs/first_frame.png"))
        with patch.dict(os.environ,env):
            self.assertEqual(record.verify_reference()["video"]["bytes"],1554000)
        other=self.root/"other.png";other.write_bytes(b"new-image")
        with patch.dict(os.environ,dict(env,FIRST_FRAME=str(other))):
            with self.assertRaisesRegex(RuntimeError,"different test case"):record.verify_reference()

    def test_check_uses_existing_reference_and_cannot_claim_reference_generation(self):
        image=self.root/"image";image.write_bytes(b"image")
        native=self.root/"result.mp4";native.write_bytes(b"native")
        ref=self.root/"reference.mp4";ref.write_bytes(b"reference")
        out=self.root/"compare";out.mkdir()
        (out/"compare.log").write_text("[result] MiniMax-H3 video comparison\n  FL2VA\n    Size: PASS\n    Length: PASS\n    Average SSIM: 94.277%\n")
        media=dict(frames=124,width=768,height=928,fps="24",audio_streams=1)
        inv=self.root/"inv.json"
        inv.write_text(json.dumps(dict(first_frame=record.fingerprint(image),native_path=str(native),
            compare_dir=str(out),provided_reference=dict(video=record.fingerprint(ref)))))
        (self.root/"native-check.json").write_text(json.dumps(dict(video=record.fingerprint(native),media=media)))
        with patch.dict(os.environ,dict(H3_RUN_DIR=str(self.root),H3_SUBMISSION_INVOCATION=str(inv))), \
                patch.object(record,"video_metadata",return_value=media),contextlib.redirect_stdout(io.StringIO()):
            record.check()
        result=json.loads((self.root/"evaluation.json").read_text())
        self.assertFalse(result["generated_reference_this_workflow"])
        self.assertFalse(result["official_same_run_speedup_available"])

    def test_staging_requires_explicit_node_before_reading_or_writing(self):
        package = self.root / "missing-package"
        for node in (None, ""):
            with self.subTest(node=node):
                with self.assertRaisesRegex(RuntimeError, "Set H3_NODE_ROOT"):
                    delivery.stage(package, node)
        self.assertFalse(package.exists())

    def test_unsafe_archive_member_is_rejected(self):
        for name in ("../escape","source/../../escape","/absolute","source\\..\\escape","C:/escape"):
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError):delivery.safe_relative(name)

    def test_invocation_records_actual_shared_diffusers_config_paths(self):
        model=self.root/"model"
        (model/"diffusers/transformer").mkdir(parents=True)
        config=model/"diffusers/transformer/config.json"
        config.write_text('{"num_layers": 32}')
        target=self.root/"invocation.json"
        env=dict(H3_SOURCE_ROOT=str(ROOT),FIRST_FRAME=str(ROOT/"inputs/first_frame.png"),
            H3_SUBMISSION_MODE="native",MODEL_PATH=str(model),H3_NATIVE_VIDEO=str(self.root/"result.mp4"),
            OUTPUT_DIR=str(self.root/"compare"),H3_SUBMISSION_INVOCATION=str(target))
        with patch.dict(os.environ,env),contextlib.redirect_stdout(io.StringIO()):
            record.begin()
        configs=json.loads(target.read_text())["model_configs"]
        self.assertEqual(configs,[record.fingerprint(config)])

if __name__ == "__main__":
    unittest.main(verbosity=2)
