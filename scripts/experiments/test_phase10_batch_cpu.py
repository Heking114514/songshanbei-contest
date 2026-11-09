"""CPU process/ledger/evidence tests; never starts a model or GPU executable."""
from __future__ import annotations
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase10_jobs import Jobs, atomic, files_in, lanes, logged
from phase10_batch import configurations, config_id, export, summary, pipeline_command, check_full, phase_budget
from phase10_evidence import checked_attention, promoted_policy
from phase10_profile import sha
from phase9_attention import shape_spec
from phase10_attention_worker import verified_baseline
from phase9_pipeline import main as pipeline_main

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parents[2]
BEST=EXPERIMENT/"phase9_live3/evidence/phase9-dispatch-mixed"


class JobsTests(unittest.TestCase):
    def make(self,folder,**kw):
        return Jobs(Path(folder)/"matrix",{"sources":"bound"},60,reserve=1,**kw)

    def command(self,path):
        code="from pathlib import Path; print('live-line',flush=True); Path("+repr(str(path/"value"))+").write_text('ok')"
        return [sys.executable,"-u","-c",code]

    def validate(self,path):
        if (path/"value").read_text()!="ok": raise ValueError("bad result")
        return {"sha256":sha(path/"value")}

    def test_pass_and_resume_revalidate_artifacts_without_executing(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs=self.make(folder)
            result=jobs.run("one",self.command,self.validate,timeout=10)
            self.assertIsNotNone(result)
            log=(result["directory"]/"terminal.log").read_text()
            self.assertIn("live-line",log)
            before=files_in(result["directory"])
            jobs=self.make(folder,resume=True)
            cached=jobs.run("one",self.command,self.validate,timeout=10)
            self.assertTrue(cached["reused"])
            self.assertEqual(before,files_in(result["directory"]))
            self.assertEqual(len(jobs.data["tasks"]["one"]["attempts"]),1)

    def test_tampered_success_and_changed_plan_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs=self.make(folder)
            result=jobs.run("one",self.command,self.validate,timeout=10)
            (result["directory"]/"value").write_text("tampered")
            with self.assertRaisesRegex(ValueError,"artifacts changed"):
                jobs.run("one",self.command,self.validate,timeout=10)
            with self.assertRaisesRegex(ValueError,"plan changed"):
                Jobs(jobs.output,{"sources":"changed"},60,reserve=1,resume=True)

    def test_failed_attempt_retained_and_retry_uses_new_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs=self.make(folder)
            fail=lambda path:[sys.executable,"-c","raise RuntimeError('deliberate')"]
            self.assertIsNone(jobs.run("one",fail,self.validate,timeout=10))
            first=jobs.data["tasks"]["one"]["attempts"][0]
            self.assertEqual(first["state"],"failed")
            self.assertIn("deliberate",(jobs.output/first["directory"]/"terminal.log").read_text())
            self.assertIsNone(jobs.run("one",self.command,self.validate,timeout=10))
            jobs=self.make(folder,resume=True,retry_failed=True)
            self.assertIsNotNone(jobs.run("one",self.command,self.validate,timeout=10))
            attempts=jobs.data["tasks"]["one"]["attempts"]
            self.assertEqual([a["state"] for a in attempts],["failed","passed"])
            self.assertNotEqual(attempts[0]["directory"],attempts[1]["directory"])

    def test_timeout_terminates_child_and_keeps_partial_log(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs=self.make(folder)
            def command(path):
                return [sys.executable,"-u","-c","import time;print('before-timeout',flush=True);time.sleep(30)"]
            started=time.monotonic()
            self.assertIsNone(jobs.run("timeout",command,self.validate,timeout=.5))
            self.assertLess(time.monotonic()-started,8)
            attempt=jobs.data["tasks"]["timeout"]["attempts"][0]
            self.assertIn("TimeoutError",attempt["error"])
            self.assertIn("before-timeout",(jobs.output/attempt["directory"]/"terminal.log").read_text())

    def test_budget_never_launches_child_and_phase_guard_restores(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs=self.make(folder)
            command=lambda path:(_ for _ in ()).throw(AssertionError("should not launch"))
            with phase_budget(jobs,0):
                self.assertIsNone(jobs.run("skip",command,self.validate,timeout=10,estimate=1))
            self.assertGreater(jobs.remaining(),10)
            self.assertEqual(jobs.data["tasks"]["skip"]["deferred"]["state"],"skipped_budget")
            self.assertEqual(jobs.data["tasks"]["skip"]["attempts"],[])

    def test_interrupted_attempt_is_marked_and_new_attempt_preserves_it(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs=self.make(folder)
            jobs.data["tasks"]["crash"]=dict(attempts=[dict(state="running",directory="old")])
            jobs.save()
            jobs=self.make(folder,resume=True)
            self.assertEqual(jobs.data["tasks"]["crash"]["attempts"][0]["state"],"interrupted")
            self.assertIsNotNone(jobs.run("crash",self.command,self.validate,timeout=10))

    def test_one_lane_per_device_and_no_full_phase_interleaving(self):
        lock=threading.Lock()
        active=set()
        visits=[]
        def worker(device,row):
            with lock:
                self.assertNotIn(device,active)
                active.add(device)
            time.sleep(.005)
            with lock:
                visits.append((device,row))
                active.remove(device)
            return row
        values=lanes([0,1,2,3],list(range(16)),worker)
        self.assertEqual(sorted(values),list(range(16)))
        self.assertFalse(active)
        self.assertEqual(len(visits),16)

    def test_partial_worker_json_does_not_block_summary_or_raw_export(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs=self.make(folder)
            def child(command,logfile,*unused):
                logfile.write_text("partial evidence retained")
                (logfile.parent/"worker.json").write_text('{"complete":')
                return 4
            with patch("phase10_jobs.logged",side_effect=child):
                self.assertIsNone(jobs.run("screen-bad",self.command,self.validate,timeout=10))
            summary(jobs)
            report=json.loads((jobs.output/"summary.json").read_text())
            self.assertEqual(len(report["evidence_parse_errors"]),1)
            receipt=export(jobs.output)
            self.assertTrue(Path(receipt["metadata"]["path"]).is_file())
            with tarfile.open(receipt["path"]) as archive:
                self.assertTrue(any(name.endswith("worker.json") for name in archive.getnames()))

    def test_archive_manifest_hashes_raw_failures_and_results(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"out"
            output.mkdir()
            atomic(output/"failed.json",{"complete":False,"error":"retained"})
            (output/"raw.log").write_text("raw")
            receipt=export(output)
            self.assertEqual(sha(receipt["path"]),receipt["sha256"])
            with tarfile.open(receipt["path"]) as archive:
                manifest=json.load(archive.extractfile("evidence-manifest.json"))
                for name,digest in manifest["files"].items():
                    import hashlib
                    self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(),digest)


def worker_fixture(candidate_sha="b"*64,device=0,speedup=1.2):
    rows=[]
    for q,k in ((65,33),(708,708),(27570,27570)):
        spec=shape_spec(q,k,False)
        spec["strides"][2]=[k*7*128*3,7*128*3,128,1]
        metric=dict(finite=True,relative_l2=.001)
        row=dict(query=q,key=k,heads=7,padded=False,spec=spec,repeat_bitwise=True,
            versus_previous_best=metric.copy(),graph_updated_inputs=metric.copy(),
            versus_math_small=metric.copy() if q==65 else None,
            speedup=speedup if q==27570 else .95,
            graph=dict(samples_ms=[8.]*9,median_ms=8.))
        row["baseline"]=dict(samples_ms=[10.]*9,median_ms=10.)
        candidate_ms=10./row["speedup"]
        row["candidate"]=dict(samples_ms=[candidate_ms]*9,median_ms=candidate_ms)
        for name in ("baseline","candidate"):
            row[name]["paired_order"]=[["baseline","candidate"] if i%2==0 else ["candidate","baseline"] for i in range(9)]
        rows.append(row)
    return dict(complete=True,screening_pass=True,arch="gfx936",baseline_scope="phase9-best-dispatch",
        layout="mixed-v",device=device,worker_source_sha256=sha(HERE/"phase10_attention_worker.py"),
        real_worst_speedup=speedup,libraries=dict(
            baseline=dict(sha256=sha(BEST/"libh3_flash_dispatch.so")),
            candidate=dict(sha256=candidate_sha,build_id="cpu-candidate")),cases=rows)


class EvidenceTests(unittest.TestCase):
    def test_valid_and_invalid_case_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"worker.json"
            data=worker_fixture()
            atomic(path,data)
            checked_attention(path,"b"*64,sha(BEST/"libh3_flash_dispatch.so"))
            for change in ("duplicate","source","nan","median","layout","missing","order"):
                bad=copy.deepcopy(data)
                if change=="duplicate":bad["cases"][1]=copy.deepcopy(bad["cases"][0])
                if change=="source":bad["worker_source_sha256"]="0"*64
                if change=="nan":bad["cases"][0]["graph_updated_inputs"]["relative_l2"]=float("nan")
                if change=="median":bad["cases"][0]["candidate"]["median_ms"]=1.
                if change=="layout":bad["cases"][0]["spec"]["strides"][2][1]+=1
                if change=="missing":bad["cases"].pop()
                if change=="order":bad["cases"][0]["baseline"].pop("paired_order")
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError,msg=change):
                    checked_attention(path,"b"*64,sha(BEST/"libh3_flash_dispatch.so"))

    def test_promotion_requires_two_same_device_runs_and_changes_only_exact_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            aot=root/"aot"
            aot.mkdir()
            lib=aot/"libh3_flash_gfx936.so"
            lib.write_bytes(b"cpu-only-candidate")
            digest=sha(lib)
            atomic(aot/"build.json",dict(complete=True,arch="gfx936",library_sha256=digest,build_id="cpu-candidate"))
            reports=[root/"a.json",root/"b.json"]
            for i,p in enumerate(reports):
                data=worker_fixture(digest)
                data["fixture_run"]=i
                atomic(p,data)
            result=promoted_policy(BEST,aot,reports)
            old=json.loads((BEST/"policy.json").read_text())
            changed=[r for r in result["rules"] if r["library"]==digest]
            self.assertEqual(len(changed),1)
            self.assertEqual(changed[0]["spec"]["query_tokens"],27570)
            self.assertEqual(result["baseline"],old["baseline"])
            self.assertEqual(len(result["rules"]),len(old["rules"]))
            with self.assertRaisesRegex(ValueError,"independent"):
                promoted_policy(BEST,aot,[reports[0],reports[0]])
            bad=json.loads(reports[1].read_text())
            bad["device"]=1
            atomic(reports[1],bad)
            with self.assertRaisesRegex(ValueError,"same GPU"):
                promoted_policy(BEST,aot,reports)

    def test_full_timing_boundaries_and_incomplete_model_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/"run").mkdir()
            run=dict(complete=True,step_records=[{}]*49,steady_synchronized_dit_median_seconds=4.395,
                     timings={"generation.pipeline_total":240.,"total.end_to_end":295.})
            atomic(path/"run/run.json",run)
            atomic(path/"run/freeze-receipt.json",dict(complete=True,dit_calls=49))
            (path/"run/output.mp4").write_bytes(b"fixture")
            checked=check_full(path)
            self.assertEqual(checked["steady_step"],4.395)
            self.assertEqual(checked["timings"]["generation.pipeline_total"],240.)
            run["complete"]=False
            atomic(path/"run/run.json",run)
            with self.assertRaisesRegex(ValueError,"Incomplete"):check_full(path)


class PlanTests(unittest.TestCase):
    def test_sixteen_unique_configs_and_default_plan_never_imports_gpu(self):
        self.assertEqual(len(configurations()),16)
        self.assertEqual(len({config_id(r) for r in configurations()}),16)
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder)
            first=work/"first.png"
            first.write_bytes(b"fixture")
            output=work/"plan"
            command=[sys.executable,str(HERE/"phase10_batch.py"),"--workspace-root",str(work),
                "--output",str(output),"--first-frame",str(first),"--baseline-bundle",str(work/"best"),
                "--library-root",str(work/"install"),"--model-path",str(work/"model")]
            result=subprocess.run(command,capture_output=True,text=True,encoding="utf-8")
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((output/"plan.json").read_text())
            self.assertFalse(report["execute"])
            self.assertEqual(report["plan"]["matrix"]["attention_variants"],16)
            self.assertEqual(set(p.name for p in output.iterdir()),{"plan.json"})

    def test_pipeline_snapshot_is_plan_only_without_execute(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder)
            argv=["phase9_pipeline.py","--workspace-root",str(work),"--output",str(work/"p"),
                  "--capture-dit-step","2","--","fl2va"]
            with patch.object(sys,"argv",argv),redirect_stdout(io.StringIO()):
                pipeline_main()
            value=json.loads((work/"p/plan.json").read_text())
            self.assertEqual(value["capture_dit_step"],2)
            self.assertFalse(value["execute"])


if __name__=="__main__":
    unittest.main(verbosity=2)
