"""CPU-only orchestration rehearsal. GPU/compiler/media children are fixtures."""
from __future__ import annotations
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import phase10_batch as batch
from phase10_evidence import promoted_policy
from phase10_jobs import Jobs,atomic
from phase10_profile import sha
from phase10_runtime import check_paths
from phase10_quality import compare_pair
from phase9_attention import shape_spec,render
from test_phase10_batch_cpu import BEST,worker_fixture


def gemm_fixture(key,run):
    candidate=dict(index=0,create_status=0,strict_eligible=True,
        selection=dict(backend="lt",index=0,key=key),repeat=dict(finite=True,bitwise_equal=True),
        all_repeats_bitwise_equal=True,vs_baseline=dict(finite=True,bitwise_equal=True),
        unaligned_rejected_before_write=True,median_gpu_ms=8.,
        paired=dict(schema=1,warmup=3,samples=[dict(baseline_before_ms=10.,baseline_after_ms=10.,
            candidate_gpu_ms=8.,candidate_wall_seconds=.008,
            baseline_bitwise=True,candidate_bitwise=True) for _ in range(7)]))
    cases=[dict(alpha=a,beta=b,baseline_repeat_bitwise=True,candidates=[
        dict(index=-1,median_gpu_ms=10.,repeat=dict(finite=True)),copy.deepcopy(candidate)])
        for a,b in ((1.,0.),(.75,0.),(.75,.25),(1.,1.))]
    return dict(complete=True,lt_executed=True,timing_mode="paired",runtime_tag="a"*64,
                fixture_run=run,cases=cases)


class MatrixTests(unittest.TestCase):
    def prepare(self,work):
        first=work/"first.png";first.write_bytes(b"fixture")
        return SimpleNamespace(workspace_root=work,output=work/"matrix",baseline_bundle=BEST,
            library_root=work/"install",model_path=work/"model",python=Path(sys.executable),
            compiler="cpu-fixture-only",freeze_dir=None,devices=list(range(8)),
            trace_devices=list(range(8)),first_frame=first,prompt="A dog falls asleep")

    def child_fixture(self,fail_profile=False):
        calls=[];active=set();lock=threading.Lock()
        def run(command,logfile,timeout,env,label):
            with lock:
                calls.append(label)
                if label.startswith(("full-","native-","quality")):
                    self.assertFalse(active,"TP8 phase overlaps local GPU lane")
                device=int(command[command.index("--device")+1]) if "--device" in command else None
                if device is not None:
                    self.assertNotIn(device,active);active.add(device)
            def arg(flag):return Path(command[command.index(flag)+1])
            try:
                logfile.write_text("CPU_FIXTURE_ONLY "+label+"\n")
                if label.startswith("full-"):
                    out=arg("--output");out.mkdir()
                    upstream=out/"upstream";upstream.mkdir()
                    atomic(upstream/"freeze.json",dict(complete=True))
                    (upstream/"upstream.pt").write_bytes(b"frozen")
                    (out/"dit-input.pt").write_bytes(b"dit-fixture")
                    atomic(out/"freeze-receipt.json",dict(complete=True,dit_calls=49,
                        snapshot_sha256="c"*64,first_input_equal=True))
                    atomic(out/"run.json",dict(complete=True,step_records=[{}]*49,
                        steady_synchronized_dit_median_seconds=4.39,
                        timings={"generation.pipeline_total":240.,"total.end_to_end":295.},
                        dit_snapshot=dict(sha256=sha(out/"dit-input.pt"),transformer_path=str(out/"resolved-transformer"))))
                    (out/"output.mp4").write_bytes(b"fixture-not-real-media")
                elif label.startswith("build-"):
                    out=arg("--output");out.mkdir()
                    lib=out/"libh3_flash_gfx936.so";lib.write_bytes(label.encode())
                    atomic(out/"build.json",dict(complete=True,arch="gfx936",
                        library_sha256=sha(lib),build_id="cpu-candidate",compiler_resources={"n_regs":99}))
                elif label.startswith(("screen-","final-")):
                    out=arg("--output")
                    data=worker_fixture(sha(arg("--candidate")),device=device,speedup=1.1)
                    data["fixture_run"]=label
                    if command[command.index("--layout")+1]=="legacy":
                        data["layout"]="legacy";data["cases"]=[]
                        for q,k in ((65,33),(2048,2048),(27570,27570)):
                            for padded in (False,True):
                                row=copy.deepcopy(worker_fixture()["cases"][0])
                                row.update(query=q,key=k,padded=padded,spec=shape_spec(q,k,padded))
                                row["versus_math_small"]=dict(finite=True,relative_l2=.001) if q==65 else None
                                row["speedup"]=1.1
                                row["candidate"].update(samples_ms=[10./1.1]*9,median_ms=10./1.1)
                                data["cases"].append(row)
                    atomic(out,data)
                elif label.startswith("promote-"):
                    out=arg("--output");out.mkdir()
                    i=command.index("--reports")+1;j=command.index("--output")
                    policy=promoted_policy(arg("--baseline-bundle"),arg("--aot"),list(map(Path,command[i:j])))
                    for digest,record in policy["libraries"].items():
                        source=(arg("--aot")/"libh3_flash_gfx936.so" if digest==policy["phase10_candidate"]
                                else BEST/record["file"])
                        shutil.copyfile(source,out/record["file"])
                    shutil.copyfile(BEST/"h3_flash_abi.h",out/"h3_flash_abi.h")
                    atomic(out/"policy.json",policy)
                    (out/"dispatcher.cpp").write_text(render(policy))
                    (out/"libh3_flash_dispatch.so").write_bytes(b"stub-dispatch-not-executable")
                    atomic(out/"bundle.json",dict(schema=1,files={p.name:sha(p) for p in out.iterdir()}))
                elif label=="native-profile":
                    out=arg("--output");out.mkdir()
                    rows=[dict(dtype=19,batch=1,m=128*(i+1),n=4096,k=256,transposed=1,
                        op_b="N",op_a="T",ldb=256,lda=256,ldc=128*(i+1)) for i in range(4)]
                    (out/"trace.log").write_text("\n".join("H3_GEMM_SHAPE "+json.dumps(row) for row in rows))
                    if fail_profile:return 7
                    atomic(out/"profile.json",dict(complete=True,devices=list(range(8)),top_kernels=[]))
                    atomic(out/"run.json",dict(complete=True))
                elif label=="quality":
                    atomic(arg("--output"),dict(complete=True,comparisons={},cpu_fixture_only=True))
                elif label.startswith("gemm-"):
                    key="h3lt1:"+("a"*64)+":"+label.rsplit("-r",1)[0]+":d"+str(device)
                    atomic(arg("--output"),gemm_fixture(key,label))
                else:raise AssertionError(label)
                if label=="native-profile" and not fail_profile:
                    # Nested DiT probe has its own output flag.
                    indexes=[i for i,v in enumerate(command) if v=="--output"]
                    out=Path(command[indexes[-1]+1]);out.mkdir()
                    atomic(out/"run.json",dict(complete=True))
                return 0
            finally:
                if device is not None:
                    with lock:active.remove(device)
        return calls,run

    def test_full_matrix_and_resume_use_real_validators_without_gpu_children(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            work=Path(folder);args=self.prepare(work)
            jobs=Jobs(args.output,{"cpu_fixture":True},3600,reserve=180)
            calls,child=self.child_fixture()
            with patch("phase10_jobs.logged",side_effect=child):
                batch.run_matrix(jobs,args)
            failures={key:t["attempts"][-1].get("error") for key,t in jobs.data["tasks"].items()
                      if t["attempts"] and t["attempts"][-1]["state"]!="passed"}
            self.assertEqual(failures,{})
            self.assertEqual(len(calls),83)
            self.assertEqual(sum(k.startswith("build-") for k in calls),16)
            self.assertEqual(sum(k.startswith("screen-") for k in calls),32)
            self.assertEqual(sum(k.startswith("final-") for k in calls),4)
            self.assertEqual(sum(k.startswith("full-") for k in calls),3)
            self.assertEqual(sum(k.startswith("gemm-") for k in calls),24)
            self.assertLess(calls.index("full-control"),calls.index("build-"+batch.config_id(batch.configurations()[0])))
            self.assertLess(calls.index("quality"),next(i for i,k in enumerate(calls) if k.startswith("gemm-")))
            self.assertTrue(jobs.data["complete"])
            resumed=Jobs(args.output,{"cpu_fixture":True},3600,reserve=180,resume=True)
            args=self.prepare(work)
            with patch("phase10_jobs.logged",side_effect=AssertionError("resume launched a child")):
                batch.run_matrix(resumed,args)
            self.assertTrue(resumed.data["complete"])
            self.assertTrue(all(len(t["attempts"])==1 for t in resumed.data["tasks"].values()))
            batch.summary(resumed)
            batch.export(resumed.output)

    def test_failed_native_profile_keeps_other_evidence_and_is_not_success(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            args=self.prepare(Path(folder));jobs=Jobs(args.output,{"cpu_fixture":True},3600,reserve=180)
            calls,child=self.child_fixture(fail_profile=True)
            with patch("phase10_jobs.logged",side_effect=child):
                batch.run_matrix(jobs,args)
            self.assertFalse(jobs.data["complete"])
            self.assertEqual(jobs.data["tasks"]["native-profile"]["attempts"][-1]["state"],"failed")
            self.assertIn("quality",calls)
            self.assertEqual(sum(k.startswith("gemm-") for k in calls),24)

    def test_budget_defers_screening_without_launching_aot(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            args=self.prepare(Path(folder));jobs=Jobs(args.output,{"cpu_fixture":True},1450,reserve=180)
            calls,child=self.child_fixture()
            with patch("phase10_jobs.logged",side_effect=child):
                batch.run_matrix(jobs,args)
            self.assertFalse(any(k.startswith("build-") for k in calls))
            self.assertFalse(jobs.data["complete"])
            self.assertEqual(sum("deferred" in t for t in jobs.data["tasks"].values()),16)
            self.assertIn("native-profile",calls)


class QualityTests(unittest.TestCase):
    def test_output_contract_same_upstream_and_first_divergence(self):
        import numpy as np
        fps=[24]
        class Container:
            def __init__(self):
                self.streams=SimpleNamespace(video=[SimpleNamespace(width=448,height=256,average_rate=fps[0])],
                    audio=[SimpleNamespace(rate=32000,codec_context=SimpleNamespace(channels=2))])
            def __enter__(self):return self
            def __exit__(self,*unused):return False
            def decode(self,**kwargs):
                if "video" in kwargs:return [None]*124
                return [SimpleNamespace(to_ndarray=lambda:np.array([[.1,.2,.3],[.3,.4,.5]]))]
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/name for name in ("baseline","candidate")]
            for i,path in enumerate(paths):
                path.mkdir()
                atomic(path/"run.json",dict(complete=True,step_records=[{}]*49,
                    steady_synchronized_dit_median_seconds=4.39,
                    timings={"generation.pipeline_total":240.,"total.end_to_end":295.}))
                atomic(path/"freeze-receipt.json",dict(complete=True,snapshot_sha256="c"*64,first_input_equal=True))
                rows=[{"fingerprint":0}]+[{"fingerprint":i} for _ in range(48)]
                (path/"dit-inputs.jsonl").write_bytes(("\n".join(map(json.dumps,rows))+"\n").encode())
                (path/"output.mp4").write_bytes(b"decoder-fixture")
            with patch.dict(sys.modules,{"av":SimpleNamespace(open=lambda *a:Container())}):
                result=compare_pair(*paths,lambda *a:{"fixture_ssim":1.})
                self.assertEqual(result["first_different_dit_input"],1)
                self.assertTrue(result["audio"]["bitwise_equal"])
                self.assertFalse(result["model_quality_accepted"])
                fps[0]=30
                with self.assertRaisesRegex(ValueError,"124 frames"):
                    compare_pair(*paths,lambda *a:{})
                fps[0]=24
                receipt=json.loads((paths[1]/"freeze-receipt.json").read_text())
                receipt["snapshot_sha256"]="d"*64
                atomic(paths[1]/"freeze-receipt.json",receipt)
                with self.assertRaisesRegex(ValueError,"same upstream"):
                    compare_pair(*paths,lambda *a:{})


class RuntimeTests(unittest.TestCase):
    def test_runtime_rejects_wrong_native_and_python_origins(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder);root=work/"install";root.mkdir()
            modules={"infinilm":work/"infinilm.py"};modules["infinilm"].write_bytes(b"module")
            libs=[root/"libinfiniop.so",root/"libinfinirt.so",work/"_infinilm.so"]
            for p in libs:p.write_bytes(b"library")
            checked=check_paths(work,root,modules,libs)
            self.assertEqual(len(checked["mapped_native_sha256"]),3)
            wrong=work/"libinfiniop.so";wrong.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError,"origin mismatch"):
                check_paths(work,root,modules,[wrong,*libs[1:]])
            with self.assertRaisesRegex(ValueError,"Python origin"):
                check_paths(root,root,modules,libs)


if __name__=="__main__":
    unittest.main(verbosity=2)
