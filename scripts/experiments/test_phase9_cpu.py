"""CPU checks include the actual upstream loop/scheduler bodies and a compiled fake-ABI dispatcher."""
from __future__ import annotations
import ast
from contextlib import nullcontext
import copy
import ctypes as C
from dataclasses import dataclass
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase9_attention import BASELINE_SHA, from_archive, checked_cases, shape_spec, select, render, sha, verify_bundle, with_layout_evidence
from phase9_freeze import FrozenLoop, fingerprints, restore_tree, generation_contract
from phase9_profile import summarize, duration, intersection
from phase9_pipeline import parse_steps
from phase8_attention_worker import Spec,Call,Api

SOURCE=Path(__file__).resolve().parents[2]
EXPERIMENT=SOURCE.parent
WORKSPACE=EXPERIMENT.parents[1]
ARCHIVE=EXPERIMENT/"phase8_validation4/attention36.tar.gz"
WHEEL=WORKSPACE/"handover/results/h3-env-wheels/diffusers-0.40.0-py3-none-any.whl"


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy=from_archive(ARCHIVE)
        with tarfile.open(ARCHIVE) as t:
            cls.worker=json.load(t.extractfile(next(m for m in t.getmembers() if m.name.endswith("/worker.json"))))

    def test_exact_layout_unknowns_use_baseline(self):
        for r in self.policy["rules"]:
            self.assertEqual(select(self.policy,r["spec"]),r["library"])
            for change in ("batch","heads","query_tokens"):
                spec=copy.deepcopy(r["spec"]);spec[change]+=1
                self.assertEqual(select(self.policy,spec),BASELINE_SHA)
            spec=copy.deepcopy(r["spec"]);spec["strides"][2][1]+=1
            self.assertEqual(select(self.policy,spec),BASELINE_SHA)
        self.assertFalse(self.policy["model_acceptance"])

    def test_rejects_incomplete_numerical_graph_and_timing_errors(self):
        for change in ("complete","duplicate","numerical","graph","median","math"):
            w=copy.deepcopy(self.worker)
            if change=="complete":w["complete"]=False
            if change=="duplicate":w["cases"][1]=copy.deepcopy(w["cases"][0])
            if change=="numerical":w["cases"][0]["versus_v4"]["relative_l2"]=.1
            if change=="graph":w["cases"][0]["graph_updated_inputs"]["finite"]=False
            if change=="median":w["cases"][0]["candidate"]["median_ms"]=1e6
            if change=="math":w["cases"][0]["versus_math_small"]["relative_l2"]=.1
            with self.assertRaises(ValueError,msg=change):checked_cases(w)

    def test_changed_worker_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"tampered.tar"
            with tarfile.open(ARCHIVE) as original,tarfile.open(path,"w") as target:
                for m in original.getmembers():
                    data=original.extractfile(m).read() if m.isfile() else None
                    if m.name.endswith("/worker.json"): data=data+b" "
                    if data is not None:m.size=len(data);target.addfile(m,io.BytesIO(data))
                    else:target.addfile(m)
            with self.assertRaisesRegex(ValueError,"hash mismatch"):from_archive(path)

    def test_negative_policy_parameters(self):
        with self.assertRaises(ValueError):from_archive(ARCHIVE,minimum_gain=-1)

    def test_render_refuses_path_injection(self):
        p=copy.deepcopy(self.policy);next(iter(p["libraries"].values()))["file"]="../child.so"
        with self.assertRaises(ValueError):render(p)


    def test_bundle_content_and_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);policy=copy.deepcopy(self.policy)
            (root/"policy.json").write_text(json.dumps(policy))
            files=["dispatcher.cpp","h3_flash_abi.h","libh3_flash_dispatch.so",
                   *(r["file"] for r in policy["libraries"].values())]
            for name in files:(root/name).write_bytes(b"cpu-fixture")
            hashes={p.name:sha(p.read_bytes()) for p in root.iterdir()}
            (root/"bundle.json").write_text(json.dumps(dict(schema=1,files=hashes)))
            verify_bundle(root)
            (root/"libh3_flash_dispatch.so").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"SHA mismatch"):verify_bundle(root)
            del hashes["libh3_flash_dispatch.so"]
            (root/"bundle.json").write_text(json.dumps(dict(schema=1,files=hashes)))
            with self.assertRaisesRegex(ValueError,"Incomplete"):verify_bundle(root)

    def mixed_evidence(self):
        candidate=next(r["library"] for r in self.policy["rules"]
                       if r["spec"]["query_tokens"]==27570 and r["spec"]["strides"][0][1]==896)
        w=copy.deepcopy(self.worker);w.update(layout="mixed-v",
            worker_source_sha256=sha((SOURCE/"scripts/experiments/phase8_attention_worker.py").read_bytes()))
        w["libraries"]["candidate"]=copy.deepcopy(self.policy["libraries"][candidate])
        rows=[]
        for q,k in ((65,33),(708,708),(27570,27570)):
            row=copy.deepcopy(self.worker["cases"][0]);row.update(query=q,key=k,padded=False,heads=7)
            spec=shape_spec(q,k,False);spec["strides"][2]=[k*2688,2688,128,1];row["spec"]=spec
            rows.append(row)
        w["cases"]=rows
        return w,candidate

    def test_live_mixed_layout_does_not_extrapolate(self):
        w,candidate=self.mixed_evidence()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"live.json";path.write_text(json.dumps(w))
            original=copy.deepcopy(self.policy);p=with_layout_evidence(self.policy,path)
            self.assertEqual(self.policy,original)
            for row in w["cases"]:
                spec=copy.deepcopy(row["spec"])
                expected=candidate if row["speedup"]>=1+p["minimum_gain"] else BASELINE_SHA
                self.assertEqual(select(p,spec),expected)
                spec["strides"][2][1]+=128
                self.assertEqual(select(p,spec),BASELINE_SHA)
            self.assertIn("live_layout_evidence_sha256",p)

    def test_live_mixed_evidence_rejects_unbound_layout_and_bad_graph(self):
        w,_=self.mixed_evidence()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"live.json"
            for change in ("layout","graph","worker","candidate","median","duplicate","incomplete"):
                bad=copy.deepcopy(w)
                if change=="layout":bad["cases"][-1]["spec"]["strides"][2][1]=896
                if change=="graph":bad["cases"][-1]["graph_updated_inputs"]["finite"]=False
                if change=="worker":bad["worker_source_sha256"]="0"*64
                if change=="candidate":bad["libraries"]["candidate"]["sha256"]="0"*64
                if change=="median":bad["cases"][-1]["graph"]["median_ms"]=1e6
                if change=="duplicate":bad["cases"][-1]=copy.deepcopy(bad["cases"][0])
                if change=="incomplete":bad["complete"]=False
                path.write_text(json.dumps(bad))
                original=copy.deepcopy(self.policy)
                with self.assertRaises(ValueError,msg=change):with_layout_evidence(self.policy,path)
                self.assertEqual(self.policy,original)



def upstream_classes():
    with zipfile.ZipFile(WHEEL) as z:
        denoise=ast.parse(z.read("diffusers/modular_pipelines/minimax_h3/denoise.py").decode())
        scheduler=ast.parse(z.read("diffusers/schedulers/scheduling_minimax_h3.py").decode())
    class Base:
        def get_block_state(self,state):return SimpleNamespace(**state)
        def set_block_state(self,state,block):state.update(vars(block))
        def progress_bar(self,total):
            class Progress:
                def __enter__(self):return self
                def update(self):pass
                def __exit__(self,*a):pass
            return Progress()
    wanted={"MiniMaxH3DenoiseLoopWrapper","MiniMaxH3LoopDenoiser","MiniMaxH3LoopSchedulerStep"}
    nodes=[n for n in denoise.body if isinstance(n,ast.ClassDef) and n.name in wanted]
    scheduler_class=next(n for n in scheduler.body if isinstance(n,ast.ClassDef) and n.name=="MiniMaxH3Scheduler")
    method=next(n for n in scheduler_class.body if isinstance(n,ast.FunctionDef) and n.name=="step")
    fake=ast.ClassDef(name="ActualScheduler",bases=[],keywords=[],body=[method],decorator_list=[])
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),
                           *nodes,fake],type_ignores=[])
    ast.fix_missing_locations(module)
    env=dict(torch=torch,inspect=inspect,ModularPipelineBlocks=Base,LoopSequentialPipelineBlocks=Base)
    exec(compile(module,str(WHEEL),"exec"),env)
    return env


@dataclass
class Inputs:
    video_hidden_states:torch.Tensor
    audio_hidden_states:torch.Tensor
    encoder_hidden_states:torch.Tensor
    timestep:torch.Tensor
    timestep_indices:torch.Tensor
    token_tags:torch.Tensor
    position_ids:torch.Tensor
    video_indices:torch.Tensor
    audio_indices:torch.Tensor
    text_indices:torch.Tensor


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.env=upstream_classes()
        self.capture=self.root/"capture";self.capture.mkdir()
        self.snapshot=self.capture/"upstream"

    def tearDown(self):self.temp.cleanup()

    def run_loop(self,mode,output,upstream_offset=0,model_scale=.1,contract=None):
        output.mkdir(exist_ok=True)
        freezer=FrozenLoop(mode,self.snapshot,output,contract or {"seed":42})
        Loop=self.env["MiniMaxH3DenoiseLoopWrapper"]
        original=Loop.get_block_state
        restore=freezer.install(Loop,{"wheel":WHEEL})
        layout=dict(token_tags=torch.tensor([0]*4+[1]*3+[2]*3),
                    position_ids=torch.arange(30).reshape(10,3),
                    video_indices=torch.arange(4),audio_indices=torch.arange(4,7),text_indices=torch.arange(7,10))
        state=dict(latents=torch.arange(8,dtype=torch.float32).reshape(4,2)+upstream_offset,
                   audio_latents=torch.arange(6,dtype=torch.float32).reshape(3,2)+upstream_offset,
                   prompt_embeds=torch.ones(1,3,4)+upstream_offset,
                   row_timestep_plan=[(torch.tensor([t]),torch.zeros(10,dtype=torch.int64)) for t in (.1,.5)],
                   timesteps=torch.tensor([.1,.5]),audio_timesteps=torch.tensor([.1,.5]),
                   denoiser_input_fields=layout,attention_kwargs=None,
                   num_condition_video_rows=1,num_condition_audio_rows=1)
        def scheduler():
            s=self.env["ActualScheduler"]();s.timesteps=state["timesteps"].clone()
            s.sigmas=torch.tensor([.9,.5,0.]);s._step_index=None;s._begin_index=None;s.num_inference_steps=2
            s.config={"algorithm":"euler"};s.index_for_timestep=lambda t:0
            return s
        class Transformer:
            def forward(owner,hidden_states,audio_hidden_states,encoder_hidden_states,timestep,timestep_indices,
                        attention_kwargs,return_dict,token_tags,position_ids,video_indices,audio_indices,text_indices):
                inputs=Inputs(hidden_states,audio_hidden_states,encoder_hidden_states,timestep,timestep_indices,
                              token_tags,position_ids,video_indices,audio_indices,text_indices)
                freezer.observe(inputs)
                return hidden_states*model_scale,audio_hidden_states*model_scale
            __call__=forward
        components=SimpleNamespace(scheduler=scheduler(),audio_scheduler=scheduler(),transformer=Transformer())
        denoiser=self.env["MiniMaxH3LoopDenoiser"]();denoiser.transformer_name="transformer"
        update=self.env["MiniMaxH3LoopSchedulerStep"]();loop=Loop()
        def step(c,b,i,t):
            c,b=denoiser(c,b,i,t);return update(c,b,i,t)
        loop.loop_step=step
        try:
            loop(components,state);freezer.close(True)
        except BaseException:
            freezer.close(False)
            raise
        finally:
            restore()
            self.assertIs(Loop.get_block_state,original)
            self.assertNotIn("get_block_state",Loop.__dict__)
        rows=[json.loads(line) for line in (output/"dit-inputs.jsonl").read_text().splitlines()]
        return state,rows

    def test_actual_loop_replay_same_model_is_bitwise_equal(self):
        baseline,rows=self.run_loop("capture",self.capture)
        replay,again=self.run_loop("replay",self.root/"replay",upstream_offset=100)
        self.assertTrue(torch.equal(baseline["latents"],replay["latents"]))
        self.assertTrue(torch.equal(baseline["audio_latents"],replay["audio_latents"]))
        self.assertEqual(rows,again)
        self.assertNotEqual(rows[0]["inputs"]["video_hidden_states"],rows[1]["inputs"]["video_hidden_states"])

    def test_candidate_recurrence_is_not_overwritten_each_step(self):
        baseline,rows=self.run_loop("capture",self.capture)
        candidate,again=self.run_loop("replay",self.root/"candidate",upstream_offset=30,model_scale=.2)
        self.assertEqual(rows[0],again[0])
        self.assertNotEqual(rows[1]["inputs"]["audio_hidden_states"],again[1]["inputs"]["audio_hidden_states"])
        self.assertFalse(torch.equal(baseline["audio_latents"],candidate["audio_latents"]))
        self.assertTrue(torch.equal(candidate["latents"][:1],baseline["latents"][:1]))

    def test_contract_and_snapshot_tampering_rejected(self):
        self.run_loop("capture",self.capture)
        with self.assertRaisesRegex(ValueError,"contract changed"):
            self.run_loop("replay",self.root/"bad-contract",contract={"seed":43})
        p=self.snapshot/"upstream.pt";p.write_bytes(p.read_bytes()+b"x")
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):
            self.run_loop("replay",self.root/"bad-sha")

    def test_first_input_and_incomplete_capture_rejected(self):
        self.run_loop("capture",self.capture)
        p=self.snapshot/"freeze.json";meta=json.loads(p.read_text())
        meta["first_dit_input"]["timestep"]["sha256"]="bad";p.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,"First complete"):
            self.run_loop("replay",self.root/"bad-first")
        meta["complete"]=False;p.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            self.run_loop("replay",self.root/"incomplete")

    def test_schema_dtype_and_unknown_object_rejected(self):
        with self.assertRaises(ValueError):restore_tree(torch.zeros(2),torch.zeros(3))
        with self.assertRaises(ValueError):restore_tree(torch.zeros(2),torch.zeros(2,dtype=torch.float64))
        with self.assertRaises(TypeError):fingerprints({"unexpected":object()})

    def test_same_content_different_node_paths_contract(self):
        from argparse import Namespace
        a=self.root/"a";b=self.root/"b";a.mkdir();b.mkdir()
        (a/"image.png").write_bytes(b"image");(b/"image.png").write_bytes(b"image")
        first=Namespace(model_path=str(a),output="first",gpus="0",first_frame=str(a/"image.png"),seed=42)
        second=Namespace(model_path=str(b),output="other",gpus="0,1",first_frame=str(b/"image.png"),seed=42)
        self.assertEqual(generation_contract(first),generation_contract(second))

    def test_resolved_model_config_changes_are_bound(self):
        from argparse import Namespace
        model=self.root/"model";model.mkdir()
        (model/"model_index.json").write_text('{"class":"MiniMaxH3"}')
        args=Namespace(model_path="irrelevant-node-alias",output="o",gpus="0",seed=42)
        paths=SimpleNamespace(modular_dir=model,partition_dir=model)
        first=generation_contract(args,paths)
        self.assertTrue(first["model_configs"])
        (model/"model_index.json").write_text('{"class":"changed"}')
        self.assertNotEqual(first,generation_contract(args,paths))



class ProfileTests(unittest.TestCase):
    def test_union_and_overlap_do_not_double_count(self):
        events=[]
        for name,start,dur,dev in [("gemm",0,100,0),("flash",20,100,0),("rcclAllReduce",80,80,0),
                                    ("gemm",0,300,1)]:
            events.append(dict(ph="X",cat="kernel",name=name,ts=start,dur=dur,args={"device":dev}))
        # CPU nested scopes must not count as device time.
        events.append(dict(ph="X",cat="cpu_op",name="gemm",ts=0,dur=10000))
        result=summarize(dict(traceEvents=events))
        d=result["devices"]["0"]
        self.assertAlmostEqual(d["busy_union_ms"],.16)
        self.assertAlmostEqual(d["communication_compute_overlap_ms"],.04)
        self.assertAlmostEqual(d["exposed_communication_ms"],.04)
        self.assertIsNone(result["wall_clock_speedup"])
        self.assertEqual(set(result["devices"]),{"0","1"})

    def test_memory_only_trace_does_not_claim_native_kernels(self):
        event=dict(ph="X",cat="gpu_memcpy",name="copy",ts=0,dur=30,args={"device":0})
        self.assertFalse(summarize({"traceEvents":[event]})["gpu_events_present"])

    def test_no_gpu_events_and_bad_intervals(self):
        self.assertFalse(summarize({"traceEvents":[]})["gpu_events_present"])
        self.assertEqual(duration([(0,2),(1,4),(4,5)]),5)
        self.assertEqual(intersection([(0,5)],[(2,3),(3,7)]),3)
        with self.assertRaises(ValueError):duration([(0,float("nan"))])
        with self.assertRaises(ValueError):parse_steps("-1")
        self.assertEqual(parse_steps("2,3,2"),{2,3})


class DispatcherTests(unittest.TestCase):
    def test_compiled_fake_children_route_errors_and_plan_lifetime(self):
        compiler=shutil.which("g++") or "D:/mingw64/bin/g++.exe"
        if not Path(compiler).is_file():self.fail("C++ compiler required for CPU ABI validation")
        assets=EXPERIMENT/"phase9/host-test-assets"
        assets.mkdir(parents=True,exist_ok=True)
        # Keep host-only fake DLLs until process exit; HIP module unloading is deliberately avoided.
        with nullcontext(Path(tempfile.mkdtemp(prefix="abi-",dir=assets))) as root:
            abi=SOURCE/"InfiniCore/src/infinicore/ops/multi_head_attention/hygon/h3_flash_abi.h"
            shutil.copyfile(abi,root/abi.name)
            def compile(name,source):
                p=root/(name+".cpp");p.write_text(source)
                output=root/(name+".so")
                command=[compiler,"-std=c++17","-O2","-shared","-I",str(root),str(p),"-o",str(output)]
                if os.name=="nt":command.extend(["-static-libstdc++","-static-libgcc"])
                else:command.extend(["-fPIC","-ldl","-pthread"])
                result=subprocess.run(command,capture_output=True,text=True)
                if result.returncode: raise RuntimeError(result.stderr)
                return output
            libs={}
            for marker,build in ((1,"baseline"),(2,"candidate")):
                body=(
                    '#include "h3_flash_abi.h"\n'
                    'int prepare(const H3FlashSpec*s,void**p){*p=nullptr;if(s->query_tokens==666)return 11;'
                    'if(s->strides[0][0]<0)return 13;*p=new int(7);return 0;}\n'
                    'int launch(void*p,const H3FlashCall*c,void*){if(c->scale<0)return 12;'
                    '*static_cast<int*>(c->out)=MARKER;return 0;}\n'
                    'void destroy(void*p){delete static_cast<int*>(p);}\n'
                    'static H3FlashApi api={1,sizeof(H3FlashApi),"gfx936","BUILD",prepare,launch,destroy};\n'
                    '#ifdef _WIN32\n#define EXPORT __declspec(dllexport)\n#else\n#define EXPORT\n#endif\n'
                    'extern "C" EXPORT const H3FlashApi*h3FlashGetApi(){return &api;}\n'
                ).replace("MARKER",str(marker)).replace("BUILD",build)
                path=compile(build,body);digest=sha(path.read_bytes())
                file=digest+".so";shutil.copyfile(path,root/file)
                libs[build]=dict(sha256=digest,file=file,build_id=build)
            baseline,candidate=libs["baseline"]["sha256"],libs["candidate"]["sha256"]
            policy=dict(libraries={baseline:libs["baseline"],candidate:libs["candidate"]},baseline=baseline,
                        rules=[dict(spec=shape_spec(65,33,False),library=candidate)])
            plugin=compile("dispatch",render(policy))
            if os.name=="nt" and hasattr(os,"add_dll_directory"):
                with os.add_dll_directory(str(Path(compiler).parent)): library=C.CDLL(str(plugin))
            else:library=C.CDLL(str(plugin))
            library.h3FlashGetApi.restype=C.POINTER(Api);api=library.h3FlashGetApi().contents
            def plan(q,stride_change=False):
                spec=Spec(1,q,33,7);values=shape_spec(q,33,False)
                for i,row in enumerate(values["strides"]):spec.strides[i][:]=row
                if stride_change:spec.strides[1][0]+=1
                pointer=C.c_void_p();rc=api.prepare(C.byref(spec),C.byref(pointer))
                return rc,pointer
            for q,changed,expected in ((65,False,2),(65,True,1),(66,False,1)):
                rc,pointer=plan(q,changed);self.assertEqual(rc,0)
                value=C.c_int();call=Call(None,None,None,C.cast(C.byref(value),C.c_void_p),1.0)
                self.assertEqual(api.launch(pointer,C.byref(call),None),0);self.assertEqual(value.value,expected)
                call.scale=-1;self.assertEqual(api.launch(pointer,C.byref(call),None),12)
                api.destroy(pointer)
            rc,pointer=plan(666);self.assertEqual(rc,11);self.assertFalse(pointer)
            # Concurrent planning must not race child initialization or unload cached modules.
            from concurrent.futures import ThreadPoolExecutor
            def repeated(_):
                rc,pointer=plan(65)
                if pointer:api.destroy(pointer)
                return rc
            with ThreadPoolExecutor(8) as pool:
                self.assertEqual(list(pool.map(repeated,range(32))),[0]*32)
            policy["libraries"][candidate]["build_id"]="wrong-build"
            bad=compile("wrong",render(policy));other=C.CDLL(str(bad));other.h3FlashGetApi.restype=C.POINTER(Api)
            pointer=C.c_void_p();spec=Spec(1,65,33,7)
            for i,row in enumerate(shape_spec(65,33,False)["strides"]):spec.strides[i][:]=row
            self.assertNotEqual(other.h3FlashGetApi().contents.prepare(C.byref(spec),C.byref(pointer)),0)
            policy["libraries"][candidate]["build_id"]="candidate"
            missing="0"*64
            policy["libraries"][missing]=dict(sha256=missing,file=missing+".so",build_id="missing")
            policy["rules"][0]["library"]=missing
            path=compile("missing",render(policy));absent=C.CDLL(str(path));absent.h3FlashGetApi.restype=C.POINTER(Api)
            pointer=C.c_void_p()
            self.assertNotEqual(absent.h3FlashGetApi().contents.prepare(C.byref(spec),C.byref(pointer)),0)
            self.assertFalse(pointer)
            self._loaded=(library,other,absent)



if __name__=="__main__":unittest.main(verbosity=2)
