"""CPU contract checks; kernel bodies execute with a bounds-checking Torch shim.

These checks cover math/control-flow/stride semantics, not Triton code generation,
GPU graph safety or GPU performance. No server, platform or AFS operations.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from contextlib import closing, contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase10_profile import analyze, read_ops, device_ids
from phase10_gemm_timing import bracketed, paired_gain
from phase6_gemm_select import select
from phase9_profile import summarize
from phase10_attention_worker import verified_baseline

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[2]


@dataclass
class Pointer:
    data: torch.Tensor
    offsets: object = 0

    def __add__(self, offsets):
        return Pointer(self.data, self.offsets + offsets)


class Language(ModuleType):
    int32 = torch.int32
    int64 = torch.int64
    float32 = torch.float32
    bfloat16 = torch.bfloat16
    constexpr = object()
    grid = (0, 0, 0)
    arange = staticmethod(torch.arange)
    exp2 = staticmethod(torch.exp2)
    maximum = staticmethod(torch.maximum)
    where = staticmethod(torch.where)

    def program_id(self, axis):
        return self.grid[axis]

    def cdiv(self, a, b):
        return (a+b-1)//b

    def full(self, shape, value, dtype):
        return torch.full(shape, value, dtype=dtype)

    def zeros(self, shape, dtype):
        return torch.zeros(shape, dtype=dtype)

    def max(self, value, axis):
        return value.max(dim=axis).values

    def sum(self, value, axis):
        return value.sum(dim=axis)

    def dot(self, a, b, accumulator=None):
        out = a.float() @ b.float()
        return out if accumulator is None else out + accumulator

    def load(self, pointer, mask=None, other=0):
        offsets = torch.as_tensor(pointer.offsets, dtype=torch.int64)
        active = torch.ones_like(offsets, dtype=torch.bool) if mask is None else torch.broadcast_to(mask, offsets.shape)
        indices = offsets[active]
        if indices.numel() and (indices.min() < 0 or indices.max() >= pointer.data.numel()):
            raise IndexError("Kernel accessed outside tensor allocation")
        result = torch.full(offsets.shape, other, dtype=pointer.data.dtype)
        result[active] = pointer.data[indices]
        return result

    def store(self, pointer, value, mask=None):
        offsets = torch.as_tensor(pointer.offsets, dtype=torch.int64)
        active = torch.ones_like(offsets, dtype=torch.bool) if mask is None else torch.broadcast_to(mask, offsets.shape)
        indices = offsets[active]
        if indices.numel() and (indices.min() < 0 or indices.max() >= pointer.data.numel()):
            raise IndexError("Kernel stored outside tensor allocation")
        pointer.data[indices] = value[active]


def load_kernel(name, language):
    triton = ModuleType("triton")
    triton.jit = lambda function: function
    triton.language = language
    with patch.dict(sys.modules, {"triton": triton, "triton.language": language}):
        spec = importlib.util.spec_from_file_location(name, HERE / (name+".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.h3_flash_gfx936


def backing(view):
    data = torch.as_strided(view, (view.untyped_storage().nbytes()//view.element_size(),), (1,), storage_offset=0)
    return Pointer(data, view.storage_offset())


def execute(kernel, language, q, k, v, out, bm, bn, prefetch=None):
    values = [backing(x) for x in (q, k, v, out)]
    values += [q.shape[1], k.shape[1], q.shape[2], 128**-0.5]
    for x in (q, k, v, out):
        values.extend(x.stride()[:3])
    values.extend((bm, bn))
    if prefetch is not None:
        values.append(prefetch)
    for batch in range(q.shape[0]):
        for head in range(q.shape[2]):
            for tile in range((q.shape[1]+bm-1)//bm):
                language.grid = (tile, head, batch)
                kernel(*values)


class AttentionStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.language = Language("triton.language")
        cls.original = staticmethod(load_kernel("phase7_flash_kernel", cls.language))
        cls.candidate = staticmethod(load_kernel("phase10_flash_kernel", cls.language))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def exercise(self, nk, bn, layout, prefetch):
        torch.manual_seed(1800+nk)
        b, nq, h, d = 2, 35, 2, 128
        q = torch.randn(b, nq, h, d).bfloat16()
        k = torch.randn(b, nk, h, d).bfloat16()
        packed = torch.randn(b, nk, h*3, d).bfloat16()
        if layout == "mixed-v":
            v = packed[:, :, h*2:]
        else:
            v = torch.randn(b, h, nk, d).bfloat16().transpose(1, 2)
        stores = [torch.full((b, nq, h*2, d), 8192, dtype=torch.bfloat16) for _ in range(2)]
        expected, actual = [x[:, :, :h] for x in stores]
        execute(self.original, self.language, q, k, v, expected, 32, bn)
        execute(self.candidate, self.language, q, k, v, actual, 32, bn, prefetch)
        self.assertTrue(torch.equal(expected, actual), (nk, bn, layout, prefetch))
        self.assertTrue(torch.isfinite(actual).all())
        for storage in stores:
            self.assertTrue(torch.all(storage[:, :, h:] == 8192), "Output padding modified")
        dense = torch.softmax((q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-1, -2))
                              * (128**-0.5), dim=-1) @ v.transpose(1, 2).float()
        error = (actual.transpose(1, 2).float()-dense).norm()/dense.norm()
        self.assertLess(float(error), .01)

    def test_short_exact_and_partial_tail_use_actual_kernel_bodies(self):
        for bn in (32, 64, 128):
            for nk in (1, bn-1, bn, bn+1, 2*bn, 2*bn+7):
                for prefetch in (False, True):
                    with self.subTest(bn=bn, nk=nk, prefetch=prefetch):
                        self.exercise(nk, bn, "mixed-v", prefetch)

    def test_bhsd_values_batch_head_and_output_padding(self):
        for prefetch in (False, True):
            self.exercise(97, 32, "bhsd", prefetch)

    def test_previous_best_bundle_is_pinned_and_tampering_rejected(self):
        bundle = EXPERIMENT / "phase9_live3/evidence/phase9-dispatch-mixed"
        verified_baseline(bundle)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/"bundle.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "immutable"):
                verified_baseline(path)


@contextmanager
def fixture_connection(path):
    with closing(sqlite3.connect(path)) as db:
        with db:
            yield db


def database(path, devices=(0, 1), table="OPS_0", memory_only=False):
    with fixture_connection(path) as db:
        db.execute("CREATE TABLE STR_TABLE(STR_ID INTEGER, STR_NAME TEXT)")
        db.executemany("INSERT INTO STR_TABLE VALUES(?,?)",
                       [(1, "gemm_native"), (2, "rcclAllReduce"), (3, "hipMemcpyAsync")])
        db.execute('CREATE TABLE "' + table + '" (BeginNs INTEGER, EndNs INTEGER, queue_id INTEGER, Name INTEGER, dev_id INTEGER)')
        for device in devices:
            rows = [(1_000_000, 1_100_000, 0, 3 if memory_only else 1, device),
                    (1_050_000, 1_150_000, 1, 3 if memory_only else 2, device)]
            db.executemany('INSERT INTO "' + table + '" VALUES(?,?,?,?,?)', rows)


class NativeProfileTests(unittest.TestCase):
    def test_eight_device_coverage_and_overlap_union(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"native.db"
            database(path, tuple(range(8)))
            report = analyze([path], list(range(8)))
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["devices"]), 8)
            for device in report["devices"].values():
                self.assertAlmostEqual(device["busy_union_ms"], .15)
                self.assertAlmostEqual(device["communication_compute_overlap_ms"], .05)
                self.assertAlmostEqual(device["exposed_communication_ms"], .05)
            self.assertIsNone(report["wall_clock_speedup"])
            self.assertFalse(report["model_acceptance"])

    def test_missing_unexpected_and_memory_only_never_complete(self):
        for devices, memory in (((0,), False), ((0, 1, 2), False), ((0, 1), True)):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"native.db"
                database(path, devices, memory_only=memory)
                self.assertFalse(analyze([path], [0, 1])["complete"])

    def test_cpu_only_stats_and_missing_schema_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"summary.db"
            with fixture_connection(path) as db:
                db.execute("CREATE TABLE stats(name TEXT, total INTEGER)")
            with self.assertRaisesRegex(ValueError, "OPS"):
                analyze([path], [0])

    def test_duplicate_databases_and_unresolved_string_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"native.db"
            database(path)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                analyze([path, path], [0, 1])
            with fixture_connection(path) as db:
                db.execute("UPDATE OPS_0 SET Name=99")
            with self.assertRaisesRegex(ValueError, "Unresolved"):
                read_ops(path)

    def test_bad_interval_and_ambiguous_tables_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"native.db"
            database(path)
            with fixture_connection(path) as db:
                db.execute("CREATE TABLE OPS_1 AS SELECT * FROM OPS_0")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                read_ops(path)
            self.assertTrue(analyze([path], [0, 1], "OPS_0")["complete"])
            with fixture_connection(path) as db:
                db.execute("UPDATE OPS_0 SET EndNs=BeginNs-1")
            with self.assertRaisesRegex(ValueError, "interval"):
                read_ops(path, "OPS_0")

    def test_pid_is_not_an_implicit_gpu_device(self):
        event = dict(ph="X", cat="kernel", name="gemm", pid=0, ts=0, dur=3)
        with self.assertRaisesRegex(ValueError, "pid"):
            summarize(dict(traceEvents=[event]))

    def test_device_id_validation(self):
        self.assertEqual(device_ids("0,3,7"), [0, 3, 7])
        for text in ("", "0,0", "-1", "0,a"):
            with self.assertRaises(ValueError):
                device_ids(text)


class PairedGemmTests(unittest.TestCase):
    def item(self, before=10., candidate=8., after=10.):
        return dict(median_gpu_ms=candidate, paired=dict(schema=1, warmup=3, samples=[
            dict(baseline_before_ms=before, baseline_after_ms=after, candidate_gpu_ms=candidate,
                 candidate_wall_seconds=.01, baseline_bitwise=True, candidate_bitwise=True)
            for _ in range(3)]))

    def test_real_scheduler_order_and_per_sample_checks(self):
        events = []
        def run(name, value):
            def call():
                events.append(name)
                return value, .01
            return call
        pair = bracketed(run("B", 10.), run("C", 8.),
                         lambda: events.append("b") or True,
                         lambda: events.append("c") or True, 3, 3)
        self.assertEqual(events[:12], ["B", "b", "C", "c"]*3)
        self.assertEqual(events[12:], ["B", "b", "C", "c", "B", "b"]*3)
        self.assertAlmostEqual(paired_gain(dict(median_gpu_ms=8., paired=pair)), .2)

    def test_local_faster_baseline_used_and_drift_rejected(self):
        self.assertAlmostEqual(paired_gain(self.item(before=10., after=10.5)), .2)
        with self.assertRaisesRegex(ValueError, "drift"):
            paired_gain(self.item(before=5., after=10.))

    def test_bit_change_invalid_and_unbound_median_rejected(self):
        for change in ("bits", "nan", "median", "missing"):
            item = self.item()
            if change == "bits": item["paired"]["samples"][0]["candidate_bitwise"] = False
            if change == "nan": item["paired"]["samples"][0]["baseline_after_ms"] = float("nan")
            if change == "median": item["median_gpu_ms"] = 1.
            if change == "missing": item["paired"]["samples"] = []
            with self.assertRaises(ValueError, msg=change):
                paired_gain(item)

    def report(self):
        key = "h3lt1:"+("a"*64)+":device:shape"
        cases = []
        for alpha, beta in ((1., 0.), (.75, 0.), (.75, .25), (1., 1.)):
            base = dict(index=-1, median_gpu_ms=20., repeat=dict(finite=True))
            candidate = self.item()
            candidate.update(index=0, create_status=0, strict_eligible=True,
                selection=dict(backend="lt", index=0, key=key),
                repeat=dict(finite=True, bitwise_equal=True), all_repeats_bitwise_equal=True,
                vs_baseline=dict(finite=True, bitwise_equal=True), unaligned_rejected_before_write=True)
            cases.append(dict(alpha=alpha, beta=beta, baseline_repeat_bitwise=True, candidates=[base, candidate]))
        return dict(complete=True, lt_executed=True, timing_mode="paired", runtime_tag="a"*64, cases=cases)

    def test_strict_selector_uses_pairs_not_old_remote_baseline(self):
        a, b = self.report(), self.report()
        chosen = select([a, b], require_paired=True)
        self.assertEqual(len(chosen), 1)
        self.assertAlmostEqual(next(iter(chosen.values()))["mean_gain"], .2)
        # Old aggregate says 60% faster; adjacent baselines show a slowdown.
        for report in (a, b):
            for case in report["cases"]:
                for sample in case["candidates"][1]["paired"]["samples"]:
                    sample["baseline_before_ms"] = sample["baseline_after_ms"] = 7.
        self.assertFalse(select([a, b], require_paired=True))

    def test_selection_requires_independent_reports_all_scalars_and_new_timing(self):
        a, b = self.report(), self.report()
        self.assertFalse(select([a], require_paired=True))
        b["cases"].pop()
        self.assertFalse(select([a, b], require_paired=True))
        b = self.report()
        b["timing_mode"] = "serial"
        with self.assertRaisesRegex(ValueError, "Paired"):
            select([a, b], require_paired=True)


class PlanOnlyTests(unittest.TestCase):
    def launch(self, script, args, folder):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, str(HERE/script), *args],
                              cwd=folder, env=env, capture_output=True, text=True, encoding="utf-8")

    def test_aot_plan_has_no_triton_import_or_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root/"aot"
            result = self.launch("phase10_build_flash.py",
                ["--workspace-root", str(root), "--output", str(output), "--prefetch-v"], folder)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((output/"build.json").read_text())
            self.assertFalse(plan["complete"])
            self.assertFalse(plan["execute_build"])
            self.assertTrue(plan["config"]["prefetch_v"])
            self.assertEqual((plan["config"]["block_m"], plan["config"]["block_n"]), (128, 32))
            self.assertEqual(set(x.name for x in output.iterdir()), {"build.json"})

    def test_native_collect_default_does_not_launch_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root/"profile"
            marker = root/"child-was-executed"
            result = self.launch("phase10_profile.py",
                ["collect", "--workspace-root", str(root), "--output", str(output),
                 "--", sys.executable, "-c", "open("+repr(str(marker))+", 'w').write('bad')"], folder)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(marker.exists())
            self.assertEqual(set(x.name for x in output.iterdir()), {"plan.json"})
            self.assertFalse(json.loads((output/"plan.json").read_text())["execute_trace"])

    def test_native_analysis_keeps_failure_receipt_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/"native.db"
            database(path, devices=(0,))
            output = root/"summary.json"
            result = self.launch("phase10_profile.py",
                ["analyze", "--workspace-root", str(root), "--output", str(output), str(path)], folder)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads(output.read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["missing_device_ids"], list(range(1, 8)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
