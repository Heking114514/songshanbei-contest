"""Diagnostic tensor records. Full hashes; bounded raw dumps; timings are not benchmarks."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time


def private_output(work, output):
    work, output = Path(work).resolve(), Path(output).resolve()
    afs = os.environ.get("IFAB_AFS_HOME")
    if output == work or not output.is_relative_to(work) or (
            afs and any(p.is_relative_to(Path(afs).resolve()) for p in (work, output))):
        raise ValueError("Use a new output subdirectory in private storage outside AFS")
    return output


@contextmanager
def native_directory(directory):
    key = "H3_PHASE8_NATIVE_TRACE_DIR"
    old = os.environ.pop(key, None)
    try:
        if directory is not None:
            directory.mkdir(exist_ok=False)
            os.environ[key] = str(directory.resolve())
        yield
    finally:
        os.environ.pop(key, None)
        if old is not None:
            os.environ[key] = old


class TensorTrace:
    def __init__(self, root, budget_mib=256, tensor_mib=16):
        self.root = Path(root)
        if budget_mib <= 0 or tensor_mib <= 0:
            raise ValueError("Trace budgets must be positive")
        self.root.mkdir(exist_ok=False)
        self.log = (self.root/"trace.jsonl").open("x", encoding="utf-8", newline="\n")
        self.budget, self.tensor_limit = budget_mib * 1024**2, tensor_mib * 1024**2
        self.used, self.sequence, self.counts = 0, 0, {}
        self.emit(dict(kind="header", schema=1, diagnostic_only=True,
                       hash="sha256/full tensor", budget_bytes=self.budget))

    def emit(self, row):
        self.log.write(json.dumps(row, allow_nan=False)+"\n"); self.log.flush()

    def record(self, stage, value, step=None, dump=False):
        import torch
        if torch.is_tensor(value):
            tensor = value.detach().contiguous().cpu()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            if len(raw) != tensor.numel() * tensor.element_size():
                raise RuntimeError("Tensor byte count mismatch")
            key = (step, stage); occurrence = self.counts.get(key, 0)
            self.counts[key] = occurrence + 1
            row = dict(kind="tensor", sequence=self.sequence, step=step, stage=stage,
                       occurrence=occurrence, dtype=str(tensor.dtype).removeprefix("torch."),
                       shape=list(tensor.shape), bytes=len(raw), byteorder="little",
                       sha256=hashlib.sha256(raw).hexdigest(), dump=None)
            self.sequence += 1
            finite = bool(torch.isfinite(tensor).all()) if tensor.is_floating_point() else True
            row["finite"] = finite
            if finite and tensor.numel():
                # Bound temporary memory independently of the tensor/dump budget.
                total, maximum = 0.0, 0.0
                for chunk in tensor.reshape(-1).split(262144):
                    x = chunk.double(); total += float(x.square().sum())
                    maximum = max(maximum, float(x.abs().max()))
                row["l2_norm"], row["max_abs"] = math.sqrt(total), maximum
            if dump:
                if len(raw) > self.tensor_limit:
                    row["dump_skipped"] = "per_tensor_limit"
                elif self.used + len(raw) > self.budget:
                    row["dump_skipped"] = "total_budget"
                else:
                    name = f"{row['sequence']:06d}.bin"
                    (self.root/name).write_bytes(raw); self.used += len(raw); row["dump"] = name
            self.emit(row)
            return
        if is_dataclass(value):
            value = {f.name: getattr(value, f.name) for f in fields(value)}
        if isinstance(value, dict):
            for key in sorted(value):
                self.record(stage+"."+str(key), value[key], step, dump)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                self.record(stage+"."+str(i), item, step, dump)
        elif hasattr(value, "sample"):
            self.record(stage+".sample", value.sample, step, dump)

    def tensor_snapshot(self, name, tensor):
        import torch
        path = self.root/(name+".pt")
        if path.exists():
            raise RuntimeError("Refusing to overwrite a tensor snapshot")
        temp = path.with_suffix(".pt.partial")
        torch.save(tensor.detach().cpu().clone(), temp)
        size = temp.stat().st_size
        if self.used + size > self.budget:
            temp.unlink()
            raise RuntimeError("Tensor snapshot exceeds diagnostic dump budget")
        temp.rename(path); self.used += size
        self.emit(dict(kind="frozen_tensor", path=path.name, bytes=size,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def snapshot(self, step, inputs):
        import torch
        values = {f.name: getattr(inputs, f.name).detach().cpu().clone() for f in fields(inputs)}
        path = self.root/f"dit-input-step-{step:03d}.pt"
        if path.exists():
            raise RuntimeError("Refusing to overwrite a frozen input")
        temp = path.with_suffix(".pt.partial")
        torch.save(values, temp)
        size = temp.stat().st_size
        if self.used + size > self.budget:
            temp.unlink()
            raise RuntimeError("Frozen input exceeds diagnostic dump budget")
        temp.rename(path); self.used += size
        self.emit(dict(kind="frozen_input", step=step, path=path.name, bytes=size,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def close(self, complete):
        self.emit(dict(kind="end", complete=complete, tensor_events=self.sequence, dumped_bytes=self.used))
        self.log.close()


class RunnerTrace:
    def __init__(self, writer, native_steps=(), capture_steps=()):
        self.writer = writer
        self.native_steps, self.capture_steps = set(native_steps), set(capture_steps)
        self.step = 0
        self.native_seen = []
        self.latencies = []

    def install(self, cls, devices):
        import torch
        original = cls.__call__
        def sync():
            for device in devices:
                torch.cuda.synchronize(device)
        def traced(runner, inputs, *args, **kwargs):
            step = self.step; self.step += 1
            if self.writer:
                self.writer.record("dit.input", inputs, step, dump=False)
                self.writer.record("dit.audio_input", inputs.audio_hidden_states, step, dump=True)
                if step in self.capture_steps:
                    self.writer.snapshot(step, inputs)
            directory = self.writer.root/f"native-step-{step:03d}" if self.writer and step in self.native_steps else None
            sync(); start = time.perf_counter()
            with native_directory(directory):
                result = original(runner, inputs, *args, **kwargs)
            sync(); elapsed = time.perf_counter()-start
            self.latencies.append(elapsed)
            if directory:
                rows = [json.loads(p.read_text()) for p in directory.glob("*.json")]
                count = int(runner.engine.hf_config.get("num_layers", 50))
                expected = {"audio_input", "audio_embeds", "packed_audio", "norm_out_audio", "audio_output",
                            *(f"block_{i}_audio" for i in range(count))}
                if len(rows) != len(expected) or {r["stage"] for r in rows} != expected:
                    raise RuntimeError("Missing native trace stages: incrementally rebuild the H3 transformer object")
                self.native_seen.append(step)
            if self.writer:
                self.writer.record("dit.video_output", result.sample, step, dump=False)
                self.writer.record("dit.audio_output", result.audio_sample, step, dump=True)
            print("PHASE8_DIT_STEP", step, elapsed, "diagnostic" if self.writer else "timed", flush=True)
            return result
        cls.__call__ = traced
        return lambda: setattr(cls, "__call__", original)
