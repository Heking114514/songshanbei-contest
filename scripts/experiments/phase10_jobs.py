"""Budgeted immutable job attempts, streaming logs and SHA-verified resume.

No platform API or shell commands. The coordinator authorizes GPU execution;
this module also supports CPU child processes in local contract tests.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from phase10_profile import sha, stamp


def atomic(path, data):
    temp = path.with_name(path.name+".tmp")
    temp.write_bytes((json.dumps(data, indent=2, allow_nan=False)+"\n").encode())
    temp.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def files_in(directory):
    return {str(p.relative_to(directory)).replace(os.sep, "/"):sha(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def verify_files(directory, manifest):
    if not manifest: raise ValueError("A passed job needs sealed raw artifacts")
    if files_in(directory) != manifest:
        raise ValueError("Completed job artifacts changed; refusing cached success")


@contextmanager
def exclusive(output):
    # Kernel-owned flock automatically unlocks even if the coordinator crashes.
    # Execution mode is Linux-only; no stale PID marker needs unsafe deletion.
    import fcntl
    with (output/"coordinator.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another coordinator owns this output directory")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stop(process):
    if process.poll() is not None: return
    if os.name == "posix":
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    else:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        else: process.kill()
        process.wait()


def logged(command, logfile, timeout, env, label, heartbeat=15):
    started = time.monotonic()
    print("PHASE10_JOB_START", label, json.dumps(command), flush=True)
    with logfile.open("x", encoding="utf-8", newline="\n") as output:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env={**os.environ, **env, "PYTHONUNBUFFERED":"1", "PYTHONDONTWRITEBYTECODE":"1"},
            start_new_session=(os.name == "posix"))
        lines = queue.Queue()
        def reader():
            try:
                for line in process.stdout: lines.put(line)
            finally:
                lines.put(None)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            while True:
                remaining = timeout-(time.monotonic()-started)
                if remaining <= 0: raise TimeoutError("Job exceeded %.1fs" % timeout)
                try:
                    line = lines.get(timeout=min(heartbeat, remaining))
                except queue.Empty:
                    line = "PHASE10_JOB_HEARTBEAT %s %s %.1fs\n" % (label, stamp(), time.monotonic()-started)
                if line is None:
                    break
                output.write(line)
                output.flush()
                print("["+label+"] "+line, end="", flush=True)
            try:
                return process.wait(timeout=max(.001, timeout-(time.monotonic()-started)))
            except subprocess.TimeoutExpired:
                raise TimeoutError("Child remained active after closing stdout")
        finally:
            stop(process)
            process.stdout.close()
            thread.join(timeout=2)


class Jobs:
    def __init__(self, output, plan, budget, reserve=180, resume=False, retry_failed=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output/"ledger.json"
        self.lock = threading.RLock()
        self.deadline = time.monotonic()+budget
        self.reserve = reserve
        self.phase_deadline = None
        self.retry_failed = retry_failed
        if budget <= reserve or reserve < 0: raise ValueError("Budget must exceed recovery reserve")
        if self.path.exists():
            if not resume: raise ValueError("Existing ledger requires --resume")
            self.data = json.loads(self.path.read_text())
            if self.data["plan_identity"] != identity(plan):
                raise ValueError("Source/input/platform/plan changed; choose a fresh output")
            for task in self.data["tasks"].values():
                for attempt in task["attempts"]:
                    if attempt["state"] == "running":
                        attempt.update(state="interrupted", finished_at=stamp())
        else:
            self.data = dict(schema=1, plan_identity=identity(plan), plan=plan, tasks={}, complete=False)
        self.data["sessions"] = self.data.get("sessions", []) + [dict(at=stamp(), budget_seconds=budget,
                                                                     reserve_seconds=reserve)]
        self.save()

    def remaining(self):
        deadline = self.deadline-self.reserve
        if self.phase_deadline is not None: deadline = min(deadline, self.phase_deadline)
        return max(0, deadline-time.monotonic())

    def save(self):
        with self.lock:
            self.data["updated_at"] = stamp()
            atomic(self.path, self.data)

    def run(self, key, command, validate, timeout, estimate=1, env=None):
        if not key or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in key):
            raise ValueError("Unsafe job identity")
        env = dict(env or {})
        with self.lock:
            task = self.data["tasks"].setdefault(key, dict(attempts=[]))
            # A successful receipt is never trusted just because its flag says passed.
            if task["attempts"] and task["attempts"][-1]["state"] == "passed":
                old = task["attempts"][-1]
                path = self.output/old["directory"]
                expected = identity(dict(command=command(path), env=env))
                if old["command_identity"] != expected: raise ValueError("Cached command changed")
                verify_files(path, old["files"])
                result = validate(path)
                print("PHASE10_REUSE_VERIFIED", key, flush=True)
                return dict(directory=path, result=result, reused=True)
            if task["attempts"] and task["attempts"][-1]["state"] == "failed" and not self.retry_failed:
                print("PHASE10_RETAIN_FAILED", key, flush=True)
                return None
            if self.remaining() < estimate:
                task["deferred"] = dict(at=stamp(), state="skipped_budget", required_seconds=estimate,
                                       available_seconds=self.remaining())
                self.save()
                print("PHASE10_SKIP_BUDGET", key, flush=True)
                return None
            directory = self.output/"jobs"/key/("attempt-%03d" % len(task["attempts"]))
            directory.mkdir(parents=True, exist_ok=False)
            argv = command(directory)
            attempt = dict(at=stamp(), state="running", directory=str(directory.relative_to(self.output)),
                           command=argv, env=env, command_identity=identity(dict(command=argv, env=env)))
            task["attempts"].append(attempt)
            task.pop("deferred", None)
            self.save()
        started = time.monotonic()
        try:
            rc = logged(argv, directory/"terminal.log", min(timeout, self.remaining()), env, key)
            if rc: raise RuntimeError("Child exit %s" % rc)
            result = validate(directory)
            with self.lock:
                attempt.update(state="passed", result=result)
        except BaseException as exc:
            with self.lock:
                attempt.update(state="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                               error=repr(exc))
            print("PHASE10_JOB_FAILURE", key, repr(exc), flush=True)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)): raise
            result = None
        finally:
            sealed=files_in(directory)
            with self.lock:
                attempt.update(finished_at=stamp(), outer_seconds=time.monotonic()-started, files=sealed)
                self.save()
        return dict(directory=directory, result=result, reused=False) if result is not None else None


def lanes(devices, rows, worker):
    # One sequential lane per physical device: no two local kernels contend on
    # the same GPU. Full TP8 phases run only after every lane returns.
    assignments = {device:rows[i::len(devices)] for i,device in enumerate(devices)}
    def lane(device):
        return [worker(device, row) for row in assignments[device]]
    with ThreadPoolExecutor(len(devices)) as pool:
        return [result for group in pool.map(lane, devices) for result in group]
