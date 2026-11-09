"""Native hipprof collection and strict OPS/STR_TABLE timeline analysis.

Default collect mode writes a local plan only. Explicit execution runs hipprof
as a process-level tracer, including native worker threads and graph replays.
Stats summaries and CPU-only traces cannot satisfy expected GPU coverage.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo
from phase8_trace import private_output
from phase9_profile import summarize


def stamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def device_ids(text):
    result = [int(x) for x in text.split(",")]
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise ValueError("Expected devices must be unique nonnegative physical IDs")
    return result


def identifier(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
        raise ValueError("Unsupported SQL identifier")
    return '"' + value + '"'


def read_ops(path, table=None):
    path = Path(path).resolve()
    before = sha(path)
    # mode=ro keeps the raw evidence database read-only (including on Windows).
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"BeginNs", "EndNs", "queue_id", "Name", "dev_id"}
        supported = []
        for name in sorted(tables):
            if not name.startswith("OPS_"):
                continue
            columns = {r[1] for r in db.execute("PRAGMA table_info(" + identifier(name) + ")")}
            if required <= columns:
                supported.append(name)
        if table is None:
            if len(supported) != 1:
                raise ValueError("Require exactly one supported OPS table, or explicit --ops-table")
            table = supported[0]
        elif table not in supported:
            raise ValueError("Selected OPS table has unsupported schema")
        if "STR_TABLE" not in tables:
            raise ValueError("Missing STR_TABLE; stats-only reports are insufficient")
        strings = {}
        for key, name in db.execute("SELECT STR_ID, STR_NAME FROM STR_TABLE"):
            if str(key) in strings or not isinstance(name, str) or not name:
                raise ValueError("Invalid or duplicate trace string")
            strings[str(key)] = name
        events = []
        sql = "SELECT BeginNs, EndNs, queue_id, Name, dev_id FROM " + identifier(table)
        for begin, end, stream, name, device in db.execute(sql):
            if any(isinstance(v, bool) or not isinstance(v, int)
                   for v in (begin, end, stream, device)):
                raise ValueError("hipprof requires integer nanosecond timestamps and IDs")
            if begin < 0 or end <= begin or device < 0 or stream < 0:
                raise ValueError("Invalid native operation interval or device/queue")
            if str(name) not in strings:
                raise ValueError("Unresolved trace string ID")
            text = strings[str(name)]
            # OPS includes device copies too. Do not count those as GPU kernels.
            category = ("gpu_memcpy" if re.search(r"memcpy|copy.*[HD]to[HD]", text, re.I) else
                        "gpu_memset" if "memset" in text.lower() else "kernel")
            events.append(dict(begin_ns=begin, end_ns=end, device=device,
                               stream=stream, name=text, cat=category))
    if sha(path) != before:
        raise ValueError("Trace database changed during analysis")
    return events, dict(path=str(path), sha256=before, ops_table=table, events=len(events))


def analyze(paths, expected, table=None):
    if not expected or len(set(expected)) != len(expected) or any(x < 0 for x in expected):
        raise ValueError("Explicit unique expected GPU IDs required")
    events, sources, identities = [], [], set()
    for path in paths:
        rows, source = read_ops(path, table)
        if any(s["sha256"] == source["sha256"] for s in sources):
            raise ValueError("Duplicate raw database")
        sources.append(source)
        for row in rows:
            key = tuple(row[k] for k in ("device", "stream", "begin_ns", "end_ns", "name", "cat"))
            if key in identities:
                raise ValueError("Duplicate device operation across trace sources")
            identities.add(key)
        events.extend(rows)
    origin = min((r["begin_ns"] for r in events), default=0)
    trace = {"traceEvents": [dict(ph="X", cat=r["cat"], name=r["name"],
        ts=(r["begin_ns"] - origin) / 1000, dur=(r["end_ns"] - r["begin_ns"]) / 1000,
        args={"device": r["device"], "stream": r["stream"]}) for r in events]}
    report = summarize(trace)
    missing = [d for d in expected if not report["devices"].get(str(d), {}).get("kernel_events")]
    unexpected = sorted({r["device"] for r in events} - set(expected))
    report.update(at=stamp(), source_format="hipprof-OPS-nanoseconds", sources=sources,
                  timestamp_origin_ns=origin, expected_device_ids=expected,
                  missing_device_ids=missing, unexpected_device_ids=unexpected,
                  complete=bool(events) and not missing and not unexpected,
                  model_acceptance=False, gpu_performance_validated=False)
    report["limitations"].extend([
        "Complete means trace-device coverage, not model correctness or speedup.",
        "Device IDs must be physical IDs in one trace namespace; no pid/device fallback.",
        "Kernel names do not identify GEMM dimensions; join with the actual shape trace.",
        "This trace covers the child command; do not infer denoise-step boundaries from gaps."])
    return report


def relay(command, output):
    started = time.monotonic()
    with (output / "trace.log").open("x", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(command, cwd=output, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
            env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True)
        lines = queue.Queue()
        def reader():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            while True:
                try:
                    line = lines.get(timeout=15)
                except queue.Empty:
                    heartbeat = "PHASE10_PROFILE_HEARTBEAT %s %.1f seconds" % (
                        stamp(), time.monotonic()-started)
                    log.write(heartbeat+"\n")
                    log.flush()
                    print(heartbeat, flush=True)
                    continue
                if line is None:
                    break
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            process.stdout.close()
            thread.join(timeout=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("collect", "analyze"):
        parser = sub.add_parser(name)
        parser.add_argument("--workspace-root", required=True, type=Path)
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
        parser.add_argument("--ops-table")
    collect = sub.choices["collect"]
    collect.add_argument("--execute-trace", action="store_true")
    collect.add_argument("--hipprof", type=Path, default=Path("/opt/dtk/bin/hipprof"))
    collect.add_argument("command", nargs=argparse.REMAINDER)
    sub.choices["analyze"].add_argument("databases", type=Path, nargs="+")
    args = ap.parse_args()
    expected = device_ids(args.devices)
    output = private_output(args.workspace_root, args.output)
    if args.mode == "analyze":
        if output.exists():
            ap.error("Choose a fresh report path")
        report = analyze(args.databases, expected, args.ops_table)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        if not report["complete"]:
            raise RuntimeError("Native GPU trace coverage incomplete; inspect report")
        return
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("Pass the frozen replay/diagnostic command after --")
    if args.execute_trace and (sys.platform != "linux" or not args.hipprof.is_file()):
        ap.error("Execution requires Linux with the actual hipprof executable")
    output.mkdir(parents=True, exist_ok=False)
    invocation = [str(args.hipprof), "--hip-trace", "--stats", "-o", "native", *command]
    report = dict(at=stamp(), execute_trace=args.execute_trace, command=invocation,
                  expected_device_ids=expected, complete=False, afs_modified=False,
                  model_acceptance=False, source_sha256=sha(__file__))
    (output/"plan.json").write_text(json.dumps(report, indent=2)+"\n")
    if not args.execute_trace:
        print("PHASE10_NATIVE_PROFILE_PLAN_ONLY", output, flush=True)
        return
    try:
        print("PHASE10_NATIVE_PROFILE_START", json.dumps(report), flush=True)
        report["returncode"] = relay(invocation, output)
        if report["returncode"]:
            raise RuntimeError("hipprof or child exited with nonzero status")
        databases = sorted({*output.glob("*.db"), *output.glob("*.sqlite")})
        if not databases:
            raise RuntimeError("No raw hipprof database; stats alone cannot prove overlap")
        summary = analyze(databases, expected, args.ops_table)
        (output/"profile.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        if not summary["complete"]:
            raise RuntimeError("Native GPU trace coverage incomplete")
        report["complete"] = True
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["finished_at"] = stamp()
        (output/"run.json").write_text(json.dumps(report, indent=2)+"\n")
        print("PHASE10_NATIVE_PROFILE_END", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
