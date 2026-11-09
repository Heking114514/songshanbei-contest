"""Measure all-reduce and all-to-all payload time; launch with torchrun."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-mib", default="16,64,256")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes_mib.split(",")]
    if min(sizes) < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("Invalid size or repetition count")
    import torch
    import torch.distributed as distributed
    if not torch.version.hip:
        raise RuntimeError("Platform HIP Torch is required")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    distributed.init_process_group("nccl", timeout=datetime.timedelta(minutes=5))
    rank, world = distributed.get_rank(), distributed.get_world_size()
    try:
        # Coordinate exclusive output checks so no rank waits after rank 0 exits.
        decision = torch.tensor([int(args.output.exists()) if rank == 0 else 0],
                                device="cuda")
        distributed.broadcast(decision, 0)
        if int(decision.item()):
            raise RuntimeError("Use a new output path")
        report = {"world_size": world, "torch": torch.__version__,
                  "hip": torch.version.hip, "cases": [], "complete": False,
                  "bandwidth_note": "payload bytes/time, not physical bus bandwidth"}
        for mib in sizes:
            elements = ((mib * 1024**2 // 2) // world) * world
            send = torch.full((elements,), rank, device="cuda", dtype=torch.bfloat16)
            receive = torch.empty_like(send)
            for operation in ("all_reduce", "all_to_all"):
                samples = []
                for iteration in range(args.warmup + args.repeats):
                    send.fill_(rank)
                    distributed.barrier()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if operation == "all_reduce":
                        distributed.all_reduce(send)
                    else:
                        distributed.all_to_all_single(receive, send)
                    torch.cuda.synchronize()
                    elapsed = torch.tensor([time.perf_counter() - start],
                                           device="cuda", dtype=torch.float64)
                    distributed.all_reduce(elapsed, op=distributed.ReduceOp.MAX)
                    if iteration >= args.warmup:
                        samples.append(float(elapsed.item()))
                if operation == "all_reduce":
                    valid = bool((send == world * (world - 1) / 2).all())
                else:
                    expected = torch.arange(world, device="cuda",
                                            dtype=torch.bfloat16)[:, None]
                    valid = bool((receive.view(world, -1) == expected).all())
                passed = torch.tensor([int(valid)], device="cuda")
                distributed.all_reduce(passed, op=distributed.ReduceOp.MIN)
                median = statistics.median(samples)
                entry = {"operation": operation, "payload_bytes": elements * 2,
                         "slowest_rank_seconds": samples, "median_seconds": median,
                         "payload_gib_per_second": elements * 2 / 1024**3 / median,
                         "values_verified": bool(passed.item())}
                report["cases"].append(entry)
                if rank == 0:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(entry), flush=True)
                if not entry["values_verified"]:
                    raise RuntimeError("Communication output validation failed")
            del send, receive
        if rank == 0:
            report["complete"] = True
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
