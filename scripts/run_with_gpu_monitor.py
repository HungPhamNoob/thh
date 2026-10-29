#!/usr/bin/env python3
"""Run a command while recording actual NVIDIA VRAM usage for its process tree."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _nvidia_query(arguments: list[str]) -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _descendants(root: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2 :].split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    selected = {root}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def _used_memory(pids: set[int]) -> int:
    lines = _nvidia_query(
        ["--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"]
    )
    total = 0
    for line in lines:
        try:
            raw_pid, raw_memory = (part.strip() for part in line.split(",", 1))
            if int(raw_pid) in pids:
                total += int(raw_memory)
        except ValueError:
            continue
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    gpu_before = _nvidia_query(
        ["--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"]
    )
    started = datetime.now(timezone.utc)
    process = subprocess.Popen(command)
    peak_mib = 0
    samples = 0
    while process.poll() is None:
        peak_mib = max(peak_mib, _used_memory(_descendants(process.pid)))
        samples += 1
        time.sleep(max(0.05, args.interval))
    peak_mib = max(peak_mib, _used_memory(_descendants(process.pid)))
    report = {
        "format_version": 1,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "gpu_before": gpu_before,
        "peak_process_tree_vram_mib": peak_mib,
        "limit_mib": 4096,
        "target_mib": 3584,
        "samples": samples,
        "return_code": process.returncode,
        "status_4gb": "PASS_4GB" if process.returncode == 0 and 0 < peak_mib <= 4096 else "NOT_TESTED",
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
