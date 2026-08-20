"""Bounded cycle telemetry shared by application and reduced runners."""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

from ..artifacts.json_codec import canonical_json_bytes


def capture_cycle_telemetry(phase: str) -> Dict[str, Any]:
    """Capture one bounded GPU, kernel-event, and scheduler observation."""

    query = (
        "index,uuid,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,"
        "clocks_event_reasons.active,ecc.errors.corrected.volatile.total,"
        "ecc.errors.uncorrected.volatile.total"
    )
    command = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    gpu_rows: List[Dict[str, Any]] = []
    query_status = "unavailable"
    query_returncode: Optional[int] = None
    query_stderr = ""
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
        query_returncode = completed.returncode
        query_stderr = completed.stderr[:4096]
        if completed.returncode == 0:
            for raw_row in csv.reader(completed.stdout[:65536].splitlines(), skipinitialspace=True):
                if len(raw_row) != 10:
                    raise ValueError(f"nvidia-smi returned {len(raw_row)} fields instead of 10")
                gpu_rows.append(
                    {
                        "index": int(raw_row[0]),
                        "uuid": raw_row[1].strip(),
                        "pstate": raw_row[2].strip(),
                        "sm_clock_mhz": float(raw_row[3]),
                        "memory_clock_mhz": float(raw_row[4]),
                        "temperature_c": float(raw_row[5]),
                        "power_w": float(raw_row[6]),
                        "clock_event_reasons_active": raw_row[7].strip(),
                        "corrected_volatile_ecc": int(raw_row[8]),
                        "uncorrected_volatile_ecc": int(raw_row[9]),
                    }
                )
            query_status = "complete" if gpu_rows else "unavailable"
    except (OSError, subprocess.TimeoutExpired) as exc:
        query_stderr = f"{type(exc).__name__}: {exc}"
    except (TypeError, ValueError) as exc:
        query_stderr = f"telemetry parse error: {exc}"
        gpu_rows = []

    xid_command = ["journalctl", "-k", "-n", "1000", "--no-pager"]
    xid_status = "unavailable"
    xid_lines: List[str] = []
    xid_stderr = ""
    try:
        xid_completed = subprocess.run(xid_command, check=False, capture_output=True, text=True, timeout=10)
        xid_stderr = xid_completed.stderr[:4096]
        if xid_completed.returncode == 0:
            xid_lines = [
                line[:1024]
                for line in xid_completed.stdout[:1048576].splitlines()
                if "NVRM: Xid" in line or "Xid (PCI:" in line
            ]
            xid_status = "complete"
    except (OSError, subprocess.TimeoutExpired) as exc:
        xid_stderr = f"{type(exc).__name__}: {exc}"

    slurm_node = os.environ.get("SLURMD_NODENAME")
    slurm_command = ["scontrol", "show", "node", str(slurm_node), "-o"]
    slurm_status = "unavailable"
    slurm_state: Optional[str] = None
    slurm_stderr = ""
    if slurm_node:
        try:
            slurm_completed = subprocess.run(slurm_command, check=False, capture_output=True, text=True, timeout=10)
            slurm_stderr = slurm_completed.stderr[:4096]
            if slurm_completed.returncode == 0:
                state_fields = [
                    field.partition("=")[2] for field in slurm_completed.stdout.split() if field.startswith("State=")
                ]
                if len(state_fields) == 1 and state_fields[0]:
                    slurm_state = state_fields[0]
                    slurm_status = "complete"
        except (OSError, subprocess.TimeoutExpired) as exc:
            slurm_stderr = f"{type(exc).__name__}: {exc}"

    return {
        "phase": phase,
        "monotonic_ns": time.monotonic_ns(),
        "gpu_observation": {
            "status": query_status,
            "command": command,
            "returncode": query_returncode,
            "gpus": gpu_rows,
            "stderr": query_stderr,
        },
        "xid_observation": {
            "status": xid_status,
            "command": xid_command,
            "lines": xid_lines,
            "sha256": hashlib.sha256(canonical_json_bytes(xid_lines)).hexdigest(),
            "stderr": xid_stderr,
        },
        "slurm_node_observation": {
            "status": slurm_status,
            "command": slurm_command,
            "node": slurm_node,
            "state": slurm_state,
            "stderr": slurm_stderr,
        },
    }


__all__ = ["capture_cycle_telemetry"]
