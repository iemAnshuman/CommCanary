"""Tiny standalone producer used by the local experiment harness tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _write_result(mode: str) -> None:
    producer_schema = os.environ["COMMCANARY_PRODUCER_SCHEMA"]
    if ".prepare." in producer_schema:
        value_us = 10.0
    elif ".consume." in producer_schema:
        value_us = 20.0
    else:
        value_us = 30.0
    payload = {
        "schema": os.environ["COMMCANARY_RESULT_SCHEMA"],
        "cell_id": os.environ["COMMCANARY_CELL_ID"],
        "cell_identity_sha256": os.environ["COMMCANARY_CELL_IDENTITY_SHA256"],
        "producer_schema": producer_schema,
        "measurement_schema": os.environ["COMMCANARY_MEASUREMENT_SCHEMA"],
        "measurement": {
            "attempt_id": os.environ["COMMCANARY_ATTEMPT_ID"],
            "config_value": os.environ.get("LOCAL_CONFIG"),
            "mode": mode,
            "samples_us": [value_us - 0.5, value_us, value_us + 0.5],
            "secret_present": "SECRET_TOKEN" in os.environ,
            "value_us": value_us,
        },
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    Path(os.environ["COMMCANARY_RESULT_PATH"]).write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("success", "fail", "fail-once", "parse-fail", "sleep", "output"),
        required=True,
    )
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--output-bytes", type=int, default=0)
    args = parser.parse_args()

    print(f"stdout:{args.mode}", flush=True)
    print(f"stderr:{args.mode}", file=sys.stderr, flush=True)
    if args.mode == "sleep":
        time.sleep(args.sleep_seconds)
        _write_result(args.mode)
        return 0
    if args.mode == "output":
        sys.stdout.write("x" * args.output_bytes)
        sys.stdout.flush()
        _write_result(args.mode)
        return 0
    if args.mode == "fail":
        return 17
    if args.mode == "fail-once":
        state_path = Path(os.environ["LOCAL_CELL_FAIL_ONCE_STATE"])
        if not state_path.exists():
            state_path.write_text("failed-once", encoding="utf-8")
            return 19
        _write_result(args.mode)
        return 0
    if args.mode == "parse-fail":
        Path(os.environ["COMMCANARY_RESULT_PATH"]).write_bytes(b"{not-json")
        return 0
    _write_result(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
