"""CLI override of the bounded-JSON input budget for Kineto imports.

A real multi-rank profiler trace legitimately exceeds the default
``max_input_bytes``; the operator raises the budget explicitly per invocation
instead of the tool silently accepting unbounded input.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from commcanary.cli import main as cli_main

SMALL_KINETO = {
    "baseTimeNanoseconds": 1000000,
    "distributedInfo": {"backend": "nccl", "rank": 0, "world_size": 4, "nccl_version": "2.27.5"},
    "traceEvents": [
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "record_param_comms",
            "pid": 7,
            "tid": 7,
            "ts": 100.0,
            "dur": 12.5,
            "args": {
                "External id": 1000,
                "Collective name": "allreduce",
                "dtype": "Float",
                "In msg nelems": 1024,
                "Out msg nelems": 1024,
                "In split size": "[]",
                "Out split size": "[]",
                "Group size": 4,
                "Process Group Name": "0",
                "Process Group Ranks": "[0, 1, 2, 3]",
            },
        }
    ],
}


class ImportKinetoInputBudgetTests(unittest.TestCase):
    def _write_profile(self, directory: str) -> str:
        path = os.path.join(directory, "profile.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(SMALL_KINETO, handle)
        return path

    def test_default_budget_rejects_oversized_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = self._write_profile(tmp)
            output = os.path.join(tmp, "trace.json")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = cli_main(
                    [
                        "import-kineto",
                        profile,
                        "--max-input-bytes",
                        "8",
                        "--output",
                        output,
                    ]
                )
            self.assertEqual(exit_code, 3)
            self.assertIn("max_input_bytes=8", stderr.getvalue())
            self.assertFalse(os.path.exists(output))

    def test_raised_budget_accepts_the_same_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = self._write_profile(tmp)
            output = os.path.join(tmp, "trace.json")
            exit_code = cli_main(
                [
                    "import-kineto",
                    profile,
                    "--max-input-bytes",
                    str(1024 * 1024),
                    "--output",
                    output,
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(output))

    def test_non_positive_budget_is_an_application_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = self._write_profile(tmp)
            output = os.path.join(tmp, "trace.json")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = cli_main(["import-kineto", profile, "--max-input-bytes", "0", "--output", output])
            self.assertEqual(exit_code, 3)
            self.assertIn("positive", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
