import hashlib
import inspect
import math
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import commcanary.capture as capture_module
from commcanary.capture import TraceRecorder
from commcanary.capture import record_collective as module_record_collective
from commcanary.cli import main
from commcanary.schema import SchemaError, load_json


class CaptureSecurityTests(unittest.TestCase):
    def test_module_record_helper_has_typed_size_alias_contract(self):
        signature = inspect.signature(module_record_collective)
        self.assertIn("byte_count", signature.parameters)
        self.assertIn("bytes", signature.parameters)
        self.assertNotIn("kwargs", signature.parameters)

        with self.assertRaisesRegex(SchemaError, "requires byte_count"):
            module_record_collective(op="all_reduce", ranks=[0])
        with self.assertRaisesRegex(SchemaError, "only one"):
            module_record_collective(op="all_reduce", ranks=[0], byte_count=16, bytes=16)

    def _recorder_for_rank(self, trace_root: Path, rank_label: str) -> TraceRecorder:
        with mock.patch.dict(
            os.environ,
            {
                "COMMCANARY_TRACE_DIR": str(trace_root),
                "COMMCANARY_RANK": rank_label,
            },
        ):
            return TraceRecorder("ignored-by-trace-dir.json")

    def _record_one(self, recorder: TraceRecorder, *, metadata=None) -> None:
        recorder.record_collective(
            op="all_reduce",
            bytes=16,
            ranks=[0],
            rank_arrival_us={"0": 0.0},
            metadata=metadata,
        )

    def test_numeric_rank_filename_compatibility_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "shards"
            recorder = self._recorder_for_rank(root, "42")

            output = Path(recorder.output_path)
            self.assertEqual(output.parent, root.resolve())
            self.assertTrue(output.name.startswith("rank-42-pid-"), output.name)

    def test_unsafe_unicode_and_oversized_rank_labels_are_bounded_slugs(self):
        labels = (
            "../escape",
            "nested/escape",
            r"nested\escape",
            "/absolute/escape",
            "..",
            "１２",
            "．．∕escape",
            "rank-💥",
            "9" * 10_000,
        )
        for label in labels:
            with self.subTest(label=label[:40]):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "shards"
                    recorder = self._recorder_for_rank(root, label)
                    output = Path(recorder.output_path)

                    self.assertEqual(output.parent.resolve(), root.resolve())
                    self.assertTrue(output.name.startswith("rank-label-"), output.name)
                    self.assertLess(len(output.name.encode("utf-8")), 128)
                    self._record_one(recorder)
                    recorder.save()
                    self.assertTrue(output.is_file())

    def test_save_rechecks_containment_before_creating_or_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "shards"
            outside = base / "outside"
            recorder = self._recorder_for_rank(root, "0")
            self._record_one(recorder)

            recorder.output_path = str(root / ".." / "outside" / "escaped.trace.json")
            with self.assertRaisesRegex(SchemaError, "escapes configured trace directory"):
                recorder.save()
            self.assertFalse(outside.exists())

            recorder.output_path = str(base / "absolute-escape.trace.json")
            with self.assertRaisesRegex(SchemaError, "escapes configured trace directory"):
                recorder.save()
            self.assertFalse((base / "absolute-escape.trace.json").exists())

    def test_context_exit_preserves_workload_error_when_save_also_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(str(Path(tmp) / "trace.json"))
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with mock.patch.object(
                    recorder,
                    "save",
                    side_effect=SchemaError("simulated save failure"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "workload failed") as caught:
                        with recorder:
                            raise RuntimeError("workload failed")

            save_error = getattr(caught.exception, "commcanary_save_error", None)
            self.assertIsInstance(save_error, SchemaError)
            self.assertIn("simulated save failure", str(save_error))

    def test_context_exit_still_raises_save_error_after_successful_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(str(Path(tmp) / "trace.json"))
            with mock.patch.object(
                recorder,
                "save",
                side_effect=SchemaError("simulated save failure"),
            ):
                with self.assertRaisesRegex(SchemaError, "simulated save failure"):
                    with recorder:
                        pass

    def test_metadata_must_be_json_serializable_before_event_is_recorded(self):
        recursive = {}
        recursive["self"] = recursive
        invalid_metadata = (
            {"object": object()},
            {"set": {1, 2}},
            {"nan": math.nan},
            {"infinity": math.inf},
            recursive,
        )

        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(str(Path(tmp) / "trace.json"))
            for metadata in invalid_metadata:
                with self.subTest(metadata_type=next(iter(metadata))):
                    with self.assertRaisesRegex(SchemaError, "metadata must be JSON serializable"):
                        self._record_one(recorder, metadata=metadata)
                    self.assertEqual(recorder.events, [])

            valid = {"unicode": "安全", "values": [1, True, None]}
            self._record_one(recorder, metadata=valid)
            valid["values"].append("mutated")
            self.assertEqual(
                recorder.events[0]["metadata"],
                {"unicode": "安全", "values": [1, True, None]},
            )

    def test_direct_output_has_one_owner_until_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "trace.json"
            first = TraceRecorder(str(output))
            with self.assertRaisesRegex(SchemaError, "already has an active recorder"):
                TraceRecorder(str(output))
            first.close()
            with self.assertRaisesRegex(SchemaError, "closed"):
                self._record_one(first)

            second = TraceRecorder(str(output))
            second.close()
            self.assertFalse((Path(tmp) / ".trace.json.commcanary.lock").exists())

    def test_direct_output_claim_is_cross_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "trace.json"
            project_root = Path(__file__).resolve().parents[1]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_root / "src")
            code = (
                "from commcanary.capture import TraceRecorder\n"
                "import sys\n"
                "recorder = TraceRecorder(sys.argv[1])\n"
                "print('ready', flush=True)\n"
                "sys.stdin.readline()\n"
                "recorder.close()\n"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", code, str(output)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            try:
                self.assertIsNotNone(process.stdout)
                self.assertEqual(process.stdout.readline().strip(), "ready")
                with self.assertRaisesRegex(SchemaError, "claimed by another process"):
                    TraceRecorder(str(output))
                self.assertIsNotNone(process.stdin)
                process.stdin.write("\n")
                process.stdin.flush()
                self.assertEqual(process.wait(timeout=10), 0)
                replacement = TraceRecorder(str(output))
                replacement.close()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)

    def test_auto_recorder_tracks_environment_enable_change_and_disable(self):
        environment_names = (
            "COMMCANARY_TRACE_DIR",
            "COMMCANARY_TRACE_OUT",
            "COMMCANARY_WORKLOAD_NAME",
            "COMMCANARY_CAPTURE_SESSION_ID",
            "COMMCANARY_TRACE_SHARDED",
            "COMMCANARY_RANK",
            "RANK",
            "OMPI_COMM_WORLD_RANK",
            "SLURM_PROCID",
            "LOCAL_RANK",
        )
        original = {name: os.environ.get(name) for name in environment_names}
        try:
            for name in environment_names:
                os.environ.pop(name, None)
            with (
                mock.patch.object(capture_module, "_AUTO_RECORDER", None),
                mock.patch.object(capture_module, "_AUTO_RECORDER_SIGNATURE", None),
                mock.patch.object(capture_module, "_AUTO_RECORDER_ATEXIT_REGISTERED", True),
            ):
                disabled = capture_module.get_recorder()
                self.assertIsInstance(disabled, capture_module.NullRecorder)

                with tempfile.TemporaryDirectory() as tmp:
                    first_root = Path(tmp) / "first"
                    second_root = Path(tmp) / "second"
                    os.environ.update(
                        {
                            "COMMCANARY_TRACE_DIR": str(first_root),
                            "COMMCANARY_WORKLOAD_NAME": "first-workload",
                            "COMMCANARY_CAPTURE_SESSION_ID": "session-1",
                            "COMMCANARY_RANK": "0",
                        }
                    )
                    first = capture_module.get_recorder()
                    self._record_one(first)

                    os.environ["COMMCANARY_TRACE_DIR"] = str(second_root)
                    os.environ["COMMCANARY_WORKLOAD_NAME"] = "second-workload"
                    second = capture_module.get_recorder()
                    self.assertIsNot(first, second)
                    self.assertTrue(first._closed)
                    self.assertTrue(Path(first.output_path).is_file())
                    self.assertEqual(second.workload["name"], "second-workload")

                    os.environ.pop("COMMCANARY_TRACE_DIR")
                    disabled_again = capture_module.get_recorder()
                    self.assertIsInstance(disabled_again, capture_module.NullRecorder)
                    self.assertTrue(second._closed)

                    os.environ["COMMCANARY_TRACE_DIR"] = str(first_root)
                    third = capture_module.get_recorder()
                    self.assertIsInstance(third, TraceRecorder)
                    self.assertIsNot(third, first)
                    third.close()
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_failed_capture_preserves_partial_shards_and_child_code(self):
        child = (
            "from commcanary.capture import TraceRecorder\n"
            "recorder = TraceRecorder.from_env()\n"
            "recorder.record_collective(op='all_reduce', bytes=16, ranks=[0], "
            "rank_arrival_us={'0': 0.0})\n"
            "recorder.save()\n"
            "raise SystemExit(7)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "successful-output.json"
            preserved = Path(tmp) / "failed-capture"
            result = main(
                [
                    "capture",
                    "--output",
                    str(output),
                    "--workload-name",
                    "failure-test",
                    "--preserve-on-failure",
                    str(preserved),
                    "--",
                    sys.executable,
                    "-c",
                    child,
                ]
            )
            self.assertEqual(result, 4)
            self.assertFalse(output.exists())
            manifest = load_json(str(preserved / "capture_failure.json"))
            self.assertEqual(manifest["format"], "commcanary.capture_failure.v1")
            self.assertEqual(manifest["child_returncode"], 7)
            self.assertEqual(manifest["workload"], {"name": "failure-test"})
            self.assertNotIn("command", manifest)
            self.assertNotIn("environment", manifest)
            self.assertEqual(len(manifest["partial_shards"]), 1)
            shard = manifest["partial_shards"][0]
            copied = preserved / "shards" / shard["name"]
            self.assertEqual(copied.stat().st_size, shard["size_bytes"])
            self.assertEqual(
                hashlib.sha256(copied.read_bytes()).hexdigest(),
                shard["sha256"],
            )

    def test_failed_capture_bundle_handles_no_shards_and_refuses_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_bundle = Path(tmp) / "empty-failure"
            result = main(
                [
                    "capture",
                    "--output",
                    str(Path(tmp) / "unused.json"),
                    "--preserve-on-failure",
                    str(empty_bundle),
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(9)",
                ]
            )
            self.assertEqual(result, 4)
            manifest = load_json(str(empty_bundle / "capture_failure.json"))
            self.assertEqual(manifest["child_returncode"], 9)
            self.assertEqual(manifest["partial_shards"], [])

            collision = Path(tmp) / "collision"
            collision.mkdir()
            result = main(
                [
                    "capture",
                    "--output",
                    str(Path(tmp) / "still-unused.json"),
                    "--preserve-on-failure",
                    str(collision),
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(5)",
                ]
            )
            self.assertEqual(result, 3)


if __name__ == "__main__":
    unittest.main()
