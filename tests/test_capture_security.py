import math
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from commcanary.capture import TraceRecorder
from commcanary.schema import SchemaError


class CaptureSecurityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
