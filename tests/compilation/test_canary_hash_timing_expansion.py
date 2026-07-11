"""Timing-record expansion contracts for the canary hash helpers.

``_calibration_observed_items`` and ``_execution_timing_items`` walk a
timing-sample list (possibly containing nested ``timing_pattern`` records) to
produce the flat sequence that feeds the calibration/execution hashes. These
tests cover:

* tolerance of malformed (non-mapping) entries mixed into a sample list,
* the last-occurrence residual correction that keeps the expanded gap total
  exactly equal to a pattern record's declared ``gap_sum_us``, and
* the optional per-item fields on ``_execution_timing_item`` /
  the ``gap_sum_us`` fallback on ``_execution_record_gap_sum``.
"""

from __future__ import annotations

import unittest

from commcanary.artifacts.canary_hashes import (
    _calibration_observed_items,
    _execution_record_gap_sum,
    _execution_timing_item,
    _execution_timing_items,
)


class CalibrationObservedItemsTests(unittest.TestCase):
    def test_non_mapping_samples_are_skipped_without_raising(self):
        samples = [
            {"observed_exposed_us": 1.0, "weight": 1},
            "garbage",
            42,
            {"observed_exposed_us": 2.0, "weight": 1},
        ]
        self.assertEqual(list(_calibration_observed_items(samples)), [1.0, 2.0])


class ExecutionTimingItemsTests(unittest.TestCase):
    def test_non_mapping_samples_are_skipped_without_raising(self):
        samples = [
            {"gap_us": 1.0, "weight": 1},
            "garbage",
            {"gap_us": 2.0, "weight": 1},
        ]
        items = list(_execution_timing_items(samples))
        self.assertEqual([item["gap_us"] for item in items], [1.0, 2.0])

    def test_pattern_residual_is_absorbed_by_the_final_expanded_occurrence(self):
        # A two-item pattern repeated twice (4 logical items). The declared
        # gap_sum_us is 0.5us higher than the naive sum of the expanded
        # pattern items, which happens whenever the pattern itself was
        # rounded independently of the recorded total. Only the very last
        # expanded item may absorb that residual, so the encoded total must
        # still equal the declared gap_sum_us exactly.
        pattern = [{"gap_us": 1.0, "weight": 1}, {"gap_us": 2.0, "weight": 1}]
        declared_total = (1.0 + 2.0) * 2 + 0.5
        sample = {"timing_pattern": pattern, "pattern_repeats": 2, "gap_sum_us": declared_total}

        items = list(_execution_timing_items([sample]))

        self.assertEqual(len(items), 4)
        self.assertEqual([item["gap_us"] for item in items[:-1]], [1.0, 2.0, 1.0])
        self.assertAlmostEqual(items[-1]["gap_us"], 2.0 + 0.5, places=9)
        self.assertAlmostEqual(sum(item["gap_us"] for item in items), declared_total, places=9)

    def test_pattern_without_residual_leaves_every_item_unadjusted(self):
        # When the declared total already matches the naive sum exactly, the
        # residual is zero and no item is adjusted (the "abs(residual) > 0.0"
        # guard stays false for every occurrence, including the last one).
        pattern = [{"gap_us": 1.0, "weight": 1}, {"gap_us": 2.0, "weight": 1}]
        sample = {"timing_pattern": pattern, "pattern_repeats": 2, "gap_sum_us": (1.0 + 2.0) * 2}

        items = list(_execution_timing_items([sample]))

        self.assertEqual([item["gap_us"] for item in items], [1.0, 2.0, 1.0, 2.0])


class ExecutionTimingItemTests(unittest.TestCase):
    def test_optional_fields_are_omitted_when_absent(self):
        item = _execution_timing_item({}, 5.0)
        self.assertEqual(item, {"gap_us": 5.0})

    def test_optional_fields_are_projected_when_present(self):
        sample = {
            "arrival_offsets_us": [1.0, 2.0],
            "compute_overlap_us": 3.0,
            "compute_pressure": 0.25,
        }
        item = _execution_timing_item(sample, 5.0)
        self.assertEqual(item["arrival_offsets_us"], [1.0, 2.0])
        self.assertEqual(item["compute_overlap_us"], 3.0)
        self.assertEqual(item["compute_pressure"], 0.25)


class ExecutionRecordGapSumTests(unittest.TestCase):
    def test_gap_sum_us_is_used_when_present(self):
        self.assertEqual(_execution_record_gap_sum({"gap_sum_us": 42.0}), 42.0)

    def test_gap_sum_falls_back_to_gap_times_weight_when_absent(self):
        self.assertEqual(_execution_record_gap_sum({"gap_us": 3.0, "weight": 4}), 12.0)


if __name__ == "__main__":
    unittest.main()
