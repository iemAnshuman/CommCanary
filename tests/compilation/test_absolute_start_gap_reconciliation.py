"""Absolute-clock gap reconciliation in trace event ordering.

When every trace event carries ``start_us``, ``_ordered_trace_events`` sorts
by that absolute clock and, for each event past the first, cross-checks any
explicit ``gap_us`` against the gap derived from consecutive ``start_us``
values. This test exercises the case where the explicit value agrees with the
derived one (within tolerance), which must be accepted silently rather than
raising a conflict error.
"""

from __future__ import annotations

import unittest

from commcanary.compilation.normalization import _ordered_trace_events


class OrderedTraceEventsAbsoluteClockTests(unittest.TestCase):
    def test_explicit_gap_matching_the_derived_start_us_gap_is_accepted(self):
        events = [
            {"start_us": 0.0},
            {"start_us": 10.0, "gap_us": 10.0},
            {"start_us": 25.0, "gap_us": 15.0},
        ]

        result, gaps, mode = _ordered_trace_events(events)

        self.assertEqual(mode, "absolute_start_us")
        self.assertEqual(result, events)
        self.assertEqual(gaps, [0.0, 10.0, 15.0])

    def test_explicit_gap_within_tolerance_of_the_derived_gap_is_accepted(self):
        # 0.0005us is inside the 0.001us reconciliation tolerance, so this
        # must take the same silent-acceptance path as an exact match.
        events = [
            {"start_us": 0.0},
            {"start_us": 10.0, "gap_us": 10.0005},
        ]

        _result, gaps, mode = _ordered_trace_events(events)

        self.assertEqual(mode, "absolute_start_us")
        self.assertEqual(gaps, [0.0, 10.0])


if __name__ == "__main__":
    unittest.main()
