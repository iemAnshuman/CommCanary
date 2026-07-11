"""Fallback provenance hashing for canaries without a matching integrity profile.

``canary_artifact_provenance_sha256`` has two paths: a fast path for canaries
whose ``compiler`` block declares the current integrity profile and
provenance algorithm, and a fallback path (``_strip_canary_hash_fields``) for
everything else, which recursively strips ``created_at`` and the
self-referential hash/size fields (``CANARY_HASH_FIELD_NAMES``) out of the
whole mapping before hashing. These tests exercise the fallback path and its
recursive handling of nested mappings, lists, and scalar leaves.
"""

from __future__ import annotations

import copy
import unittest

from commcanary.artifacts.canary_hashes import canary_artifact_provenance_sha256


def _legacy_canary():
    return {
        "format": "commcanary.canary.v2",
        "created_at": "2020-01-01T00:00:00Z",
        # No "integrity_profile"/"artifact_provenance_algorithm" match, so
        # this canary takes the fallback stripping path.
        "compiler": {"note": "legacy exporter, no integrity profile"},
        "events": [{"phase": "decode", "bytes": 10, "ranks": [0, 1]}],
        "canary_bytes": 999,
    }


class ProvenanceFallbackTests(unittest.TestCase):
    def test_created_at_is_ignored_by_the_fallback_hash(self):
        canary = _legacy_canary()
        baseline = canary_artifact_provenance_sha256(canary)

        touched = copy.deepcopy(canary)
        touched["created_at"] = "2099-12-31T23:59:59Z"

        self.assertEqual(canary_artifact_provenance_sha256(touched), baseline)

    def test_self_referential_hash_fields_are_ignored_by_the_fallback_hash(self):
        canary = _legacy_canary()
        baseline = canary_artifact_provenance_sha256(canary)

        touched = copy.deepcopy(canary)
        touched["canary_bytes"] = 1  # a CANARY_HASH_FIELD_NAMES member

        self.assertEqual(canary_artifact_provenance_sha256(touched), baseline)

    def test_fallback_hash_is_still_sensitive_to_real_event_content(self):
        canary = _legacy_canary()
        baseline = canary_artifact_provenance_sha256(canary)

        touched = copy.deepcopy(canary)
        touched["events"][0]["bytes"] = 20  # a nested scalar leaf, not a hash field

        self.assertNotEqual(canary_artifact_provenance_sha256(touched), baseline)

    def test_fallback_hash_is_deterministic(self):
        canary = _legacy_canary()
        self.assertEqual(
            canary_artifact_provenance_sha256(canary),
            canary_artifact_provenance_sha256(copy.deepcopy(canary)),
        )


if __name__ == "__main__":
    unittest.main()
