from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from commcanary.adapters.param import audit_param_program_compute_operations
from commcanary.cli import main as cli_main
from commcanary.compiler import compile_trace
from commcanary.interop import (
    canary_to_param_comms_trace,
    kineto_trace_to_commcanary_trace,
    kineto_traces_to_commcanary_trace,
)
from commcanary.schema import SchemaError, load_json, validate_trace, write_json
from tests.builders import small_trace


class InteropTests(unittest.TestCase):
    def _synthetic_kineto_trace(self):
        def comms_event(index, ts, name, dtype, nelems, group="0", extra=None):
            args = {
                "External id": 1000 + index,
                "Collective name": name,
                "dtype": dtype,
                "In msg nelems": nelems,
                "Out msg nelems": nelems,
                "In split size": "[]",
                "Out split size": "[]",
                "Group size": 4,
                "Process Group Name": group,
                "Process Group Ranks": "[0, 1, 2, 3]",
            }
            if extra:
                args.update(extra)
            return {
                "ph": "X",
                "cat": "cpu_op",
                "name": "record_param_comms",
                "pid": 7,
                "tid": 7,
                "ts": ts,
                "dur": 12.5,
                "args": args,
            }

        return {
            "baseTimeNanoseconds": 1000000,
            "distributedInfo": {"backend": "nccl", "rank": 0, "world_size": 4, "nccl_version": "2.27.5"},
            "traceEvents": [
                comms_event(0, 100.0, "allreduce", "Float", 1024),
                comms_event(1, 220.5, "_allgather_base", "BFloat16", 2048),
                comms_event(2, 300.0, "wait", "Float", 0),
                comms_event(3, 410.0, "reduce", "Float", 512),
                comms_event(4, 500.25, "send", "Half", 256, extra={"Src Rank": 0, "Dst Rank": 2, "Seq": 9}),
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "ncclDevKernel_AllReduce_Sum_f32(ncclDevKernelArgsStorage<4096ul>)",
                    "ts": 105.0,
                    "dur": 40.0,
                    "args": {"External id": 1000},
                },
            ],
        }

    def _multi_rank_kineto_trace(self, rank, starts, overlaps):
        events = []
        for index, (start_us, overlap_us) in enumerate(zip(starts, overlaps)):
            external_id = 1000 + index
            events.append(
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "record_param_comms",
                    "pid": 7,
                    "tid": 7,
                    "ts": start_us,
                    "dur": 10.0,
                    "args": {
                        "External id": external_id,
                        "Collective name": "allreduce",
                        "dtype": "Float",
                        "In msg nelems": 1024 * (index + 1),
                        "Out msg nelems": 1024 * (index + 1),
                        "In split size": "[]",
                        "Out split size": "[]",
                        "Group size": 2,
                        "Process Group Name": "0",
                        "Process Group Ranks": "[0, 1]",
                    },
                }
            )
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "ncclDevKernel_AllReduce_Sum_f32",
                    "pid": 0,
                    "tid": 7,
                    "ts": start_us + 20.0,
                    "dur": 20.0,
                    "args": {
                        "External id": external_id,
                        "device": 0,
                        "stream": 7,
                    },
                }
            )
            if overlap_us:
                events.append(
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "gemm_compute",
                        "pid": 0,
                        "tid": 1,
                        "ts": start_us + 20.0,
                        "dur": overlap_us,
                        "args": {
                            "External id": 2000 + index,
                            "device": 0,
                            "stream": 1,
                        },
                    }
                )
        return {
            "baseTimeNanoseconds": 1000000,
            "distributedInfo": {
                "backend": "nccl",
                "rank": rank,
                "world_size": 2,
                "nccl_version": "2.27.5",
            },
            "traceEvents": events,
        }

    def test_kineto_import_maps_collectives_to_trace_events(self):
        trace = kineto_trace_to_commcanary_trace(self._synthetic_kineto_trace(), phase="decode")
        validate_trace(trace)
        events = trace["events"]
        self.assertEqual(len(events), 4)
        self.assertEqual([event["op"] for event in events], ["all_reduce", "all_gather", "reduce", "send"])
        self.assertEqual(
            [event["dtype"] for event in events],
            ["float32", "bfloat16", "float32", "float16"],
        )
        self.assertEqual(events[0]["bytes"], 1024 * 4)
        self.assertEqual(events[1]["bytes"], 2048 * 2)
        self.assertEqual(events[3]["bytes"], 256 * 2)
        self.assertTrue(events[2].get("custom_op"))
        self.assertEqual(events[0]["ranks"], [0, 1, 2, 3])
        self.assertEqual(events[0]["group"], "0")
        self.assertEqual(events[0]["start_us"], 0.0)
        self.assertEqual(events[1]["start_us"], 120.5)
        self.assertEqual(events[3]["start_us"], 400.25)
        self.assertNotIn("kineto_trace_start_us", trace["workload"])
        self.assertNotIn("kineto_base_time_ns", trace["system"])
        self.assertEqual(events[0]["phase"], "decode")
        self.assertEqual(events[3]["metadata"]["kineto_dst_rank"], 2)
        self.assertEqual(events[3]["sender_rank"], 0)
        self.assertEqual(events[3]["receiver_rank"], 2)
        self.assertEqual(trace["workload"]["skipped_control_events"], 1)
        self.assertEqual(trace["system"]["kineto_nccl_version"], "2.27.5")
        self.assertTrue(all(event["compute_overlap_unknown"] for event in events))
        self.assertEqual(trace["workload"]["overlap_derived_events"], 0)
        self.assertEqual(trace["workload"]["overlap_unknown_events"], 4)
        self.assertEqual(trace["workload"]["message_shapes_derived_events"], 2)
        self.assertEqual(trace["workload"]["message_shapes_unknown_events"], 2)
        self.assertEqual(events[0]["metadata"]["kineto_in_split_sizes"], [])
        self.assertEqual(events[0]["metadata"]["kineto_out_split_sizes"], [])
        self.assertEqual(events[0]["metadata"]["kineto_message_shape_status"], "derived")
        self.assertEqual(
            events[1]["metadata"]["kineto_message_shape_status"],
            "unavailable_in_out_mismatch",
        )
        self.assertEqual(
            [event["metadata"]["kineto_overlap_status"] for event in events],
            [
                "unavailable_incomplete_linked_communication_kernel",
                "unavailable_no_linked_communication_kernel",
                "unavailable_no_linked_communication_kernel",
                "unavailable_no_linked_communication_kernel",
            ],
        )
        with self.assertRaisesRegex(SchemaError, "unknown compute overlap"):
            compile_trace(trace)

    def test_kineto_import_derives_broadcast_root_from_containing_c10d_call(self):
        kineto = {
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "c10d::broadcast_",
                    "pid": 7,
                    "tid": 8,
                    "ts": 90.0,
                    "dur": 40.0,
                    "args": {
                        "Concrete Inputs": ["", "", "1", "0", "True", "-1"],
                        "Input type": ["TensorList", "", "Scalar", "Scalar", "Scalar", "Scalar"],
                    },
                },
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "record_param_comms",
                    "pid": 7,
                    "tid": 8,
                    "ts": 100.0,
                    "dur": 10.0,
                    "args": {
                        "External id": 42,
                        "Collective name": "broadcast",
                        "dtype": "Float",
                        "In msg nelems": 1024,
                        "Out msg nelems": 1024,
                        "Group size": 2,
                        "Process Group Name": "0",
                        "Process Group Ranks": "[0, 1]",
                    },
                },
            ]
        }

        trace = kineto_trace_to_commcanary_trace(kineto)
        event = trace["events"][0]
        self.assertEqual(event["root_rank"], 1)
        self.assertEqual(event["metadata"]["kineto_broadcast_root_rank"], 1)
        self.assertEqual(event["metadata"]["kineto_broadcast_root_status"], "derived")
        self.assertEqual(
            event["metadata"]["kineto_broadcast_root_method"],
            "c10d-concrete-input-root-rank.v1",
        )
        self.assertEqual(trace["workload"]["broadcast_roots_derived_events"], 1)
        self.assertEqual(trace["workload"]["broadcast_roots_unknown_events"], 0)

        missing_parent = copy.deepcopy(kineto)
        missing_parent["traceEvents"] = missing_parent["traceEvents"][1:]
        unknown = kineto_trace_to_commcanary_trace(missing_parent)
        self.assertNotIn("root_rank", unknown["events"][0])
        self.assertEqual(
            unknown["events"][0]["metadata"]["kineto_broadcast_root_status"],
            "unknown",
        )
        self.assertEqual(unknown["workload"]["broadcast_roots_derived_events"], 0)
        self.assertEqual(unknown["workload"]["broadcast_roots_unknown_events"], 1)

    def test_kineto_import_derives_union_of_cross_stream_kernel_overlap(self):
        kineto = self._synthetic_kineto_trace()
        # Replace the deliberately incomplete kernel fixture with complete
        # Kineto-style CUDA activities for every imported collective.
        kineto["traceEvents"] = [raw for raw in kineto["traceEvents"] if raw.get("cat") != "kernel"]

        def kernel(index, ts, dur, name, stream, external_id, *, device=0, extra=None):
            args = {
                "External id": external_id,
                "device": device,
                "stream": stream,
            }
            if extra:
                args.update(extra)
            return {
                "ph": "X",
                "cat": "kernel",
                "name": name,
                "pid": device,
                "tid": stream,
                "ts": ts,
                "dur": dur,
                "args": args,
                "_test_index": index,
            }

        kineto["traceEvents"].extend(
            [
                kernel(0, 150.0, 50.0, "ncclDevKernel_AllReduce", 7, 1000),
                # [150,160] and [155,180] overlap the communication kernel.
                # Their union is [150,180], not the double-counted sum 35us.
                kernel(1, 140.0, 20.0, "vectorized_elementwise_kernel", 1, 2001),
                kernel(2, 155.0, 25.0, "ampere_sgemm_128x128", 2, 2002),
                # Same-stream timestamps do not demonstrate concurrency.
                kernel(3, 160.0, 30.0, "same_stream_helper", 7, 2003),
                # Another device cannot overlap this rank's communication.
                kernel(4, 150.0, 40.0, "other_device_gemm", 1, 2004, device=1),
                # An unrelated communication kernel is not compute.
                kernel(5, 160.0, 20.0, "ncclDevKernel_Broadcast", 3, 9999),
                # Kineto can round very short kernels to a measured empty
                # interval. It cannot contribute overlap and is safely ignored.
                kernel(9, 170.0, 0.0, "zero_duration_helper", 4, 2005),
                kernel(6, 230.0, 20.0, "ncclDevKernel_AllGather", 7, 1001),
                kernel(7, 420.0, 20.0, "ncclDevKernel_Reduce", 7, 1003),
                # Copied collective metadata is also sufficient to classify a
                # linked communication kernel when its name drifts.
                kernel(
                    8,
                    510.0,
                    20.0,
                    "vendor_generated_kernel_42",
                    7,
                    1004,
                    extra={"Collective name": "send"},
                ),
            ]
        )

        trace = kineto_trace_to_commcanary_trace(kineto)
        validate_trace(trace, require_known_overlap=True)
        events = trace["events"]
        self.assertEqual([event["compute_overlap_us"] for event in events], [30.0, 0.0, 0.0, 0.0])
        self.assertTrue(all("compute_overlap_unknown" not in event for event in events))
        self.assertEqual(trace["workload"]["overlap_derived_events"], 4)
        self.assertEqual(trace["workload"]["overlap_unknown_events"], 0)
        first_metadata = events[0]["metadata"]
        self.assertEqual(first_metadata["kineto_overlap_status"], "derived")
        self.assertEqual(
            first_metadata["kineto_overlap_method"],
            "linked-kernel-interval-union.v1",
        )
        self.assertEqual(first_metadata["kineto_communication_kernel_count"], 1)
        self.assertEqual(first_metadata["kineto_compute_kernel_count"], 2)
        self.assertEqual(first_metadata["kineto_communication_duration_us"], 50.0)
        compile_trace(trace)

    def test_kineto_import_derives_rank_local_exact_gemm_recipe_between_issue_and_wait(self):
        def profile(rank, *, gemm_m, kernel_duration):
            collective_args = {
                "External id": 1000,
                "Collective name": "allreduce",
                "dtype": "BFloat16",
                "In msg nelems": 32768,
                "Out msg nelems": 32768,
                "In split size": "[]",
                "Out split size": "[]",
                "Group size": 2,
                "Process Group Name": "0",
                "Process Group Ranks": "[0, 1]",
            }
            return {
                "distributedInfo": {
                    "backend": "nccl",
                    "rank": rank,
                    "world_size": 2,
                    "nccl_version": "2.20.5",
                },
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "record_param_comms",
                        "pid": 7,
                        "tid": 7,
                        "ts": 100.0,
                        "dur": 10.0,
                        "args": collective_args,
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "ncclDevKernel_AllReduce_Sum_bf16",
                        "pid": 0,
                        "tid": 18,
                        "ts": 115.0,
                        "dur": 160.0,
                        "args": {"External id": 1000, "device": 0, "stream": 18},
                    },
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::matmul",
                        "pid": 7,
                        "tid": 7,
                        "ts": 120.0,
                        "dur": 9.0,
                        "args": {"External id": 2000},
                    },
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::mm",
                        "pid": 7,
                        "tid": 7,
                        "ts": 121.0,
                        "dur": 7.0,
                        "args": {
                            "External id": 2001,
                            "Input Dims": [[gemm_m, 8192], [8192, 8192]],
                            "Input Strides": [[8192, 1], [8192, 1]],
                            "Input type": ["c10::BFloat16", "c10::BFloat16"],
                        },
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "ampere_bf16_gemm",
                        "pid": 0,
                        "tid": 7,
                        "ts": 130.0,
                        "dur": kernel_duration,
                        "args": {"External id": 2001, "device": 0, "stream": 7},
                    },
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "record_param_comms",
                        "pid": 7,
                        "tid": 7,
                        "ts": 140.0,
                        "dur": 4.0,
                        "args": {
                            "External id": 3000,
                            "Collective name": "wait",
                            "dtype": "Byte",
                            "In msg nelems": 0,
                            "Out msg nelems": 0,
                            "Group size": 1,
                            "Process Group Ranks": "[]",
                        },
                    },
                ],
            }

        rank0 = profile(0, gemm_m=12, kernel_duration=80.0)
        rank1 = profile(1, gemm_m=20, kernel_duration=120.0)
        single = kineto_trace_to_commcanary_trace(rank0)
        self.assertEqual(
            single["events"][0]["compute_recipe"],
            [
                {
                    "op": "gemm",
                    "dtype": "bfloat16",
                    "m": 12,
                    "n": 8192,
                    "k": 8192,
                    "source_kernel_count": 1,
                    "source_kernel_duration_us": 80.0,
                }
            ],
        )
        self.assertEqual(
            single["events"][0]["metadata"]["kineto_compute_recipe_method"],
            "explicit-wait-linked-contiguous-gemm.v1",
        )
        self.assertEqual(single["workload"]["compute_recipes_derived_events"], 1)
        validate_trace(single)

        merged = kineto_traces_to_commcanary_trace(
            [rank1, rank0],
            assume_shared_clock=True,
        )
        event = merged["events"][0]
        self.assertNotIn("compute_recipe", event)
        self.assertEqual(event["compute_recipe_by_rank"]["0"][0]["m"], 12)
        self.assertEqual(event["compute_recipe_by_rank"]["1"][0]["m"], 20)
        self.assertEqual(
            event["compute_recipe_by_rank"]["1"][0]["source_kernel_duration_us"],
            120.0,
        )
        validate_trace(merged, require_known_overlap=True)
        noncanonical_dtype = copy.deepcopy(merged)
        noncanonical_dtype["events"][0]["compute_recipe_by_rank"]["0"][0]["dtype"] = "Half"
        with self.assertRaisesRegex(SchemaError, "canonical spelling"):
            validate_trace(noncanonical_dtype)
        incomplete_rank_recipe = copy.deepcopy(merged)
        incomplete_rank_recipe["events"][0]["compute_recipe_by_rank"].pop("1")
        with self.assertRaisesRegex(SchemaError, "keys must match ranks"):
            validate_trace(incomplete_rank_recipe)

        missing_wait = copy.deepcopy(rank0)
        missing_wait["traceEvents"] = missing_wait["traceEvents"][:-1]
        unavailable = kineto_trace_to_commcanary_trace(missing_wait)
        self.assertNotIn("compute_recipe", unavailable["events"][0])
        self.assertEqual(
            unavailable["events"][0]["metadata"]["kineto_compute_recipe_status"],
            "unavailable_missing_explicit_wait",
        )

        unsupported = copy.deepcopy(rank0)
        unsupported["traceEvents"][3]["name"] = "aten::add"
        refused_recipe = kineto_trace_to_commcanary_trace(unsupported)
        self.assertNotIn("compute_recipe", refused_recipe["events"][0])
        self.assertEqual(
            refused_recipe["events"][0]["metadata"]["kineto_compute_recipe_status"],
            "unavailable_unsupported_compute_operator",
        )

    def test_multi_rank_kineto_import_preserves_arrivals_and_rank_local_overlap(self):
        rank0 = self._multi_rank_kineto_trace(0, [100.0, 200.0], [5.0, 0.0])
        rank1 = self._multi_rank_kineto_trace(1, [104.0, 203.0], [2.0, 0.0])

        trace = kineto_traces_to_commcanary_trace(
            [rank1, rank0],
            clock_offsets_us={0: 0.0, 1: 0.0},
        )
        validate_trace(trace, require_known_overlap=True)
        self.assertEqual(trace["workload"]["import_mode"], "multi-rank")
        self.assertEqual(trace["workload"]["imported_ranks"], [0, 1])
        self.assertEqual(trace["workload"]["source_rank_events"], 4)
        self.assertEqual(trace["system"]["clock_alignment"], "explicit_offset_us")
        self.assertNotIn("created_at", trace)

        first, second = trace["events"]
        self.assertEqual(first["reduction_op"], "sum")
        self.assertEqual(second["reduction_op"], "sum")
        self.assertEqual(trace["workload"]["reduction_ops_derived_events"], 2)
        self.assertEqual(trace["workload"]["reduction_ops_unknown_events"], 0)
        self.assertEqual(first["rank_arrival_us"], {"0": 0.0, "1": 4.0})
        self.assertEqual(second["rank_arrival_us"], {"0": 0.0, "1": 3.0})
        self.assertEqual(first["compute_overlap_us"], 2.0)
        self.assertEqual(first["compute_by_rank"]["0"]["compute_overlap_us"], 5.0)
        self.assertEqual(first["compute_by_rank"]["1"]["compute_overlap_us"], 2.0)
        self.assertTrue(first["compute_fields_uncertain"])
        compile_trace(trace)

        # Source argument order must not change the resulting artifact.
        reordered = kineto_traces_to_commcanary_trace(
            [rank0, rank1],
            clock_offsets_us={1: 0.0, 0: 0.0},
        )
        self.assertEqual(trace, reordered)

        # Rank 1 can use a different raw clock when the supplied additive
        # offset maps it back to the same reference timeline.
        shifted_rank1 = self._multi_rank_kineto_trace(1, [1104.0, 1203.0], [2.0, 0.0])
        offset_trace = kineto_traces_to_commcanary_trace(
            [rank0, shifted_rank1],
            clock_offsets_us={0: 0.0, 1: -1000.0},
        )
        self.assertEqual(
            [event["rank_arrival_us"] for event in offset_trace["events"]],
            [event["rank_arrival_us"] for event in trace["events"]],
        )

    def test_multi_rank_kineto_import_refuses_unproven_alignment_and_incomplete_evidence(self):
        rank0 = self._multi_rank_kineto_trace(0, [100.0, 200.0], [5.0, 0.0])
        rank1 = self._multi_rank_kineto_trace(1, [104.0, 203.0], [2.0, 0.0])

        uncalibrated = kineto_traces_to_commcanary_trace([rank0, rank1])
        self.assertTrue(all(event["arrival_skew_unknown"] for event in uncalibrated["events"]))
        with self.assertRaisesRegex(SchemaError, "uncalibrated cross-rank arrival skew"):
            compile_trace(uncalibrated)

        with self.assertRaisesRegex(SchemaError, "exactly match imported ranks"):
            kineto_traces_to_commcanary_trace(
                [rank0, rank1],
                clock_offsets_us={0: 0.0},
            )

        missing = copy.deepcopy(rank1)
        missing["traceEvents"] = [
            raw for raw in missing["traceEvents"] if not (raw.get("cat") == "cpu_op" and raw.get("ts") == 203.0)
        ]
        with self.assertRaisesRegex(SchemaError, "missing rank contributions"):
            kineto_traces_to_commcanary_trace(
                [rank0, missing],
                clock_offsets_us={0: 0.0, 1: 0.0},
            )

        conflicting = copy.deepcopy(rank1)
        conflicting["traceEvents"][0]["args"]["In msg nelems"] = 2048
        conflicting["traceEvents"][0]["args"]["Out msg nelems"] = 2048
        with self.assertRaisesRegex(SchemaError, "conflicting records"):
            kineto_traces_to_commcanary_trace(
                [rank0, conflicting],
                clock_offsets_us={0: 0.0, 1: 0.0},
            )

        same_max_but_different_shape = copy.deepcopy(rank1)
        same_max_but_different_shape["traceEvents"][0]["args"]["In msg nelems"] = 512
        with self.assertRaisesRegex(SchemaError, "conflicting records"):
            kineto_traces_to_commcanary_trace(
                [rank0, same_max_but_different_shape],
                clock_offsets_us={0: 0.0, 1: 0.0},
            )

        duplicate_rank = copy.deepcopy(rank1)
        duplicate_rank["distributedInfo"]["rank"] = 0
        with self.assertRaisesRegex(SchemaError, "duplicate Kineto profile"):
            kineto_traces_to_commcanary_trace([rank0, duplicate_rank])

        unknown = copy.deepcopy(rank1)
        unknown["traceEvents"] = [
            raw
            for raw in unknown["traceEvents"]
            if not (
                raw.get("cat") == "kernel"
                and raw.get("args", {}).get("External id") == 1000
                and "nccl" in raw.get("name", "").lower()
            )
        ]
        unknown_trace = kineto_traces_to_commcanary_trace(
            [rank0, unknown],
            assume_shared_clock=True,
        )
        self.assertTrue(unknown_trace["events"][0]["compute_overlap_unknown"])
        self.assertTrue(unknown_trace["events"][0]["compute_by_rank"]["1"]["compute_overlap_unknown"])
        self.assertNotIn("reduction_op", unknown_trace["events"][0])
        self.assertEqual(
            unknown_trace["events"][0]["metadata"]["kineto_reduction_status"],
            "unavailable_incomplete_rank_evidence",
        )
        with self.assertRaisesRegex(SchemaError, "unknown compute overlap"):
            compile_trace(unknown_trace)

    def test_multi_rank_kineto_import_refuses_conflicting_broadcast_roots(self):
        profiles = []
        for rank, root_rank in ((0, 0), (1, 1)):
            profile = self._multi_rank_kineto_trace(rank, [100.0], [0.0])
            record = profile["traceEvents"][0]
            record["args"]["Collective name"] = "broadcast"
            profile["traceEvents"].insert(
                0,
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "c10d::broadcast_",
                    "pid": record["pid"],
                    "tid": record["tid"],
                    "ts": record["ts"] - 1.0,
                    "dur": record["dur"] + 2.0,
                    "args": {
                        "Concrete Inputs": ["", "", str(root_rank), "0", "True", "-1"],
                        "Input type": ["TensorList", "", "Scalar", "Scalar", "Scalar", "Scalar"],
                    },
                },
            )
            profiles.append(profile)

        with self.assertRaisesRegex(SchemaError, "conflicting records"):
            kineto_traces_to_commcanary_trace(
                profiles,
                assume_shared_clock=True,
            )

    def test_multi_rank_kineto_import_pairs_send_and_recv(self):
        profiles = []
        for rank, op in ((0, "send"), (2, "recv")):
            profile = self._synthetic_kineto_trace()
            source = copy.deepcopy(profile["traceEvents"][4])
            source["args"]["Collective name"] = op
            profile["distributedInfo"]["rank"] = rank
            profile["traceEvents"] = [
                source,
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": f"ncclDevKernel_{op.title()}",
                    "pid": 0,
                    "tid": 7,
                    "ts": source["ts"] + 2.0,
                    "dur": 5.0,
                    "args": {
                        "External id": source["args"]["External id"],
                        "device": 0,
                        "stream": 7,
                    },
                },
            ]
            profiles.append(profile)

        trace = kineto_traces_to_commcanary_trace(
            profiles,
            assume_shared_clock=True,
        )
        validate_trace(trace, require_known_overlap=True)
        self.assertEqual(len(trace["events"]), 1)
        event = trace["events"][0]
        self.assertEqual(event["op"], "point_to_point")
        self.assertEqual(event["ranks"], [0, 2])
        self.assertEqual(event["sender_rank"], 0)
        self.assertEqual(event["receiver_rank"], 2)
        self.assertEqual(event["message_sequence"], 9)
        compile_trace(trace)

    def test_kineto_import_keeps_overlap_unknown_for_ambiguous_kernel_evidence(self):
        kineto = self._synthetic_kineto_trace()
        kineto["traceEvents"][-1].update({"pid": 0, "tid": 7})
        kineto["traceEvents"][-1]["args"].update({"device": 0, "stream": 7})
        # Reusing an external id for two selected CPU collectives makes the
        # source-to-kernel ownership ambiguous rather than producing two zeros.
        kineto["traceEvents"][1]["args"]["External id"] = 1000
        trace = kineto_trace_to_commcanary_trace(kineto)
        statuses = [event["metadata"]["kineto_overlap_status"] for event in trace["events"]]
        self.assertEqual(statuses[:2], ["unavailable_nonunique_external_id"] * 2)
        self.assertTrue(all(event["compute_overlap_unknown"] for event in trace["events"]))

        malformed = self._synthetic_kineto_trace()
        malformed["traceEvents"][-1].update({"pid": 0, "tid": 7})
        malformed["traceEvents"][-1]["args"].update({"device": 0, "stream": 7})
        malformed["traceEvents"].append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "compute_without_stream_identity",
                "pid": 0,
                "ts": 110.0,
                "dur": 5.0,
                "args": {"device": 0},
            }
        )
        trace = kineto_trace_to_commcanary_trace(malformed)
        self.assertTrue(all(event["compute_overlap_unknown"] for event in trace["events"]))
        self.assertTrue(
            all(
                event["metadata"]["kineto_overlap_status"] == "unavailable_incomplete_kernel_activity"
                for event in trace["events"]
            )
        )

        zero_duration_communication = self._synthetic_kineto_trace()
        linked = zero_duration_communication["traceEvents"][-1]
        linked.update({"pid": 0, "tid": 7, "dur": 0.0})
        linked["args"].update({"device": 0, "stream": 7})
        trace = kineto_trace_to_commcanary_trace(zero_duration_communication)
        self.assertEqual(
            trace["events"][0]["metadata"]["kineto_overlap_status"],
            "unavailable_incomplete_linked_communication_kernel",
        )
        self.assertTrue(trace["events"][0]["compute_overlap_unknown"])

    def test_kineto_import_rebases_monotonic_scale_timestamps(self):
        kineto = self._synthetic_kineto_trace()
        # Monotonic-clock timestamps on a long-uptime host exceed the schema's
        # MAX_TIME_US unless the importer rebases to the trace start.
        base = 1.2e15
        for raw in kineto["traceEvents"]:
            raw["ts"] = base + raw["ts"]
        trace = kineto_trace_to_commcanary_trace(kineto)
        validate_trace(trace)
        self.assertEqual(trace["events"][0]["start_us"], 0.0)
        self.assertEqual(trace["events"][1]["start_us"], 120.5)
        self.assertNotIn(str(base + 100.0), json.dumps(trace))

    def test_kineto_import_reconstructs_ranks_and_fails_closed(self):
        kineto = self._synthetic_kineto_trace()
        args = kineto["traceEvents"][0]["args"]
        args["Process Group Ranks"] = "[0, 2, ...]"
        args["Global rank start"] = 0
        args["Global rank stride"] = 2
        trace = kineto_trace_to_commcanary_trace(kineto)
        self.assertEqual(trace["events"][0]["ranks"], [0, 2, 4, 6])

        with self.assertRaisesRegex(SchemaError, "no importable"):
            kineto_trace_to_commcanary_trace({"traceEvents": []})
        broken = self._synthetic_kineto_trace()
        broken["traceEvents"][0]["args"]["dtype"] = "MysteryType"
        with self.assertRaisesRegex(SchemaError, "unknown kineto dtype"):
            kineto_trace_to_commcanary_trace(broken)

    def test_kineto_import_refuses_to_fabricate_truncated_group_ranks(self):
        # Truncated rank list from a non-uniform group: torch omits Global
        # rank start/stride, so membership cannot be reconstructed.
        kineto = self._synthetic_kineto_trace()
        args = kineto["traceEvents"][0]["args"]
        args["Process Group Ranks"] = "[8, 9, 11, 14, ...]"
        with self.assertRaisesRegex(SchemaError, "refusing to fabricate"):
            kineto_trace_to_commcanary_trace(kineto)

        # A non-positive stride sentinel must not be coerced to 1.
        kineto = self._synthetic_kineto_trace()
        args = kineto["traceEvents"][0]["args"]
        args["Process Group Ranks"] = "[0, 2, 5, 9, ...]"
        args["Global rank start"] = 0
        args["Global rank stride"] = -1
        with self.assertRaisesRegex(SchemaError, "refusing to fabricate"):
            kineto_trace_to_commcanary_trace(kineto)

        # An entirely absent rank list is the world-group convention and is
        # allowed, but the assumption is flagged.
        kineto = self._synthetic_kineto_trace()
        for raw in kineto["traceEvents"]:
            raw.get("args", {}).pop("Process Group Ranks", None)
        trace = kineto_trace_to_commcanary_trace(kineto)
        self.assertEqual(trace["events"][0]["ranks"], [0, 1, 2, 3])
        self.assertTrue(trace["events"][0]["metadata"]["kineto_ranks_assumed"])

    def test_kineto_import_drops_nested_duplicate_collective_events(self):
        # torch >= 2.4 emits a frontend/backend record_param_comms pair per
        # collective, BOTH carrying named args; only the outer copy counts.
        kineto = self._synthetic_kineto_trace()
        outer = kineto["traceEvents"][0]
        inner = copy.deepcopy(outer)
        outer["dur"] = 50.0
        inner["ts"] = outer["ts"] + 5.0
        inner["dur"] = 20.0
        inner["args"] = dict(outer["args"])
        inner["args"]["External id"] = 9999
        kineto["traceEvents"].insert(1, inner)
        linked_kernel = kineto["traceEvents"][-1]
        linked_kernel.update({"pid": 0, "tid": 7})
        linked_kernel["args"].update(
            {
                "External id": 9999,
                "device": 0,
                "stream": 7,
            }
        )
        trace = kineto_trace_to_commcanary_trace(kineto)
        validate_trace(trace)
        self.assertEqual(trace["workload"]["skipped_nested_events"], 1)
        self.assertEqual(len(trace["events"]), 4)
        all_reduce_events = [e for e in trace["events"] if e["op"] == "all_reduce"]
        self.assertEqual(len(all_reduce_events), 1)
        self.assertEqual(all_reduce_events[0]["compute_overlap_us"], 0.0)
        self.assertEqual(all_reduce_events[0]["metadata"]["kineto_overlap_status"], "derived")

        # non-overlapping events are untouched
        plain = kineto_trace_to_commcanary_trace(self._synthetic_kineto_trace())
        self.assertEqual(plain["workload"]["skipped_nested_events"], 0)

    def test_import_kineto_cli_marks_overlap_unknown_and_compile_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            kineto_path = os.path.join(tmp, "kineto.json")
            trace_path = os.path.join(tmp, "imported.trace.json")
            with open(kineto_path, "w", encoding="utf-8") as handle:
                json.dump(self._synthetic_kineto_trace(), handle)
            source_bytes = Path(kineto_path).read_bytes()
            self.assertEqual(
                cli_main(
                    [
                        "import-kineto",
                        kineto_path,
                        "--workload-name",
                        "unit-import",
                        "--output",
                        trace_path,
                    ]
                ),
                0,
            )
            imported = load_json(trace_path)
            validate_trace(imported)
            self.assertEqual(imported["workload"]["name"], "unit-import")
            self.assertNotIn("kineto_trace_start_us", imported["workload"])
            self.assertNotIn("kineto_base_time_ns", imported["system"])
            self.assertEqual(
                imported["system"]["kineto_source_profiles"],
                [
                    {
                        "rank": 0,
                        "sha256": hashlib.sha256(source_bytes).hexdigest(),
                        "size_bytes": len(source_bytes),
                    }
                ],
            )
            self.assertNotIn("kineto.json", json.dumps(imported))
            self.assertNotIn(tmp, json.dumps(imported))
            self.assertTrue(all(event["compute_overlap_unknown"] for event in imported["events"]))
            with self.assertRaisesRegex(SchemaError, "unknown compute overlap"):
                compile_trace(imported)

    def test_import_kineto_cli_commits_exact_source_bytes_not_only_json_meaning(self):
        with tempfile.TemporaryDirectory() as tmp:
            kineto = self._synthetic_kineto_trace()
            compact_path = os.path.join(tmp, "private-compact-profile.json")
            pretty_path = os.path.join(tmp, "private-pretty-profile.json")
            compact_bytes = json.dumps(kineto, separators=(",", ":")).encode("utf-8")
            pretty_bytes = json.dumps(kineto, indent=2, sort_keys=True).encode("utf-8")
            with open(compact_path, "wb") as handle:
                handle.write(compact_bytes)
            with open(pretty_path, "wb") as handle:
                handle.write(pretty_bytes)

            imported = []
            for profile_path, label in (
                (compact_path, "compact"),
                (pretty_path, "pretty"),
            ):
                output_path = os.path.join(tmp, f"{label}.trace.json")
                self.assertEqual(
                    cli_main(["import-kineto", profile_path, "--output", output_path]),
                    0,
                )
                imported.append(load_json(output_path))

            compact_identity = imported[0]["system"].pop("kineto_source_profiles")
            pretty_identity = imported[1]["system"].pop("kineto_source_profiles")
            self.assertEqual(imported[0], imported[1])
            self.assertEqual(compact_identity[0]["size_bytes"], len(compact_bytes))
            self.assertEqual(pretty_identity[0]["size_bytes"], len(pretty_bytes))
            self.assertEqual(
                compact_identity[0]["sha256"],
                hashlib.sha256(compact_bytes).hexdigest(),
            )
            self.assertEqual(
                pretty_identity[0]["sha256"],
                hashlib.sha256(pretty_bytes).hexdigest(),
            )
            self.assertNotEqual(
                compact_identity[0]["sha256"],
                pretty_identity[0]["sha256"],
            )

    def test_import_kineto_cli_merges_multiple_rank_profiles_with_explicit_clock_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_paths = []
            source_identities = {}
            for rank, starts, overlaps in (
                (0, [100.0, 200.0], [5.0, 0.0]),
                (1, [104.0, 203.0], [2.0, 0.0]),
            ):
                profile_path = os.path.join(tmp, f"rank-{rank}.json")
                with open(profile_path, "w", encoding="utf-8") as handle:
                    json.dump(self._multi_rank_kineto_trace(rank, starts, overlaps), handle)
                profile_paths.append(profile_path)
                source_bytes = Path(profile_path).read_bytes()
                source_identities[rank] = {
                    "rank": rank,
                    "sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "size_bytes": len(source_bytes),
                }
            trace_path = os.path.join(tmp, "multi-rank.trace.json")

            self.assertEqual(
                cli_main(
                    [
                        "import-kineto",
                        *reversed(profile_paths),
                        "--assume-shared-clock",
                        "--output",
                        trace_path,
                    ]
                ),
                0,
            )
            imported = load_json(trace_path)
            self.assertEqual(imported["system"]["clock_alignment"], "assumed_shared_clock")
            self.assertNotIn("kineto_trace_start_us_by_rank", imported["workload"])
            self.assertTrue(all("kineto_base_time_ns" not in shard for shard in imported["system"]["shard_systems"]))
            self.assertEqual(
                imported["system"]["kineto_source_profiles"],
                [source_identities[0], source_identities[1]],
            )
            self.assertEqual(imported["events"][0]["rank_arrival_us"], {"0": 0.0, "1": 4.0})
            canary = compile_trace(imported)
            self.assertTrue(all(event["dtype"] == "float32" for event in canary["events"]))
            self.assertEqual(
                canary["system"]["kineto_source_profiles"],
                [source_identities[0], source_identities[1]],
            )

            invalid_path = os.path.join(tmp, "invalid.trace.json")
            self.assertEqual(
                cli_main(
                    [
                        "import-kineto",
                        *profile_paths,
                        "--clock-offset-us",
                        "0=0",
                        "--output",
                        invalid_path,
                    ]
                ),
                3,
            )
            self.assertFalse(os.path.exists(invalid_path))

    def test_prepare_qualification_cli_builds_and_recipient_verifies_portable_request(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            profile_paths = []
            for rank, starts, overlaps in (
                (0, [100.0, 200.0], [5.0, 0.0]),
                (1, [104.0, 203.0], [2.0, 0.0]),
            ):
                profile_path = os.path.join(tmp, f"rank-{rank}.json")
                profile = self._multi_rank_kineto_trace(rank, starts, overlaps)
                for index, start_us in enumerate(starts):
                    compute_external_id = 3000 + index
                    profile["traceEvents"].extend(
                        [
                            {
                                "ph": "X",
                                "cat": "cpu_op",
                                "name": "aten::mm",
                                "pid": 7,
                                "tid": 7,
                                "ts": start_us + 11.0,
                                "dur": 20.0,
                                "args": {
                                    "External id": compute_external_id,
                                    "Input Dims": [[rank + 2, 8], [8, 8]],
                                    "Input Strides": [[8, 1], [8, 1]],
                                    "Input type": ["BFloat16", "BFloat16"],
                                },
                            },
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": "ampere_bf16_gemm",
                                "pid": 0,
                                "tid": 1,
                                "ts": start_us + 20.0,
                                "dur": 5.0,
                                "args": {
                                    "External id": compute_external_id,
                                    "device": 0,
                                    "stream": 1,
                                },
                            },
                            {
                                "ph": "X",
                                "cat": "cpu_op",
                                "name": "record_param_comms",
                                "pid": 7,
                                "tid": 7,
                                "ts": start_us + 50.0,
                                "dur": 2.0,
                                "args": {
                                    "External id": 4000 + index,
                                    "Collective name": "wait",
                                    "dtype": "Byte",
                                    "In msg nelems": 0,
                                    "Out msg nelems": 0,
                                    "Group size": 1,
                                    "Process Group Ranks": "[]",
                                },
                            },
                        ]
                    )
                with open(profile_path, "w", encoding="utf-8") as handle:
                    json.dump(profile, handle)
                profile_paths.append(profile_path)
            bundle = os.path.join(tmp, "qualification-request")

            self.assertEqual(
                cli_main(
                    [
                        "prepare-qualification",
                        *reversed(profile_paths),
                        "--assume-shared-clock",
                        "--output-directory",
                        bundle,
                    ]
                ),
                0,
            )
            request = load_json(os.path.join(bundle, "qualification-request.json"))
            request_canary = load_json(os.path.join(bundle, "canary.json"))
            self.assertEqual(request["claims"]["source_correspondence"], "source_verified")
            self.assertEqual(request["claims"]["physical_fidelity"], "unproven")
            self.assertEqual(
                request_canary["compiler"]["fidelity"]["mode"],
                "lossless_timing",
            )
            self.assertFalse(os.path.exists(os.path.join(bundle, "param.json")))
            self.assertEqual(cli_main(["verify-qualification", bundle]), 0)

            uncalibrated_bundle = os.path.join(tmp, "uncalibrated-request")
            self.assertEqual(
                cli_main(
                    [
                        "prepare-qualification",
                        *profile_paths,
                        "--output-directory",
                        uncalibrated_bundle,
                    ]
                ),
                3,
            )
            self.assertFalse(os.path.exists(uncalibrated_bundle))

    def test_param_export_expands_canary_into_replayable_entries(self):
        trace = small_trace()
        canary = compile_trace(trace)
        entries = canary_to_param_comms_trace(canary)
        # one PG init entry (required by PARAM's groupRanks registry) + body
        self.assertEqual(len(entries), len(trace["events"]) + 1)
        init = entries[0]
        self.assertEqual(init["comms"], "init")
        self.assertEqual(init["pg_id"], 0)
        self.assertEqual(init["global_ranks"], [0, 1, 2, 3])
        body = entries[1:]
        self.assertTrue(all(entry["comms"] == "all_reduce" for entry in body))
        self.assertTrue(all(entry["dtype"] == "float32" for entry in body))
        self.assertTrue(all(entry["in_msg_size"] == (128 * 1024) // 4 for entry in body))
        self.assertTrue(all(entry["world_size"] == 4 for entry in body))
        self.assertEqual([entry["req"] for entry in entries], list(range(len(entries))))
        start_times = [entry["startTime_ns"] for entry in body]
        self.assertEqual(start_times, sorted(start_times))
        self.assertGreater(start_times[-1], start_times[0])

    def test_param_export_preserves_source_bound_dtype_per_event(self):
        trace = small_trace()
        for index, event in enumerate(trace["events"]):
            event["dtype"] = "float16" if index < 3 else "float32"
        canary = compile_trace(trace)

        self.assertEqual([event["dtype"] for event in canary["events"]], ["float16", "float32"])
        entries = canary_to_param_comms_trace(canary)
        body = [entry for entry in entries if entry.get("comms") == "all_reduce"]
        self.assertEqual([entry["dtype"] for entry in body], ["float16"] * 3 + ["float32"] * 3)
        self.assertEqual(body[0]["in_msg_size"], (128 * 1024) // 2)
        self.assertEqual(body[-1]["in_msg_size"], (128 * 1024) // 4)

        overridden = canary_to_param_comms_trace(canary, dtype="bfloat16")
        overridden_body = [entry for entry in overridden if entry.get("comms") == "all_reduce"]
        self.assertTrue(all(entry["dtype"] == "bfloat16" for entry in overridden_body))

    def test_param_export_pairs_p2p_and_rejects_unsupported_ops(self):
        trace = small_trace()
        trace["events"][0] = {
            "id": "p2p-0",
            "phase": "decode",
            "op": "point_to_point",
            "bytes": 64 * 1024,
            "ranks": [0, 1],
            "group": "pp0",
            "start_us": trace["events"][0]["start_us"],
            "compute_overlap_us": 0.0,
            "sender_rank": 0,
            "receiver_rank": 1,
        }
        canary = compile_trace(trace)
        entries = canary_to_param_comms_trace(canary)
        # 2 PG inits (pp0, tp0) + send/recv pair + 5 all_reduce
        self.assertEqual(len(entries), len(trace["events"]) + 3)
        self.assertEqual([e["comms"] for e in entries[:2]], ["init", "init"])
        send_entry, recv_entry = entries[2], entries[3]
        self.assertEqual(send_entry["comms"], "send")
        self.assertEqual(recv_entry["comms"], "recv")
        for entry in (send_entry, recv_entry):
            self.assertEqual(entry["src_rank"], 0)
            self.assertEqual(entry["dst_rank"], 1)
            self.assertIs(entry["use_batch"], False)
        self.assertEqual(send_entry["startTime_ns"], recv_entry["startTime_ns"])
        self.assertNotEqual(send_entry["req"], recv_entry["req"])

        # send/recv ops without peer ranks cannot produce a parseable PARAM
        # trace and must fail closed.
        one_sided = small_trace()
        one_sided["events"][0]["op"] = "send"
        one_sided_canary = compile_trace(one_sided)
        with self.assertRaisesRegex(SchemaError, "sender_rank and receiver_rank"):
            canary_to_param_comms_trace(one_sided_canary)
        dropped = canary_to_param_comms_trace(one_sided_canary, skip_unsupported=True)
        self.assertEqual(len(dropped), len(one_sided["events"]) - 1 + 1)  # + pg init

        custom = small_trace()
        custom["events"][0]["op"] = "mystery_collective"
        custom["events"][0]["custom_op"] = True
        custom_canary = compile_trace(custom)
        with self.assertRaisesRegex(SchemaError, "no PARAM comms-replay equivalent"):
            canary_to_param_comms_trace(custom_canary)
        skipped = canary_to_param_comms_trace(custom_canary, skip_unsupported=True)
        self.assertEqual(len(skipped), len(custom["events"]) - 1 + 1)  # + pg init

    def test_param_export_uses_asymmetric_sizes_for_sharded_collectives(self):
        trace = small_trace()
        trace["events"][0]["op"] = "all_gather"
        trace["events"][1]["op"] = "reduce_scatter"
        canary = compile_trace(trace)
        entries = canary_to_param_comms_trace(canary)
        by_op = {entry["comms"]: entry for entry in entries}
        nelems = (128 * 1024) // 4
        shard = nelems // 4
        self.assertEqual(by_op["all_gather"]["in_msg_size"], shard)
        self.assertEqual(by_op["all_gather"]["out_msg_size"], nelems)
        self.assertEqual(by_op["reduce_scatter"]["in_msg_size"], nelems)
        self.assertEqual(by_op["reduce_scatter"]["out_msg_size"], shard)
        self.assertEqual(by_op["all_reduce"]["in_msg_size"], nelems)
        self.assertEqual(by_op["all_reduce"]["out_msg_size"], nelems)

        indivisible = small_trace()
        indivisible["events"][0]["op"] = "all_gather"
        indivisible["events"][0]["bytes"] = 6
        with self.assertRaisesRegex(SchemaError, "divide evenly"):
            canary_to_param_comms_trace(compile_trace(indivisible))

        indivisible_all_to_all = small_trace()
        indivisible_all_to_all["events"][0]["op"] = "all_to_all"
        indivisible_all_to_all["events"][0]["dtype"] = "float32"
        indivisible_all_to_all["events"][0]["bytes"] = 20
        with self.assertRaisesRegex(SchemaError, "all_to_all event bytes do not divide evenly"):
            canary_to_param_comms_trace(compile_trace(indivisible_all_to_all))

        dtype_indivisible = small_trace()
        dtype_indivisible["events"][0]["dtype"] = "float32"
        dtype_indivisible["events"][0]["bytes"] = 6
        with self.assertRaisesRegex(SchemaError, "element width=4"):
            canary_to_param_comms_trace(compile_trace(dtype_indivisible))

    def test_param_export_compute_fill_converts_gaps_to_gemm_entries(self):
        trace = small_trace()  # events 40us apart -> gaps 0,40,40,40,40,40
        canary = compile_trace(trace)
        entries = canary_to_param_comms_trace(canary, compute_fill_us_per_gemm=10.0, compute_fill_gemm_dim=512)
        gemms = [e for e in entries if e.get("compute") == "gemm"]
        comms = [e for e in entries if e.get("comms") == "all_reduce"]
        # Every occurrence carries rank-specific arrival fill. The first has
        # base count 0; the other five get 40/10 = 4.
        self.assertEqual(len(comms), 6)
        self.assertEqual(len(gemms), 6)
        self.assertEqual([g["count"] for g in gemms], [0, 4, 4, 4, 4, 4])
        self.assertTrue(all(g["global_ranks"] == [0, 1, 2, 3] for g in gemms))
        self.assertTrue(all(g["rank_extra_counts"] == {"0": 0, "1": 0, "2": 0, "3": 1} for g in gemms))
        self.assertTrue(all(g["mm_dim"] == 512 for g in gemms))
        self.assertEqual(
            audit_param_program_compute_operations(entries),
            {
                "gemm_entry_count": 6,
                "base_gemm_operation_count": 20,
                "rank_extra_gemm_operation_count": 6,
                "total_rank_gemm_operation_count": 86,
                "max_rank_gemm_operation_count": 26,
                "rank_gemm_operation_counts": {
                    "0": 20,
                    "1": 20,
                    "2": 20,
                    "3": 26,
                },
            },
        )
        # a gemm entry directly precedes each filled collective
        for index, entry in enumerate(entries):
            if entry.get("compute") == "gemm":
                self.assertEqual(entries[index + 1].get("comms"), "all_reduce")
        # req ids remain unique and sequential across init+compute+comm entries
        self.assertEqual([e["req"] for e in entries], list(range(len(entries))))

        # without the flag, no compute entries are emitted
        plain = canary_to_param_comms_trace(canary)
        self.assertFalse(any("compute" in e for e in plain))

        with self.assertRaisesRegex(SchemaError, "compute_fill_us_per_gemm"):
            canary_to_param_comms_trace(canary, compute_fill_us_per_gemm=0.0)
        with self.assertRaisesRegex(SchemaError, "compute_fill_dtype requires"):
            canary_to_param_comms_trace(canary, compute_fill_dtype="bfloat16")
        with self.assertRaisesRegex(SchemaError, "supported compute dtypes"):
            canary_to_param_comms_trace(
                canary,
                compute_fill_us_per_gemm=10.0,
                compute_fill_dtype="int64",
            )
        with self.assertRaisesRegex(SchemaError, "max_param_gemm_dim"):
            canary_to_param_comms_trace(
                canary,
                compute_fill_us_per_gemm=10.0,
                compute_fill_gemm_dim=16_385,
            )

    def test_param_export_overlap_structure_interleaves_issue_gemm_wait(self):
        trace = small_trace()  # gaps: 0, then 40us x5
        canary = compile_trace(trace)
        entries = canary_to_param_comms_trace(
            canary,
            compute_fill_us_per_gemm=10.0,
            compute_fill_gemm_dim=512,
            overlap_structure=True,
        )
        kinds = ["gemm" if e.get("compute") == "gemm" else e.get("comms") for e in entries]
        # Only the source overlap component precedes the prior wait. The
        # remainder and rank-arrival fill are serialized after it.
        self.assertEqual(
            kinds,
            ["init", "gemm", "all_reduce"] + ["gemm", "wait", "gemm", "all_reduce"] * 5 + ["gemm", "wait"],
        )
        comms = [e for e in entries if e.get("comms") == "all_reduce"]
        waits = [e for e in entries if e.get("comms") == "wait"]
        overlap_compute = [e for e in entries if e.get("compute_phase") == "source-overlap"]
        serialized_compute = [e for e in entries if e.get("compute_phase") == "serialized-readiness"]
        self.assertEqual(len(waits), len(comms))
        self.assertEqual(len(overlap_compute), len(comms))
        self.assertEqual(len(serialized_compute), len(comms))
        self.assertTrue(all(e["count"] == 2 for e in overlap_compute))
        self.assertTrue(all(not any(e["rank_extra_counts"].values()) for e in overlap_compute))
        # each wait's req equals its issuing collective's req, in order
        self.assertEqual([w["req"] for w in waits], [c["req"] for c in comms])
        self.assertEqual(
            [e["overlap_request"] for e in overlap_compute],
            [c["req"] for c in comms],
        )
        # issue lines are marked so parsers do not read them as completions
        self.assertTrue(all("issue" in c["markers"][0] for c in comms))
        self.assertTrue(all("issue" not in w["markers"][0] for w in waits))

        with self.assertRaisesRegex(SchemaError, "requires compute_fill"):
            canary_to_param_comms_trace(canary, overlap_structure=True)

        pipelined = small_trace()
        pipelined["events"][0]["compute_overlap_us"] = 41.0
        with self.assertRaisesRegex(
            SchemaError,
            r"overlap_us=41\.0.*gap_us=40\.0.*requires an explicit dependency graph",
        ):
            canary_to_param_comms_trace(
                compile_trace(pipelined),
                compute_fill_us_per_gemm=10.0,
                overlap_structure=True,
            )

        zero_overlap = small_trace()
        zero_overlap["events"][0]["compute_overlap_us"] = 0.0
        zero_entries = canary_to_param_comms_trace(
            compile_trace(zero_overlap),
            compute_fill_us_per_gemm=10.0,
            overlap_structure=True,
        )
        first_issue = next(e for e in zero_entries if e.get("comms") == "all_reduce")
        first_wait_index = next(
            index
            for index, entry in enumerate(zero_entries)
            if entry.get("comms") == "wait" and entry["req"] == first_issue["req"]
        )
        following_serialized_index = next(
            index
            for index, entry in enumerate(zero_entries[first_wait_index + 1 :], first_wait_index + 1)
            if entry.get("compute_phase") == "serialized-readiness"
        )
        self.assertLess(first_wait_index, following_serialized_index)

    def test_export_param_cli_writes_json_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            canary_path = os.path.join(tmp, "canary.json")
            output_path = os.path.join(tmp, "param.json")
            write_json(canary_path, compile_trace(small_trace()))
            self.assertEqual(
                cli_main(["export-param", canary_path, "--output", output_path]),
                0,
            )
            with open(output_path, "r", encoding="utf-8") as handle:
                entries = json.load(handle)
            self.assertIsInstance(entries, list)
            self.assertEqual(len(entries), len(small_trace()["events"]) + 1)
            self.assertEqual(entries[0]["comms"], "init")
            self.assertEqual(entries[1]["comms"], "all_reduce")
