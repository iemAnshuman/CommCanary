from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest

from commcanary.cli import main as cli_main
from commcanary.compiler import compile_trace
from commcanary.interop import canary_to_param_comms_trace, kineto_trace_to_commcanary_trace
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

    def test_kineto_import_maps_collectives_to_trace_events(self):
        trace = kineto_trace_to_commcanary_trace(self._synthetic_kineto_trace(), phase="decode")
        validate_trace(trace)
        events = trace["events"]
        self.assertEqual(len(events), 4)
        self.assertEqual([event["op"] for event in events], ["all_reduce", "all_gather", "reduce", "send"])
        self.assertEqual(events[0]["bytes"], 1024 * 4)
        self.assertEqual(events[1]["bytes"], 2048 * 2)
        self.assertEqual(events[3]["bytes"], 256 * 2)
        self.assertTrue(events[2].get("custom_op"))
        self.assertEqual(events[0]["ranks"], [0, 1, 2, 3])
        self.assertEqual(events[0]["group"], "0")
        self.assertEqual(events[0]["start_us"], 0.0)
        self.assertEqual(events[1]["start_us"], 120.5)
        self.assertEqual(events[3]["start_us"], 400.25)
        self.assertEqual(trace["workload"]["kineto_trace_start_us"], 100.0)
        self.assertEqual(trace["system"]["kineto_base_time_ns"], 1000000)
        self.assertEqual(events[0]["phase"], "decode")
        self.assertEqual(events[3]["metadata"]["kineto_dst_rank"], 2)
        self.assertEqual(events[3]["sender_rank"], 0)
        self.assertEqual(events[3]["receiver_rank"], 2)
        self.assertEqual(trace["workload"]["skipped_control_events"], 1)
        self.assertEqual(trace["system"]["kineto_nccl_version"], "2.27.5")
        compile_trace(trace)

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
        self.assertEqual(trace["workload"]["kineto_trace_start_us"], base + 100.0)

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
        trace = kineto_trace_to_commcanary_trace(kineto)
        validate_trace(trace)
        self.assertEqual(trace["workload"]["skipped_nested_events"], 1)
        self.assertEqual(len(trace["events"]), 4)
        all_reduce_events = [e for e in trace["events"] if e["op"] == "all_reduce"]
        self.assertEqual(len(all_reduce_events), 1)

        # non-overlapping events are untouched
        plain = kineto_trace_to_commcanary_trace(self._synthetic_kineto_trace())
        self.assertEqual(plain["workload"]["skipped_nested_events"], 0)

    def test_import_kineto_cli_round_trips_through_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            kineto_path = os.path.join(tmp, "kineto.json")
            trace_path = os.path.join(tmp, "imported.trace.json")
            with open(kineto_path, "w", encoding="utf-8") as handle:
                json.dump(self._synthetic_kineto_trace(), handle)
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
            canary = compile_trace(imported)
            self.assertEqual(canary["compiler"]["source_events"], 4)

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

    def test_param_export_compute_fill_converts_gaps_to_gemm_entries(self):
        trace = small_trace()  # events 40us apart -> gaps 0,40,40,40,40,40
        canary = compile_trace(trace)
        entries = canary_to_param_comms_trace(canary, compute_fill_us_per_gemm=10.0, compute_fill_gemm_dim=512)
        gemms = [e for e in entries if e.get("compute") == "gemm"]
        comms = [e for e in entries if e.get("comms") == "all_reduce"]
        # first occurrence has gap 0 -> no fill; the other five get 40/10 = 4
        self.assertEqual(len(comms), 6)
        self.assertEqual(len(gemms), 5)
        self.assertTrue(all(g["count"] == 4 for g in gemms))
        self.assertTrue(all(g["mm_dim"] == 512 for g in gemms))
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
        # init, comm0(async), then per occurrence: gemm, wait(prev), comm;
        # trailing wait for the final collective.
        self.assertEqual(
            kinds,
            ["init", "all_reduce"] + ["gemm", "wait", "all_reduce"] * 5 + ["wait"],
        )
        comms = [e for e in entries if e.get("comms") == "all_reduce"]
        waits = [e for e in entries if e.get("comms") == "wait"]
        self.assertEqual(len(waits), len(comms))
        # each wait's req equals its issuing collective's req, in order
        self.assertEqual([w["req"] for w in waits], [c["req"] for c in comms])
        # issue lines are marked so parsers do not read them as completions
        self.assertTrue(all("issue" in c["markers"][0] for c in comms))
        self.assertTrue(all("issue" not in w["markers"][0] for w in waits))

        with self.assertRaisesRegex(SchemaError, "requires compute_fill"):
            canary_to_param_comms_trace(canary, overlap_structure=True)

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
