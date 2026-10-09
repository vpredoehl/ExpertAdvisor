#!/usr/bin/env python3
"""Offline regressions for the authentic layout-8 experiment 681 fixture.

Negative cases mutate only in-memory copies of genuine retained evidence.
Pass a final instrumented evidence directory to also verify native replay.
"""
import json
from pathlib import Path
import sys
import unittest

from AnalyzeHistoricalFixtureQualification import (
    ROOT, ROUTING, verify_chain, verify_analysis_source, verify_analysis,
    select_fixture_registry, select_snapshot_invocations, sha)
from AnalyzeResourceQualificationPreflight import copy_rows

EVIDENCE = ROOT / "DerivedData/ExpertAdvisor/Phase24X/Fixture681-20261008-01"


class SecondFixtureTests(unittest.TestCase):
    def setUp(self):
        raw = (EVIDENCE / "data.sql").read_text()
        self.experiment, = copy_rows(raw, "experiment")
        self.model, = copy_rows(raw, "model")
        self.inference, = copy_rows(raw, "inference_eval_result")
        self.analysis, = copy_rows(raw, "experiment_analysis_result")
        self.attempts = {x["worker_attempt_id"]: x for x in copy_rows(raw, "experiment_scheduler_worker_attempt")}
        self.train = (EVIDENCE / self.experiment["train_log_path"]).read_text()
        self.infer = (EVIDENCE / self.experiment["infer_log_path"]).read_text()
        self.fixture = json.loads((EVIDENCE / "fixture.json").read_text())
        self.fixture["producer_worker_attempt_ids"] = {
            "train": self.model["producer_worker_attempt_id"],
            "infer": self.inference["producer_worker_attempt_id"]}
        self.registry = json.loads((ROUTING / "registry.json").read_text())

    def verify(self):
        verify_chain(self.experiment, self.model, self.inference, self.attempts, self.train, self.infer)
        verify_analysis_source(self.analysis, self.experiment, self.model, self.inference)

    def test_genuine_second_chain_and_accepted_baseline(self):
        self.verify()
        self.assertEqual((self.experiment["experiment_id"], self.model["model_id"], self.inference["id"]),
                         ("681", "1982", "979"))
        self.assertEqual((self.analysis["accept_count"], self.analysis["reject_count"]), ("1", "0"))
        self.assertEqual((self.experiment["model_input_width"], self.experiment["model_input_semantic_layout_version"]), ("80", "8"))
        self.assertGreater(sum(x["bytes"] for x in self.fixture["logs"]), 14461)

    def test_registry_keeps_current_and_exact_historical_producers(self):
        selected = select_fixture_registry(self.registry, self.fixture)
        self.assertEqual(selected["current_layout"], self.registry["current_layout"])
        self.assertEqual([w for w in selected["workers"] if w["worker_rule"] == "current"],
                         [w for w in self.registry["workers"] if w["worker_rule"] == "current"])
        self.assertEqual(len(selected["workers"]), 4)
        for worker in selected["workers"]:
            self.assertEqual(sha(ROUTING / worker["executable"]), worker["sha256"])
        historical = {w["worker_role"]: w for w in selected["workers"] if w["semantic_layout"] == 8}
        for role, producer in self.fixture["producer_worker_attempt_ids"].items():
            self.assertEqual(historical[role]["sha256"], self.attempts[producer]["executable_sha256"])

    def test_registry_does_not_substitute_wrong_source(self):
        next(x for x in self.fixture["historical_attempts"] if x["worker_attempt_id"] == "1238")["source_commit"] = "wrong"
        with self.assertRaises(ValueError):
            select_fixture_registry(self.registry, self.fixture)

    def test_registry_does_not_substitute_wrong_sha(self):
        next(x for x in self.fixture["historical_attempts"] if x["worker_attempt_id"] == "1228")["executable_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            select_fixture_registry(self.registry, self.fixture)

    def test_missing_inference_producer_route_rejected(self):
        self.registry["workers"] = [w for w in self.registry["workers"] if not (w["semantic_layout"] == 8 and w["worker_role"] == "infer")]
        with self.assertRaises(ValueError):
            select_fixture_registry(self.registry, self.fixture)

    def test_missing_runtime_rejected(self):
        self.registry["runtimes"] = []
        with self.assertRaises(AssertionError):
            select_fixture_registry(self.registry, self.fixture)

    def test_incomplete_producer_mapping_rejected(self):
        del self.fixture["producer_worker_attempt_ids"]["infer"]
        with self.assertRaises(AssertionError):
            select_fixture_registry(self.registry, self.fixture)

    def test_analysis_from_wrong_model_rejected_before_replay(self):
        self.analysis["model_id"] = "2090"
        with self.assertRaises(AssertionError): self.verify()

    def test_mismatched_analysis_inference_result_rejected(self):
        self.analysis["infer_accuracy"] = "0.5510282005127366"
        with self.assertRaises(AssertionError): self.verify()

    def test_changed_ablation_evidence_rejected(self):
        self.train = self.train.replace("effective_mask=tg4_inner_break_any", "effective_mask=changed")
        with self.assertRaises(AssertionError): self.verify()

    def test_changed_layout_log_evidence_rejected(self):
        self.train = self.train.replace("model_input_width=80,semantic_layout=8", "model_input_width=171,semantic_layout=13")
        with self.assertRaises(AssertionError): self.verify()

    def test_changed_calendar_log_evidence_rejected(self):
        self.infer = self.infer.replace("content_hash=fnv1a64:67610f94f5c8e7cc", "content_hash=changed")
        with self.assertRaises(AssertionError): self.verify()

    def test_changed_training_seed_provenance_rejected(self):
        self.attempts["1228"]["command_line"] = self.attempts["1228"]["command_line"].replace("--fresh-initialization-seed=46", "--fresh-initialization-seed=1002")
        with self.assertRaises(AssertionError): self.verify()

    def test_incomplete_inference_persistence_rejected(self):
        self.infer = self.infer[:self.infer.index("stage=persistence_transaction_committed")]
        with self.assertRaises(AssertionError): self.verify()

    def test_snapshot_includes_original_released_lease_owner(self):
        rows = json.loads((ROOT / "DerivedData/ExpertAdvisor/Phase24X/Prerequisites-20261008-01/precutover-experiment_scheduler_invocation.json").read_text())
        leases = copy_rows((EVIDENCE / "data.sql").read_text(), "experiment_scheduler_lease")
        selected = select_snapshot_invocations(rows,list(self.attempts.values()),leases)
        self.assertEqual(len(selected),3)
        self.assertIn(leases[0]["owner_scheduler_invocation_id"], {r["scheduler_invocation_id"] for r in selected})
        self.assertTrue(all(r["ended_at"] != "\\N" for r in selected))

    def test_snapshot_rejects_missing_lease_owner(self):
        raw = (EVIDENCE / "data.sql").read_text()
        rows = copy_rows(raw,"experiment_scheduler_invocation")
        leases = copy_rows(raw,"experiment_scheduler_lease")
        rows = [r for r in rows if r["scheduler_invocation_id"] != leases[0]["owner_scheduler_invocation_id"]]
        with self.assertRaises(AssertionError):
            select_snapshot_invocations(rows,list(self.attempts.values()),leases)

    def test_native_replay_preserves_every_scientific_field(self):
        path = EVIDENCE / "single-worker-results.json"
        if not path.exists(): self.skipTest("offline preparation only; supply final native evidence for replay comparison")
        result = json.loads(path.read_text())
        self.assertEqual(result["single_worker"], "PASS")
        self.assertEqual(result["resource_qualification"], "PASS")
        self.assertEqual(result["cleanup"], "PASS")
        verify_analysis(result["analysis_result"], self.analysis)
        result["analysis_result"]["accept_count"] = 0
        with self.assertRaises(AssertionError):
            verify_analysis(result["analysis_result"], self.analysis)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        EVIDENCE = Path(sys.argv.pop()).resolve()
    unittest.main()
