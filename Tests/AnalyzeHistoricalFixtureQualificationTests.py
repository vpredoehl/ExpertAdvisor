#!/usr/bin/env python3
"""Offline rejection checks using only the retained genuine experiment 746 corpus."""
import json
from pathlib import Path
import sys
import unittest

from AnalyzeHistoricalFixtureQualification import verify_chain, verify_analysis, copy_sql
from AnalyzeResourceQualificationPreflight import copy_rows, ROOT

EVIDENCE = ROOT / "DerivedData/ExpertAdvisor/Phase24X/Single746-20261008-02"


class GenuineFixtureTests(unittest.TestCase):
    def setUp(self):
        raw = (EVIDENCE / "data.sql").read_text()
        self.experiment, = copy_rows(raw, "experiment")
        self.model, = copy_rows(raw, "model")
        self.inference, = copy_rows(raw, "inference_eval_result")
        self.attempts = {row["worker_attempt_id"]: row for row in copy_rows(raw, "experiment_scheduler_worker_attempt")}
        self.train = (EVIDENCE / self.experiment["train_log_path"]).read_text()
        self.infer = (EVIDENCE / self.experiment["infer_log_path"]).read_text()

    def verify(self):
        verify_chain(self.experiment, self.model, self.inference, self.attempts, self.train, self.infer)

    def test_genuine_chain(self):
        self.verify()

    def test_inference_from_another_model_rejected(self):
        self.inference["model_id"] = "2088"
        with self.assertRaises(AssertionError):
            self.verify()

    def test_changed_inference_dates_rejected(self):
        self.inference["from_date"] = "2024-01-01"
        with self.assertRaises(AssertionError):
            self.verify()

    def test_incomplete_training_log_rejected(self):
        self.train = self.train[:len(self.train) // 2]
        with self.assertRaises(AssertionError):
            self.verify()

    def test_inference_without_persistence_completion_rejected(self):
        self.infer = self.infer[:self.infer.index("stage=persistence_transaction_committed")]
        with self.assertRaises(AssertionError):
            self.verify()

    def test_failed_producer_rejected(self):
        self.attempts[self.model["producer_worker_attempt_id"]]["exit_code"] = "1"
        with self.assertRaises(AssertionError):
            self.verify()

    def test_missing_producer_rejected(self):
        del self.attempts[self.inference["producer_worker_attempt_id"]]
        with self.assertRaises(KeyError):
            self.verify()

    def test_copy_roundtrip_preserves_archived_fields(self):
        for table, rows in (("experiment", [self.experiment]), ("model", [self.model]),
                            ("inference_eval_result", [self.inference])):
            self.assertEqual(copy_rows(copy_sql(table, rows), table), rows)

    def test_native_result_preserves_all_scientific_fields(self):
        result = json.loads((EVIDENCE / "single-worker-results.json").read_text())
        expected, = copy_rows((EVIDENCE / "data.sql").read_text(), "experiment_analysis_result")
        verify_analysis(result["analysis_result"], expected)
        result["analysis_result"]["confusion_down_down"] = 1
        with self.assertRaises(AssertionError):
            verify_analysis(result["analysis_result"], expected)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        EVIDENCE = Path(sys.argv.pop()).resolve()
    unittest.main()
