#!/usr/bin/env python3

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Scripts" / "phase19c_causal_path_predictability.py"
SPEC = importlib.util.spec_from_file_location("phase19c", SCRIPT)
phase19c = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
import sys
sys.modules[SPEC.name] = phase19c
SPEC.loader.exec_module(phase19c)


PREDICTORS = (
    "predictor__predicted_class", "predictor__direction",
    "predictor__directional_probability",
    "predictor__normalized_directional_confidence", "predictor__causal_x")
CATEGORICAL = frozenset(("predictor__predicted_class", "predictor__direction"))


def row(outcome, confidence, causal, direction="long"):
    return {
        "outcome__outcome_class": outcome,
        "predictor__predicted_class": "2" if direction == "long" else "0",
        "predictor__direction": direction,
        "predictor__directional_probability": str((confidence * 2 / 3) + 1 / 3),
        "predictor__normalized_directional_confidence": str(confidence),
        "predictor__causal_x": str(causal),
    }


def run(model, start, rows, role="primary"):
    end = "2026-01-01" if start == "2025-01-01" else "2026-09-01"
    return phase19c.RunData(
        model, start, end, role, tuple(rows), PREDICTORS, CATEGORICAL,
        {"joined_count": str(len(rows)), "join_failure_count": "0",
         "feature_layout_identity": "fixture"}, Path("fixture"))


def frozen_runs(diagnostic_causal=1000.0):
    result = []
    for model in phase19c.PRIMARY_MODELS:
        result.append(run(model, "2025-01-01", (
            row("SAVED", 0.8, 1.0), row("HARMED", 0.6, -1.0),
            row("UNCHANGED", 0.7, 99.0))))
        result.append(run(model, "2026-01-01", (
            row("SAVED", 0.7, 0.5), row("HARMED", 0.65, -0.5),
            row("UNCHANGED", 0.9, -99.0))))
    result.append(run(1805, "2025-01-01", (
        row("SAVED", 0.1, diagnostic_causal),
        row("HARMED", 0.9, -diagnostic_causal)), "diagnostic"))
    result.append(run(1805, "2026-01-01", (
        row("SAVED", 0.1, diagnostic_causal),
        row("HARMED", 0.9, -diagnostic_causal)), "diagnostic"))
    return result


def write_empty_dataset(directory: Path):
    stem = "model1729_USDCAD_H6_2025"
    dataset = directory / f"{stem}.phase19c.dataset.tsv"
    dataset_header = (
        "schema_identity", "schema_version", "model_id", "experiment_id",
        "symbol", "cohort_role", "prediction_horizon", "window_start",
        "window_end", "observation_id", "observation_ordinal",
        "entry_source_row", "entry_timestamp_unix_seconds",
        "feature_layout_identity", "feature_layout_hash",
        "source_phase19b_observations_hash", "source_phase19b_result_hash",
        *PREDICTORS, "outcome__extension_minus_fixed_return",
        "outcome__outcome_class")
    phase19c.write_table(dataset, dataset_header, ())
    dataset_hash = phase19c.fnv1a64(dataset.read_bytes())
    feature_hash = "fnv1a64:feature"
    source_result_hash = "fnv1a64:source-result"
    result_canonical = (
        "phase19c_causal_dataset_result_v1;dataset_hash=" + dataset_hash +
        ";feature_layout_hash=" + feature_hash +
        ";source_phase19b_result_hash=" + source_result_hash + ";")
    metadata = {
        "analysis_plan_identity": phase19c.PLAN, "model_id": "1729",
        "experiment_id": "NULL", "symbol": "USDCAD",
        "cohort_role": "primary", "prediction_horizon": "6",
        "window_start": "2025-01-01", "window_end": "2026-01-01",
        "activated_count": "0", "saved_count": "0", "harmed_count": "0",
        "unchanged_count": "0", "joined_count": "0",
        "join_failure_count": "0", "usable_predictor_count": "5",
        "model_input_width": "1", "semantic_layout_version": "1",
        "feature_layout_identity": "fixture", "feature_layout_hash": feature_hash,
        "feature_ablation_identity": "", "source_phase19b_observations_hash":
        "fnv1a64:source-observations", "source_phase19b_result_hash":
        source_result_hash, "dataset_hash": dataset_hash,
        "result_hash": phase19c.fnv1a64(result_canonical.encode()),
        "target_comparison_semantics": "exact_binary64_delta_sign_v1",
        "primary_target_population": "SAVED_vs_HARMED_only",
        "unchanged_retained_not_classified": "true",
        "predictor_state_semantics":
        "completed_entry_decision_row_and_history_only",
        "post_entry_predictors_included": "false",
        "leakage_audit_passed": "true", "requires_read_only_transaction": "true",
        "production_rows_modified": "false",
    }
    metadata_rows = ((phase19c.DATASET_SCHEMA, "1", key, value)
                     for key, value in metadata.items())
    phase19c.write_table(
        directory / f"{stem}.phase19c.metadata.tsv",
        ("schema_identity", "schema_version", "key", "value"), metadata_rows)
    manifest_rows = [
        (phase19c.DATASET_SCHEMA, "1", "predicted_class",
         "frozen_decision_state", "at_entry_decision", "", "categorical", "true"),
        (phase19c.DATASET_SCHEMA, "1", "direction",
         "frozen_decision_state", "at_entry_decision", "", "categorical", "true"),
        (phase19c.DATASET_SCHEMA, "1", "directional_probability",
         "frozen_decision_state", "at_entry_decision", "", "numeric", "true"),
        (phase19c.DATASET_SCHEMA, "1", "normalized_directional_confidence",
         "frozen_decision_state", "at_entry_decision", "", "numeric", "true"),
        (phase19c.DATASET_SCHEMA, "1", "causal_x",
         "authoritative_model_input_tensor_decision_row",
         "completed_decision_row_at_or_before_entry", "0", "numeric", "true"),
    ]
    phase19c.write_table(
        directory / f"{stem}.phase19c.predictors.tsv",
        ("schema_identity", "schema_version", "predictor_name", "source_category",
         "causal_timing", "model_input_column", "value_type",
         "included_in_full_model"), manifest_rows)


class Phase19CAnalysisTests(unittest.TestCase):
    def test_header_only_zero_activation_dataset_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            write_empty_dataset(Path(directory))
            runs = phase19c.load_runs(Path(directory), allow_incomplete=True)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].model_id, 1729)
            self.assertEqual(runs[0].rows, ())

    def test_undefined_metrics_when_class_absent(self):
        rows = [row("HARMED", 0.5, 0.0), row("HARMED", 0.6, 1.0)]
        metric = phase19c.numeric_metric(rows, "predictor__causal_x")
        self.assertIsNone(metric["auc"])
        self.assertIsNone(metric["ap"])
        self.assertIsNone(metric["effect"])

    def test_logistic_is_fixed_unweighted_and_deterministic(self):
        matrix = [[-2.0], [-1.0], [1.0], [2.0]]
        labels = [0, 0, 1, 1]
        first = phase19c.fit_logistic(matrix, labels, ["x"])
        second = phase19c.fit_logistic(matrix, labels, ["x"])
        self.assertEqual(first, second)
        self.assertEqual(phase19c.L2_STRENGTH, 1.0)
        self.assertEqual(phase19c.LOGISTIC_MAX_ITERATIONS, 100)

    def test_primary_temporal_protocol_and_diagnostic_exclusion(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = phase19c.analyze(frozen_runs(1000.0), Path(first_dir))
            second = phase19c.analyze(frozen_runs(-1000.0), Path(second_dir))
            self.assertEqual(first["train_n"], 12)
            self.assertEqual(first["oot_n"], 12)
            self.assertEqual(first["full_oot"], second["full_oot"])
            metadata = dict(csv.reader(
                (Path(first_dir) / "analysis.metadata.tsv").read_text().splitlines()[1:],
                delimiter="\t"))
            self.assertEqual(metadata["train_window"], "2025-01-01/2026-01-01")
            self.assertEqual(metadata["oot_window"], "2026-01-01/2026-09-01")
            self.assertEqual(metadata["diagnostic_model_excluded_from_primary_fit"], "true")
            self.assertEqual(metadata["hyperparameter_tuning"], "false")

    def test_output_hashes_and_tsv_shapes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            phase19c.analyze(frozen_runs(), Path(first_dir))
            phase19c.analyze(frozen_runs(), Path(second_dir))
            for name in ("counts.tsv", "univariate.tsv", "temporal_stability.tsv",
                         "multivariate_metrics.tsv", "coefficients.tsv",
                         "analysis.metadata.tsv"):
                first = (Path(first_dir) / name).read_bytes()
                second = (Path(second_dir) / name).read_bytes()
                self.assertEqual(first, second)
                parsed = list(csv.reader(first.decode().splitlines(), delimiter="\t"))
                self.assertTrue(parsed)
                self.assertTrue(all(len(item) == len(parsed[0]) for item in parsed))

    def test_forbidden_predictor_registry(self):
        self.assertIn("terminal", phase19c.FORBIDDEN_PREDICTOR_TOKENS)
        self.assertIn("recovery", phase19c.FORBIDDEN_PREDICTOR_TOKENS)


if __name__ == "__main__":
    unittest.main()
