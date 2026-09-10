#!/usr/bin/env python3

import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase19_aggregate", ROOT / "Scripts" / "aggregate_phase19_state_analysis.py"
)
assert SPEC and SPEC.loader
AGGREGATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATE)


class Phase19CrossModelAggregationTests(unittest.TestCase):
    def write_grid(self, directory: pathlib.Path) -> None:
        for window_index, window in enumerate(AGGREGATE.WINDOWS):
            for model_id in sorted(AGGREGATE.PRIMARY_MODELS | {AGGREGATE.DIAGNOSTIC_MODEL}):
                role = "diagnostic" if model_id == AGGREGATE.DIAGNOSTIC_MODEL else "primary"
                accepted = "false" if model_id == 1662 else "true"
                delta = (1 if model_id % 2 else -1) * (window_index + 1) * 0.01
                content = (
                    "PHASE19_ANALYSIS,model_id={model},window_label={window},"
                    "cohort_role={role},model_acceptance_qualified={accepted}\n"
                    "PHASE19_STATE_STRATUM,scope=volatility_regime:quartile_1,"
                    "activated_count=40\n"
                    "PHASE19_STATE_STRATUM_DELTA,scope=volatility_regime:quartile_1,"
                    "aggregate_return_delta={delta}\n"
                ).format(model=model_id, window=window, role=role,
                         accepted=accepted, delta=delta)
                (directory / f"{model_id}-{window}.csv").write_text(content)

    def test_complete_grid_and_primary_only_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            self.write_grid(directory)
            artifacts = AGGREGATE.read_artifacts(directory)
            rows = AGGREGATE.aggregate(artifacts)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["models_with_support"], 6)
                self.assertEqual(row["total_activated_count"], 240)
                self.assertEqual(row["accepted_models_with_support"], 5)
                self.assertEqual(row["nontrivial_support_models"], 6)
                self.assertFalse(row["sign_agrees_across_nontrivial_models"])
            self.assertTrue(rows[0]["aggregate_sign_agrees_between_windows"])
            self.assertTrue(rows[1]["aggregate_sign_agrees_between_windows"])

    def test_incomplete_grid_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            self.write_grid(directory)
            next(directory.glob("*.csv")).unlink()
            with self.assertRaisesRegex(ValueError, "artifact cohort mismatch"):
                AGGREGATE.read_artifacts(directory)


if __name__ == "__main__":
    unittest.main()
