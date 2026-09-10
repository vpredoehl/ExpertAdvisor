#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Scripts" / "aggregate_phase19_state_analysis.py"
SPEC = importlib.util.spec_from_file_location("phase19_aggregate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
phase19 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase19)


def result(model_id: int, window: str, count: int, delta: float) -> dict[str, object]:
    scope = "volatility_regime:quartile_3"
    return {
        "path": pathlib.Path(f"model{model_id}-{window}.txt"),
        "header": {},
        "model_id": model_id,
        "window": window,
        "role": "primary",
        "accepted": True,
        "strata": {scope: {"activated_count": str(count)}},
        "deltas": {scope: {"aggregate_return_delta": str(delta)}},
        "populations": {},
    }


def write_minimal_artifact(path: pathlib.Path, model_id: int, window: str, role: str) -> None:
    path.write_text(
        "PHASE19_ANALYSIS,"
        f"model_id={model_id},"
        f"window_label={window},"
        f"cohort_role={role},"
        "model_acceptance_qualified=true\n",
        encoding="utf-8",
    )


class Phase19CrossWindowConsensusTests(unittest.TestCase):
    def test_consensus_reversal_is_not_masked_by_positive_aggregate(self) -> None:
        rows = phase19.aggregate([
            result(1745, phase19.WINDOWS[0], 100, +0.0020),
            result(1743, phase19.WINDOWS[0], 100, +0.0020),
            result(1729, phase19.WINDOWS[0], 10, -0.0001),
            result(1745, phase19.WINDOWS[1], 100, -0.0010),
            result(1743, phase19.WINDOWS[1], 100, -0.0010),
            result(1729, phase19.WINDOWS[1], 10, +0.0030),
        ])
        by_window = {str(row["window"]): row for row in rows}
        d = by_window[phase19.WINDOWS[0]]
        t = by_window[phase19.WINDOWS[1]]

        self.assertGreater(float(d["combined_aggregate_delta"]), 0.0)
        self.assertGreater(float(t["combined_aggregate_delta"]), 0.0)
        self.assertTrue(d["aggregate_sign_agrees_between_windows"])
        self.assertTrue(t["aggregate_sign_agrees_between_windows"])
        self.assertEqual(d["nontrivial_consensus_sign"], +1)
        self.assertEqual(t["nontrivial_consensus_sign"], -1)
        self.assertFalse(d["nontrivial_consensus_agrees_between_windows"])
        self.assertFalse(t["nontrivial_consensus_agrees_between_windows"])

    def test_read_artifacts_accepts_native_txt_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            for model_id in sorted(phase19.PRIMARY_MODELS | {phase19.DIAGNOSTIC_MODEL}):
                role = "diagnostic" if model_id == phase19.DIAGNOSTIC_MODEL else "primary"
                for window in phase19.WINDOWS:
                    write_minimal_artifact(
                        directory / f"model{model_id}-{window}.txt",
                        model_id,
                        window,
                        role,
                    )
            self.assertEqual(len(phase19.read_artifacts(directory)), 14)

    def test_duplicate_txt_and_csv_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            for model_id in sorted(phase19.PRIMARY_MODELS | {phase19.DIAGNOSTIC_MODEL}):
                role = "diagnostic" if model_id == phase19.DIAGNOSTIC_MODEL else "primary"
                for window in phase19.WINDOWS:
                    write_minimal_artifact(
                        directory / f"model{model_id}-{window}.txt",
                        model_id,
                        window,
                        role,
                    )
            write_minimal_artifact(
                directory / "duplicate.csv",
                1745,
                phase19.WINDOWS[0],
                "primary",
            )
            with self.assertRaisesRegex(ValueError, "duplicate Phase 19 artifact"):
                phase19.read_artifacts(directory)


if __name__ == "__main__":
    unittest.main()
