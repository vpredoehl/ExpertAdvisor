#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from CampaignOperationsH1ArtifactSnapshot import capture_regular_file  # noqa: E402
from CampaignOperationsH1RestoreEvidence import EXPECTED, HEADER, validate_snapshot_bytes  # noqa: E402


class SnapshotConsumerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ea-h1-restore-snapshot-")
        self.root = Path(self.temp.name)
        self.run_id = "run-1"
        rows = []
        self.artifact_snapshots = {}
        for scenario, expected in EXPECTED.items():
            relative = f"artifact-{scenario}.log"
            data = f"retained restore scenario {scenario}\n".encode()
            path = self.root / relative; path.write_bytes(data)
            snapshot = capture_regular_file(path, f"ART-RECORD-H1RESTORE{scenario}", self.run_id, relative)
            self.artifact_snapshots[relative] = snapshot
            row = {
                "format_version": "h1-restore-runtime-v2", "run_id": self.run_id,
                "scenario_id": scenario, "source_cluster_state": "disposable",
                "target_cluster_pre_state": "disposable", "role_creation_source": "fixture",
                "dump_type": "custom", "pre_restore_audit": expected[0],
                "restore_attempted": expected[1], "restore_result": expected[2],
                "post_role_audit": expected[3], "post_database_audit": expected[4],
                "expected_sqlstate": expected[5], "expected_diagnostic": expected[6],
                "actual_sqlstate": expected[5], "actual_diagnostic": expected[6],
                "historical_bytes": expected[7], "final_disposition": expected[8],
                "artifact_id": f"ART-RECORD-H1RESTORE{scenario}", "artifact_path": relative,
                "artifact_digest": snapshot.digest, "record_digest": "",
            }
            values = [row[field] for field in HEADER]
            row["record_digest"] = hashlib.sha256("\t".join(values[:-1]).encode()).hexdigest()
            rows.append(row)
        target = io.StringIO(newline="")
        writer = csv.DictWriter(target, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
        self.runtime_path = self.root / "restore.tsv"; self.runtime_path.write_text(target.getvalue())
        self.runtime_snapshot = capture_regular_file(self.runtime_path, "ART-SUPPORT-RESTORE",
                                                     self.run_id, "restore.tsv")

    def tearDown(self):
        self.temp.cleanup()

    def test_restore_validation_uses_only_snapshot_bytes_after_same_content_replacement(self):
        path = self.root / "artifact-A.log"
        retained = path.read_bytes(); path.unlink(); path.write_bytes(retained)
        validate_snapshot_bytes(self.runtime_snapshot.data,
                                {key: value.data for key, value in self.artifact_snapshots.items()},
                                self.run_id)

    def test_changed_snapshot_bundle_is_rejected_even_if_path_still_matches(self):
        bundle = {key: value.data for key, value in self.artifact_snapshots.items()}
        bundle["artifact-A.log"] = b"forged\n"
        with self.assertRaises(SystemExit):
            validate_snapshot_bytes(self.runtime_snapshot.data, bundle, self.run_id)

    def test_graph_report_and_index_carry_snapshot_identity_fields(self):
        source = (ROOT / "Scripts/CampaignOperationsH1EvidenceGraph.py").read_text()
        for field in ("snapshot_id", "artifact_lexical_path", "artifact_device",
                      "artifact_inode", "artifact_size", "snapshot_digest"):
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
