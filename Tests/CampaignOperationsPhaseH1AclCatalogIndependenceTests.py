#!/usr/bin/env python3
"""Permanent independent ACL/default expected-versus-observed regressions."""
from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from CampaignOperationsH1AclCatalog import (  # noqa: E402
    AclCatalogError, EXPECTED_FIELDS, OBSERVED_FIELDS, reconcile_acl_catalog,
)
from CampaignOperationsH1ArtifactSnapshot import capture_regular_file  # noqa: E402


class AclCatalogIndependenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ea-h1-acl-catalog-")
        self.root = Path(self.temp.name); self.expected = self.root / "expected.tsv"; self.observed = self.root / "observed.tsv"
        self.requirement = "H1-ACL-INDEPENDENT"
        tuples = [
            ("object_acl", "table", "public.requests", "owner", "reader", "SELECT", "false", "explicit", "", "r"),
            ("column_acl", "column", "public.requests.state", "owner", "writer", "UPDATE", "false", "explicit", "", "r"),
            ("schema_acl", "schema", "public", "db_owner", "PUBLIC", "USAGE", "false", "explicit", "", "n"),
            ("default_acl", "default", "owner:<global>:f", "owner", "PUBLIC", "EXECUTE", "false", "null", "<global>", "f"),
            ("default_acl", "default", "owner:public:r", "owner", "reader", "SELECT", "false", "explicit", "public", "r"),
            ("object_acl", "table", "public.null_origin", "owner", "owner", "SELECT", "false", "null", "", "r"),
            ("object_acl", "table", "public.explicit_origin", "owner", "owner", "SELECT", "false", "explicit", "", "r"),
        ]
        self.expected_rows = [dict(zip(EXPECTED_FIELDS, (self.requirement, *row))) for row in tuples]
        self.observed_rows = []
        for row in self.expected_rows:
            observed = {field: row[field] for field in EXPECTED_FIELDS if field != "requirement_id"}
            observed.update({"capture_version": "h1-acl-catalog-capture-v1",
                             "query_id": "pg-catalog-acl-query-v1", "cluster_id": "cluster-1",
                             "run_id": "run-1", "query_execution_id": "QEXEC-1",
                             "raw_acl": "{owner=arwdDxt/owner}" if row["origin"] == "explicit" else "",
                             "default_acl_source": "acldefault(r,owner)" if row["origin"] == "null" else "none"})
            self.observed_rows.append(observed)
        self.write(self.expected, EXPECTED_FIELDS, self.expected_rows)
        self.write(self.observed, OBSERVED_FIELDS, self.observed_rows)

    def tearDown(self): self.temp.cleanup()
    @staticmethod
    def write(path, fields, rows):
        with path.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    @staticmethod
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def reconcile(self, expected_digest=None, observed_digest=None, run="run-1", cluster="cluster-1", attested=None):
        expected_snapshot = capture_regular_file(self.expected, "EXPECTED", run)
        observed_snapshot = capture_regular_file(self.observed, "OBSERVED", run)
        return reconcile_acl_catalog(self.expected, self.observed, self.requirement, run, cluster,
                                     {"QEXEC-1"} if attested is None else attested,
                                     expected_digest or self.digest(self.expected),
                                     observed_digest or self.digest(self.observed),
                                     expected_manifest_bytes=expected_snapshot.data,
                                     observed_capture_bytes=observed_snapshot.data)
    def assert_error(self, message, function, *args):
        with self.assertRaisesRegex(AclCatalogError, message): function(*args)

    def test_explicit_column_schema_global_schema_default_and_origins_reconcile(self):
        self.assertEqual(self.reconcile()["tuple_count"], "7")
    def test_missing_tuple_is_rejected(self):
        self.write(self.observed, OBSERVED_FIELDS, self.observed_rows[:-1]); self.assert_error("missing-catalog-tuple", self.reconcile)
    def test_extra_tuple_is_rejected(self):
        extra = dict(self.observed_rows[0]); extra["grantee"] = "extra"; self.write(self.observed, OBSERVED_FIELDS, self.observed_rows + [extra]); self.assert_error("extra-catalog-tuple", self.reconcile)
    def test_duplicate_tuple_is_rejected(self):
        self.write(self.observed, OBSERVED_FIELDS, self.observed_rows + [self.observed_rows[0]]); self.assert_error("duplicate-observed-tuple", self.reconcile)
    def test_expected_only_change_is_rejected(self):
        self.expected_rows[0]["privilege"] = "UPDATE"; self.write(self.expected, EXPECTED_FIELDS, self.expected_rows); self.assert_error("missing-catalog-tuple", self.reconcile)
    def test_observed_only_change_is_rejected(self):
        self.observed_rows[0]["privilege"] = "UPDATE"; self.write(self.observed, OBSERVED_FIELDS, self.observed_rows); self.assert_error("missing-catalog-tuple", self.reconcile)
    def test_changed_manifest_with_unchanged_catalog_is_rejected(self): self.test_expected_only_change_is_rejected()
    def test_changed_catalog_with_unchanged_manifest_is_rejected(self): self.test_observed_only_change_is_rejected()
    def test_wrong_origin_is_rejected(self):
        self.observed_rows[0]["origin"] = "null"; self.observed_rows[0]["default_acl_source"] = "acldefault(r,owner)"; self.write(self.observed, OBSERVED_FIELDS, self.observed_rows); self.assert_error("missing-catalog-tuple", self.reconcile)
    def test_hand_authored_observed_file_without_attested_query_is_rejected(self): self.assert_error("unattested-catalog-query-execution", self.reconcile, None, None, "run-1", "cluster-1", set())
    def test_stale_catalog_capture_is_rejected(self):
        self.observed_rows[0]["capture_version"] = "old"; self.write(self.observed, OBSERVED_FIELDS, self.observed_rows); self.assert_error("stale-catalog-capture", self.reconcile)
    def test_wrong_cluster_and_run_identity_are_rejected(self):
        self.assert_error("wrong-observed-run-identity", self.reconcile, None, None, "run-2", "cluster-1")
        self.assert_error("wrong-observed-cluster-identity", self.reconcile, None, None, "run-1", "cluster-2")
    def test_stale_expected_and_observed_digests_are_rejected(self):
        self.assert_error("stale-expected-manifest-digest", self.reconcile, "0" * 64)
        self.assert_error("stale-observed-catalog-digest", self.reconcile, None, "0" * 64)


if __name__ == "__main__": unittest.main(verbosity=2)
