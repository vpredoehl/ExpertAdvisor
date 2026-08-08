#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from CampaignOperationsH1Provenance import (  # noqa: E402
    CHAIN_FIELDS, EDGE_FIELDS, NODE_SEQUENCE, ProvenanceError, validate_provenance,
)


class ProvenanceIntegrationTests(unittest.TestCase):
    def chain(self):
        values = {field: field.upper() for field in NODE_SEQUENCE}
        values["obligation_id"] = "OBL-1"
        return {**values, "run_id": "run-1"}

    def edges(self, chain):
        rows = []
        for source_type, target_type in zip(NODE_SEQUENCE, NODE_SEQUENCE[1:]):
            for left, right in ((source_type, target_type), (target_type, source_type)):
                rows.append(dict(zip(EDGE_FIELDS, (
                    left, chain[left], f"{left}_to_{right}", right, chain[right], "run-1"))))
        return rows

    def test_complete_chain_requires_all_forward_and_reverse_edges(self):
        chain = self.chain()
        validate_provenance([chain], self.edges(chain), {"OBL-1"}, "run-1")

    def test_each_missing_boundary_edge_fails(self):
        chain = self.chain(); complete = self.edges(chain)
        for index in range(len(complete)):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ProvenanceError, "missing-provenance-edge"):
                    validate_provenance([chain], complete[:index] + complete[index + 1:],
                                        {"OBL-1"}, "run-1")

    def test_removing_any_provenance_node_fails(self):
        for field in NODE_SEQUENCE:
            with self.subTest(field=field):
                chain = self.chain(); chain[field] = ""
                with self.assertRaisesRegex(ProvenanceError, "incomplete-or-stale-chain"):
                    validate_provenance([chain], self.edges(self.chain()), {"OBL-1"}, "run-1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
