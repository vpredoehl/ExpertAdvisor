#!/usr/bin/env python3
"""Generate the frozen H1 clause inventory from governing documents only.

This intentionally has no imports from, and never reads, requirement, fixture,
test, evidence, validator, graph, report, or artifact registries.  The reviewed
line selections below are clause-sized excerpts, not whole-section authority.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Tests/fixtures/CampaignOperationsH1NormativeClauses.tsv"
VERSION = "h1-normative-authority-v2"

# id, document, stable section, first line, last line, normalized requirement,
# architectural requirement identity, evidence class, executable policy,
# status policy, cardinality policy, precedence rank, H1 scope policy.
CLAUSES = [
    ("ADR19-AUTHORITY", "docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md", "ADR-0019 §2", 39, 44, "Production dispatch is default-off and authorized only by the immutable alternating enable chain.", "NREQ-H1-PRODUCTION-AUTHORITY", "runtime", "executable", "RECONCILED", "aggregate", 1, "in_h1"),
    ("ADR19-MANAGER-BOUNDARY", "docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md", "ADR-0019 §2.2", 66, 76, "The Campaign Manager reuses one Phase E engine and acquires no scheduler or worker authority.", "NREQ-H1-MANAGER-BOUNDARY", "runtime", "executable", "RECONCILED", "split", 1, "in_h1"),
    ("ADR19-SCHEDULER-BOUNDARY", "docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md", "ADR-0019 §2.3", 80, 92, "H1 consumes pinned scheduler evidence through a narrow read interface and never inherits future-generation approval.", "NREQ-H1-SCHEDULER-BOUNDARY", "runtime", "executable", "RECONCILED", "split", 1, "in_h1"),
    ("ADR19-SCOPE", "docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md", "ADR-0019 §2.4", 94, 111, "H1 excludes scheduler, worker, scientific, lifecycle, archival, cancellation, and continuous-manager authority.", "NREQ-H1-SCOPE", "runtime", "executable", "RECONCILED", "split", 1, "in_h1"),
    ("ADR19-COMPATIBILITY", "docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md", "ADR-0019 §5", 160, 172, "Migration 055 is additive, preserves V1 identities, and grants no production login membership.", "NREQ-H1-COMPATIBILITY", "restore", "executable", "RECONCILED", "split", 1, "in_h1"),
    ("ADR19-H1-BOUNDARY", "docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md", "ADR-0019 §6", 174, 185, "H1 creates authority, persistence, and readiness without production handoff.", "NREQ-H1-PHASE-BOUNDARY", "runtime", "executable", "RECONCILED", "split", 1, "in_h1"),
    ("ADR19-VERIFICATION", "docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md", "ADR-0019 §7", 187, 199, "Pre-enablement readiness and rollout require exact migration, scheduler, build, role, historical, and blocker evidence.", "NREQ-H1-VERIFICATION", "pipeline", "executable", "RECONCILED", "split", 1, "in_h1"),

    ("ADR19A-IDENTIFIER", "docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md", "ADR-0019A §3", 48, 62, "The acquisition function has the exact 61-byte catalog identifier and no truncation collision.", "NREQ-H1-ACQUISITION-IDENTIFIER", "runtime", "executable", "RECONCILED", "split", 2, "in_h1"),
    ("ADR19A-SEALED-OWNER", "docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md", "ADR-0019A §4.1", 68, 83, "Every protected H1 authority object is owned by the unreachable sealed boundary role.", "NREQ-H1-SEALED-OWNER", "role_security", "executable", "RECONCILED", "split", 2, "in_h1"),
    ("ADR19A-DEFINERS", "docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md", "ADR-0019A §8", 184, 195, "Every H1 security-definer boundary has sealed ownership, pinned resolution, static SQL, and exact ACLs.", "NREQ-H1-DEFINERS", "acl_catalog", "executable", "RECONCILED", "split", 2, "in_h1"),
    ("ADR19A-REPLAY", "docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md", "ADR-0019A §9", 199, 224, "Enable, disable, and acquisition prove complete immutable replay evidence before mutable state.", "NREQ-H1-REPLAY", "runtime", "executable", "RECONCILED", "split", 2, "in_h1"),
    ("ADR19A-LOCK-ORDER", "docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md", "ADR-0019A §10", 228, 255, "Acquisition proves the complete 0a through 5 lock order with independent PostgreSQL blocker evidence.", "NREQ-H1-LOCK-ORDER", "lock", "executable", "RECONCILED", "split", 2, "in_h1"),
    ("ADR19A-MIGRATION", "docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md", "ADR-0019A §11", 259, 267, "Migration 055 remains transactional and idempotent with regenerated checksum evidence.", "NREQ-H1-MIGRATION", "checksum", "executable", "RECONCILED", "split", 2, "in_h1"),
    ("ADR19A-HISTORY", "docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md", "ADR-0019A §14", 289, 294, "A genuine schema-054 fixture proves Attempt V1 and Completion V1 byte preservation.", "NREQ-H1-HISTORICAL-BYTES", "restore", "executable", "RECONCILED", "split", 2, "in_h1"),

    ("ADR19B-FAILURE-CONTRACT", "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md", "ADR-0019B §2", 23, 44, "Deployment failures use exact SQLSTATE and stable H1A diagnostics at the intended branch.", "NREQ-H1-FAILURE-CONTRACT", "runtime", "executable", "RECONCILED", "split", 3, "in_h1"),
    ("ADR19B-ROLE-IDENTITY", "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md", "ADR-0019B §3", 48, 79, "The sealed role has the complete exact frozen catalog identity and is never normalized.", "NREQ-H1-ROLE-IDENTITY", "role_security", "executable", "RECONCILED", "split", 3, "in_h1"),
    ("ADR19B-ROLE-GRAPH", "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md", "ADR-0019B §4", 88, 104, "The recursive role graph is empty in both directions and exposes no inherited or SET ROLE path.", "NREQ-H1-ROLE-GRAPH", "role_security", "executable", "RECONCILED", "split", 3, "in_h1"),
    ("ADR19B-OWNERSHIP", "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md", "ADR-0019B §5.1", 108, 131, "Only the exact dependency-bound H1 relation and coupled-object manifest may be boundary-owned.", "NREQ-H1-OWNERSHIP", "role_security", "executable", "RECONCILED", "split", 3, "in_h1"),
    ("ADR19B-ALL-SCHEMA", "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md", "ADR-0019B §6", 159, 184, "All non-system schemas and relevant object classes are scanned for unauthorized H1 authority entry points.", "NREQ-H1-ALL-SCHEMA", "role_security", "executable", "RECONCILED", "split", 3, "in_h1"),
    ("ADR19B-ACL", "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md", "ADR-0019B §7", 188, 208, "Current and default ACLs are expanded from PostgreSQL catalogs and compared exactly, including NULL origin.", "NREQ-H1-ACL", "acl_catalog", "executable", "RECONCILED", "split", 3, "in_h1"),
    ("ADR19B-DEPLOYMENT-AUDIT", "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md", "ADR-0019B §8", 212, 241, "The versioned deployment audit is read-only, stage-bound, and fail-closed.", "NREQ-H1-DEPLOYMENT-AUDIT", "runtime", "executable", "RECONCILED", "split", 3, "in_h1"),
    ("ADR19B-RESTORE", "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md", "ADR-0019B §9", 245, 261, "Restore scenarios A through J retain role, catalog, ACL, and historical evidence independently.", "NREQ-H1-RESTORE", "restore", "executable", "RECONCILED", "split", 3, "in_h1"),
    ("ADR19B-LOCK-EVIDENCE", "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md", "ADR-0019B §11", 274, 293, "Lock acceptance retains backend identities, pg_locks, pg_blocking_pids, outcomes, and reverse-wait absence.", "NREQ-H1-LOCK-EVIDENCE", "lock", "executable", "RECONCILED", "split", 3, "in_h1"),
    ("ADR19B-NEGATIVE-AUTHENTICITY", "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md", "ADR-0019B §12", 297, 304, "Negative tests retain exact prerequisites, statements, expected and observed failures, and results.", "NREQ-H1-NEGATIVE-AUTHENTICITY", "mutation", "executable", "RECONCILED", "split", 3, "in_h1"),
    ("ADR19B-HISTORICAL", "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md", "ADR-0019B §13", 308, 314, "Historical Attempt V1 and Completion V1 bytes are captured before 055 and compared after upgrade and restore.", "NREQ-H1-RESTORE-HISTORY", "restore", "executable", "RECONCILED", "split", 3, "in_h1"),
    ("ADR19B-INERTNESS", "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md", "ADR-0019B §15", 327, 331, "H1 remains default-off and destructive verification remains confined to disposable local clusters.", "NREQ-H1-INERTNESS", "runtime", "executable", "RECONCILED", "split", 3, "in_h1"),

    ("PHASEH-LOCK-ORDER", "docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md", "Corrected Phase H §13", 508, 537, "The corrected architecture freezes lock ordering and prohibits external work while locks are held.", "NREQ-H1-CORRECTED-LOCK-ORDER", "lock", "executable", "RECONCILED", "aggregate", 4, "incorporated_h1"),
    ("PHASEH-PRIVILEGES", "docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md", "Corrected Phase H §14", 586, 607, "The corrected architecture freezes exact role identity, literal ownership, all-schema audit, deployment audit, and restore evidence.", "NREQ-H1-CORRECTED-PRIVILEGES", "acl_catalog", "executable", "RECONCILED", "aggregate", 4, "incorporated_h1"),
    ("PHASEH-MIGRATION", "docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md", "Corrected Phase H §17", 726, 750, "Migration 055 remains additive, inert, sealed, exact-replay-safe, and catalog verified.", "NREQ-H1-CORRECTED-MIGRATION", "runtime", "executable", "RECONCILED", "aggregate", 4, "incorporated_h1"),
    ("PHASEH-INCREMENTS", "docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md", "Corrected Phase H §18", 754, 781, "H1 contains only authority and persistence; H2, H3, and H4 remain separate increments.", "NREQ-H1-CORRECTED-BOUNDARY", "runtime", "executable", "RECONCILED", "aggregate", 4, "incorporated_h1"),
    ("PHASEH-VERIFICATION", "docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md", "Corrected Phase H §19", 800, 822, "Verification uses disposable state, independent lock evidence, and safe-window process suites.", "NREQ-H1-CORRECTED-VERIFICATION", "pipeline", "executable", "RECONCILED", "aggregate", 4, "incorporated_h1"),

    ("VOLUMEXII-MIGRATION", "docs/architecture/Volume_XII_Database.md", "Volume XII §2.3", 97, 107, "Volume XII incorporates migration 055 as additive authority and persistence with no production-state mutation.", "NREQ-H1-VOLUME-MIGRATION", "runtime", "executable", "RECONCILED", "aggregate", 5, "incorporated_h1"),
    ("VOLUMEXII-SEALED", "docs/architecture/Volume_XII_Database.md", "Volume XII §2.3", 128, 145, "Volume XII incorporates sealed ownership, unreachable roles, context lifecycle, and NULL/default ACL semantics.", "NREQ-H1-VOLUME-SEALED", "acl_catalog", "executable", "RECONCILED", "aggregate", 5, "incorporated_h1"),
    ("VOLUMEXII-AUDIT", "docs/architecture/Volume_XII_Database.md", "Volume XII §2.3", 147, 156, "Volume XII incorporates automatic fail-closed all-schema deployment audits.", "NREQ-H1-VOLUME-AUDIT", "role_security", "executable", "RECONCILED", "aggregate", 5, "incorporated_h1"),
    ("VOLUMEXII-TESTING", "docs/architecture/Volume_XII_Database.md", "Volume XII §9.4", 355, 366, "Volume XII requires historical bytes, exact ACLs, replay, lock, owner-threat, and default-ACL tests.", "NREQ-H1-VOLUME-TESTING", "pipeline", "executable", "RECONCILED", "aggregate", 5, "incorporated_h1"),
    ("VOLUMEXII-PERMISSIONS", "docs/architecture/Volume_XII_Database.md", "Volume XII §10.2", 388, 397, "Volume XII incorporates the exact H1 roles, sealed ownership, and scheduler evidence privilege boundary.", "NREQ-H1-VOLUME-PERMISSIONS", "role_security", "executable", "RECONCILED", "aggregate", 5, "incorporated_h1"),
]


def main() -> None:
    fields = ["authority_version", "normative_clause_id", "source_document", "exact_stable_section",
              "exact_source_anchor", "canonical_excerpt", "canonical_clause_digest",
              "normalized_normative_requirement", "derived_h1_requirement_identity", "evidence_class",
              "executable_policy", "status_policy", "cardinality_policy", "precedence_rank", "scope_policy"]
    rows = []
    for identifier, document, section, first, last, normalized, requirement, evidence, executable, status, cardinality, rank, scope in CLAUSES:
        lines = (ROOT / document).read_text().splitlines(keepends=True)
        if first > last or last > len(lines):
            raise SystemExit(f"invalid reviewed anchor {identifier}: lines {first}-{last}")
        raw = "".join(lines[first - 1:last])
        excerpt = re.sub(r"\s+", " ", raw).strip()
        rows.append({
            "authority_version": VERSION, "normative_clause_id": identifier,
            "source_document": document, "exact_stable_section": section,
            "exact_source_anchor": f"lines {first}-{last}", "canonical_excerpt": excerpt,
            "canonical_clause_digest": hashlib.sha256(raw.encode()).hexdigest(),
            "normalized_normative_requirement": normalized,
            "derived_h1_requirement_identity": requirement, "evidence_class": evidence,
            "executable_policy": executable, "status_policy": status,
            "cardinality_policy": cardinality, "precedence_rank": str(rank), "scope_policy": scope,
        })
    with OUTPUT.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"generated {len(rows)} reviewed normative clauses from five governing sources")


if __name__ == "__main__":
    main()
