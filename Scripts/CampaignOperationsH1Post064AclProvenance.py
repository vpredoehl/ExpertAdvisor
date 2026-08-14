#!/usr/bin/env python3
"""Produce the deterministic post-064 H1 ACL provenance classification.

The input is the literal H1A006 tuple stream emitted by the frozen 055 ACL
manifest.  This tool deliberately does not query a database and does not read
backup material.  Its accepted later-authority rules are closed and sourced
from the checked-in 056 and 059 contracts; predecessor rules are the surviving
045--054 grants which 055 did not revoke when it transferred ownership.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


FIELDS = [
    "tuple_number", "diff_direction", "requirement_id", "object_kind",
    "database_name", "schema_name", "object_identity", "owner_identity",
    "grantee_identity", "privilege", "grantable", "hierarchy_inheritance_flag",
    "object_acl_type_code", "classification", "first_authorizing_migration_or_contract",
    "055_effect", "later_reauthorizing_migration_or_contract",
    "checked_in_evidence", "rationale", "post_h1_composition_disposition",
]
AUTHORITY_FIELDS = [
    "authority_version", "tuple_number", "diff_direction", "requirement_id", "object_kind",
    "database_name", "schema_name", "object_identity", "owner_identity", "grantee_identity",
    "privilege", "grantable", "hierarchy_inheritance_flag", "object_acl_type_code",
    "classification", "first_authorizing_migration_or_contract",
    "later_reauthorizing_migration_or_contract", "checked_in_evidence",
]
TUPLE_KEY_FIELDS = [
    "diff_direction", "requirement_id", "object_kind", "database_name", "schema_name",
    "object_identity", "owner_identity", "grantee_identity", "privilege", "grantable",
    "hierarchy_inheritance_flag", "object_acl_type_code",
]

H2_PHASE_E_TABLES = {
    "public.campaign_operations_operational_request",
    "public.campaign_operations_dispatch_attempt",
    "public.campaign_operations_dispatch_audit_reference_event",
}
H2_PHASE_E_LOCKS = {
    "public.lock_campaign_operations_authorization_head(bigint,text)",
    "public.lock_campaign_operations_budget_head(bigint)",
    "public.lock_campaign_operations_campaign(bigint)",
    "public.lock_campaign_operations_reservation(bigint)",
    "public.lock_campaign_operations_request(bigint)",
}
H2_PHASE_E_AUDIT_SEQUENCE = (
    "public.campaign_operations_dispatch__dispatch_audit_reference_even_seq"
)
AUTHORIZED_V3 = (
    "public.campaign_operations_production_dispatch_authorized_v3("
    "bigint,integer,text,timestamp with time zone,text,text,text)"
)
READINESS_GATE = "public.campaign_operations_production_dispatch_readiness_gate_v1(text)"


def fail(message: str) -> None:
    raise SystemExit(f"provenance-classification: {message}")


def load_manifest_keys(path: Path) -> set[tuple[str, str, str, str]]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        required = {"object_class", "object_identity", "grantee", "privilege"}
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            fail(f"invalid-manifest-header:{path}")
        return {
            (row["object_class"], row["object_identity"], row["grantee"], row["privilege"])
            for row in rows
        }


def parse_tuples(path: Path) -> list[list[str]]:
    tuples: list[list[str]] = []
    for line in path.read_text().splitlines():
        if not line.startswith("H1A006|explicit_acl|"):
            continue
        fields = line.split("|")
        if len(fields) not in (13, 14):
            fail(f"invalid-tuple-field-count:{len(fields)}:{line}")
        tuples.append(fields)
    if len(tuples) != 136:
        fail(f"tuple-count:{len(tuples)}")
    if len({tuple(item) for item in tuples}) != len(tuples):
        fail("duplicate-tuple")
    return sorted(tuples, key=lambda row: (row[2], row[4], row[7], row[9], row[10]))


def predecessor_evidence(grantee: str) -> tuple[str, str]:
    evidence = {
        "campaign_operations_campaign_creator": (
            "045", "Database/migrations/045_campaign_operations_foundation.sql:602-708"),
        "campaign_operations_authorizer": (
            "045", "Database/migrations/045_campaign_operations_foundation.sql:602-708"),
        "campaign_operations_owner": (
            "045", "Database/migrations/045_campaign_operations_foundation.sql:602-708"),
        "campaign_operations_budget_administrator": (
            "047", "Database/migrations/047_campaign_operations_budget_request_acceptance.sql:1489-1629"),
        "campaign_operations_request_acceptor": (
            "047", "Database/migrations/047_campaign_operations_budget_request_acceptance.sql:1489-1629"),
        "campaign_operations_dispatcher": (
            "048", "Database/migrations/048_campaign_operations_durable_dispatch_handoff.sql:1107-1299,1345-1346"),
        "campaign_operations_phase5_transactional": (
            "048", "Database/migrations/048_campaign_operations_durable_dispatch_handoff.sql:1107-1299,1345-1346"),
    }
    try:
        return evidence[grantee]
    except KeyError:
        fail(f"unmapped-predecessor-grantee:{grantee}")


def h2_phase_e(row: list[str]) -> bool:
    _, _, direction, _, kind, _, _, identity, _, grantee, privilege, *_ = row
    if direction != "actual_minus_expected" or grantee != "campaign_operations_production_phase5_transactional":
        return False
    if kind == "table" and identity in H2_PHASE_E_TABLES and privilege == "SELECT":
        return True
    if kind == "column" and identity.startswith("public.campaign_operations_dispatch_audit_reference_event.") and privilege == "INSERT":
        return True
    if kind == "function" and identity in H2_PHASE_E_LOCKS and privilege == "EXECUTE":
        return True
    return kind == "sequence" and identity == H2_PHASE_E_AUDIT_SEQUENCE and privilege == "USAGE"


def classification(row: list[str], h2_manifest: set[tuple[str, str, str, str]],
                   h8_manifest: set[tuple[str, str, str, str]]) -> tuple[str, str, str, str, str, str]:
    _, _, direction, _, kind, _, _, identity, _, grantee, privilege, *_ = row
    key = (kind, identity, grantee, privilege)
    if direction == "expected_minus_actual":
        return (
            "C", "055", "055 explicitly requires this frozen owner UPDATE tuple; it is absent.", "NONE",
            "Database/migrations/055_campaign_operations_production_admission_foundation.sql:4269-4273; Database/manifests/055_campaign_operations_h1_column_acl.tsv:4",
            "No checked-in 056-064 revocation permits removal of this required 055 tuple.",
        )
    if grantee == "pqxx":
        return (
            "C", "NONE", "055 freezes a non-pqxx H1 ACL surface; no checked-in 045-064 grant authorizes this tuple.", "NONE",
            "Database/migrations/055_campaign_operations_production_admission_foundation.sql:4204-4213,4228-4251,4564-4573,4619-4621",
            "Current pqxx ACL is production drift, not authority; it remains fail-closed.",
        )
    if key in h2_manifest:
        return (
            "B1", "056", "Not in the frozen 055 external exact manifest; 056 adds this exact H2 contract.", "056",
            "Database/migrations/056_campaign_operations_h2_privilege_deployment_contract.sql:117-217; Database/manifests/056_campaign_operations_h2_explicit_acl.tsv",
            "Exact later H2 manifest authority; accepted only as the checked-in tuple.",
        )
    if h2_phase_e(row):
        return (
            "B1", "056", "Not in the frozen 055 external exact manifest; 056 carries forward this exact Phase-E surface.", "056",
            "Database/migrations/056_campaign_operations_h2_privilege_deployment_contract.sql:219-332",
            "Exact later H2 Phase-E authority; accepted only as the enumerated tuple.",
        )
    if key in h8_manifest or (kind == "function" and identity == READINESS_GATE and
                              grantee == "campaign_operations_h1_boundary_authority" and privilege == "EXECUTE"):
        return (
            "B1", "059", "Not in the frozen 055 external exact manifest; 059 creates/seals the direct-SQL boundary tuple.", "059",
            "Database/migrations/059_campaign_operations_direct_sql_readiness_boundary.sql:194-252; Database/manifests/059_campaign_operations_direct_sql_boundary_explicit_acl.tsv",
            "Exact later direct-SQL boundary authority; accepted only as the enumerated tuple.",
        )
    version, evidence = predecessor_evidence(grantee)
    return (
        "A", version,
        "055 transfers protected-relation ownership at 055:4309-4314 but does not revoke this checked-in predecessor grant.",
        "NONE", evidence,
        "Checked-in 045-054 predecessor ACL survives the ownership transfer and is omitted only by the frozen external H1 exact-state comparison.",
    )


def generate(input_path: Path, output_path: Path, root: Path) -> dict[str, int]:
    h2_manifest = load_manifest_keys(root / "Database/manifests/056_campaign_operations_h2_explicit_acl.tsv")
    h8_manifest = load_manifest_keys(root / "Database/manifests/059_campaign_operations_direct_sql_boundary_explicit_acl.tsv")
    output: list[dict[str, str]] = []
    for number, item in enumerate(parse_tuples(input_path), start=1):
        code, first, effect, later, evidence, rationale = classification(item, h2_manifest, h8_manifest)
        _, _, direction, requirement, kind, database, schema, identity, owner, grantee, privilege, grantable, hierarchy, *acl_type = item
        output.append({
            "tuple_number": str(number), "diff_direction": direction, "requirement_id": requirement,
            "object_kind": kind, "database_name": database, "schema_name": schema,
            "object_identity": identity, "owner_identity": owner, "grantee_identity": grantee,
            "privilege": privilege, "grantable": grantable,
            "hierarchy_inheritance_flag": hierarchy, "object_acl_type_code": acl_type[0] if acl_type else "",
            "classification": code, "first_authorizing_migration_or_contract": first,
            "055_effect": effect, "later_reauthorizing_migration_or_contract": later,
            "checked_in_evidence": evidence, "rationale": rationale,
            "post_h1_composition_disposition": "ACCEPT" if code in {"A", "B1"} else "REJECT",
        })
    counts = {key: sum(row["classification"] == key for row in output) for key in ("A", "B1", "B2", "C")}
    if counts != {"A": 83, "B1": 35, "B2": 0, "C": 18}:
        fail(f"unexpected-counts:{counts}")
    accepted = sum(row["post_h1_composition_disposition"] == "ACCEPT" for row in output)
    if accepted != 118:
        fail(f"unexpected-accept-count:{accepted}")
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    return {**counts, "ACCEPT": accepted, "REJECT": len(output) - accepted}


def read_classification(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        if rows.fieldnames != FIELDS:
            fail(f"invalid-classification-header:{path}")
        result = list(rows)
    if len(result) != 136 or len({row["tuple_number"] for row in result}) != 136:
        fail("invalid-classification-cardinality")
    counts = {key: sum(row["classification"] == key for row in result) for key in ("A", "B1", "B2", "C")}
    if counts != {"A": 83, "B1": 35, "B2": 0, "C": 18}:
        fail(f"invalid-classification-counts:{counts}")
    if any((row["classification"] in {"A", "B1"}) !=
           (row["post_h1_composition_disposition"] == "ACCEPT") for row in result):
        fail("invalid-classification-disposition")
    return result


def emit_authority(classification_path: Path, authority_path: Path) -> None:
    rows = [row for row in read_classification(classification_path)
            if row["post_h1_composition_disposition"] == "ACCEPT"]
    if len(rows) != 118 or any(row["diff_direction"] != "actual_minus_expected" for row in rows):
        fail("invalid-accepted-composition-set")
    with authority_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUTHORITY_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            authority_row = {"authority_version": "post064-h1-acl-composition-v1", **{
                field: row[field] for field in AUTHORITY_FIELDS if field != "authority_version"
            }}
            # A database name identifies an audit run, not a PostgreSQL ACL.
            # The authority set must remain usable by deterministic disposable
            # clusters while preserving every ACL-bearing identity field.
            authority_row["database_name"] = "*"
            writer.writerow(authority_row)


def load_authority(path: Path) -> dict[tuple[str, ...], str]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        if rows.fieldnames != AUTHORITY_FIELDS:
            fail(f"invalid-authority-header:{path}")
        data = list(rows)
    if len(data) != 118:
        fail(f"invalid-authority-count:{len(data)}")
    keys: dict[tuple[str, ...], str] = {}
    for row in data:
        if row["authority_version"] != "post064-h1-acl-composition-v1" or \
           row["classification"] not in {"A", "B1"} or \
           row["diff_direction"] != "actual_minus_expected" or row["database_name"] != "*":
            fail("invalid-authority-row")
        key = tuple(row[field] for field in TUPLE_KEY_FIELDS)
        keys[key] = row["classification"]
    if len(keys) != len(data):
        fail("duplicate-authority-tuple")
    return keys


def filter_findings(authority_path: Path, allow_b1: bool) -> None:
    accepted = load_authority(authority_path)
    for line in sys.stdin:
        fields = line.rstrip("\n").split("|")
        if len(fields) in (13, 14) and fields[:2] == ["H1A006", "explicit_acl"]:
            raw_key_fields = fields[2:] + ([""] if len(fields) == 13 else [])
            raw_key_fields[3] = "*"
            raw_key = tuple(raw_key_fields)
            # Authority keys include all tuple fields after the fixed H1A006
            # prefix, in exactly the emitted order.  No object-name, grantee,
            # or privilege pattern is accepted as a substitute.
            if accepted.get(raw_key) == "A" or (allow_b1 and accepted.get(raw_key) == "B1"):
                continue
        sys.stdout.write(line)


def main(argv: list[str]) -> None:
    if len(argv) == 5 and argv[1] == "generate":
        counts = generate(Path(argv[2]), Path(argv[3]), Path(argv[4]))
        print("H1_POST064_PROVENANCE_OK " + " ".join(f"{key}={value}" for key, value in counts.items()))
        return
    if len(argv) == 4 and argv[1] == "emit-authority":
        emit_authority(Path(argv[2]), Path(argv[3]))
        print("H1_POST064_COMPOSITION_AUTHORITY_OK tuples=118")
        return
    if len(argv) in (3, 4) and argv[1] == "filter-findings" and \
       (len(argv) == 3 or argv[3] == "--allow-b1"):
        filter_findings(Path(argv[2]), len(argv) == 4)
        return
    fail("usage: generate INPUT_AUDIT_LOG OUTPUT_TSV REPOSITORY_ROOT | emit-authority CLASSIFICATION_TSV AUTHORITY_TSV | filter-findings AUTHORITY_TSV [--allow-b1]")


if __name__ == "__main__":
    main(sys.argv)
