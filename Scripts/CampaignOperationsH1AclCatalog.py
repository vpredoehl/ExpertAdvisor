#!/usr/bin/env python3
"""Independent expected-manifest and observed-catalog ACL reconciliation."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path


class AclCatalogError(RuntimeError):
    pass


EXPECTED_FIELDS = ["requirement_id", "tuple_kind", "object_class", "object_identity", "owner",
                   "grantee", "privilege", "grant_option", "origin", "default_scope", "object_kind"]
OBSERVED_FIELDS = ["capture_version", "query_id", "cluster_id", "run_id", "query_execution_id",
                   "tuple_kind", "object_class", "object_identity", "owner",
                   "raw_acl", "origin", "default_acl_source", "grantee", "privilege", "grant_option",
                   "default_scope", "object_kind"]
TUPLE_FIELDS = EXPECTED_FIELDS[1:]

OWNER_PRIVILEGES = {
    "r": ("INSERT", "SELECT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER", "MAINTAIN"),
    "S": ("USAGE",), "f": ("EXECUTE",),
    "n": ("CREATE", "USAGE"), "T": ("USAGE",),
}


def _read(path: Path, expected_fields: list[str], data: bytes | None = None) -> list[dict[str, str]]:
    source_object = io.StringIO(data.decode()) if data is not None else path.open(newline="")
    with source_object as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != expected_fields:
            raise AclCatalogError("invalid-acl-catalog-schema")
        return list(reader)


def _tuple(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in TUPLE_FIELDS)


def expected_rows_from_manifest_set(manifest_root: Path,
                                    observed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize expected manifests only at the comparator boundary."""
    def table(name: str) -> list[dict[str, str]]:
        path = manifest_root / name
        with path.open(newline="") as source:
            return list(csv.DictReader(source, delimiter="\t"))

    inventory = table("055_campaign_operations_h1_object_inventory.tsv")
    explicit = table("055_campaign_operations_h1_explicit_acl.tsv")
    defaults = table("055_campaign_operations_h1_default_acl.tsv")
    columns = table("055_campaign_operations_h1_column_acl.tsv")
    result: list[dict[str, str]] = []
    for item in inventory:
        kind = {"schema": "n", "sequence": "S", "function": "f", "type": "T"}.get(
            item["object_class"], "r")
        owner_privileges = (("USAGE",) if kind == "S" and item["expected_acl_origin"] == "explicit"
                            else OWNER_PRIVILEGES[kind])
        for privilege in owner_privileges:
            result.append({"requirement_id": item["requirement_id"], "tuple_kind": "object_acl",
                           "object_class": item["object_class"], "object_identity": item["object_identity"],
                           "owner": item["owner"], "grantee": item["owner"], "privilege": privilege,
                           "grant_option": "false", "origin": item["expected_acl_origin"],
                           "default_scope": "", "object_kind": kind})
        if item["expected_acl_origin"] == "null" and kind in {"f", "T"}:
            for privilege in (("EXECUTE",) if kind == "f" else ("USAGE",)):
                result.append({"requirement_id": item["requirement_id"], "tuple_kind": "object_acl",
                               "object_class": item["object_class"],
                               "object_identity": item["object_identity"], "owner": item["owner"],
                               "grantee": "PUBLIC", "privilege": privilege, "grant_option": "false",
                               "origin": "null", "default_scope": "", "object_kind": kind})
    owner_by_object = {(row["object_class"], row["object_identity"]): row["owner"] for row in inventory}
    kind_by_object = {(row["object_class"], row["object_identity"]):
                      {"schema": "n", "sequence": "S", "function": "f", "type": "T"}.get(row["object_class"], "r")
                      for row in inventory}
    for item in explicit:
        key = (item["object_class"], item["object_identity"])
        result.append({"requirement_id": item["requirement_id"], "tuple_kind": "object_acl",
                       "object_class": item["object_class"], "object_identity": item["object_identity"],
                       "owner": owner_by_object[key], "grantee": item["grantee"],
                       "privilege": item["privilege"], "grant_option": item["grant_option"],
                       "origin": "explicit", "default_scope": "", "object_kind": kind_by_object[key]})
    observed_columns = sorted({row["object_identity"] for row in observed_rows
                               if row["tuple_kind"] == "column_acl"})
    table_owner = {row["object_identity"]: row["owner"] for row in inventory if row["object_class"] == "table"}
    for item in columns:
        prefix = item["table_identity"] + "."
        available = [identity for identity in observed_columns if identity.startswith(prefix)]
        selector = item["column_selector"]
        if selector.startswith("*except:"):
            excluded = set(selector.removeprefix("*except:").split(","))
            identities = [identity for identity in available if identity.removeprefix(prefix) not in excluded]
        else:
            wanted = set(selector.split(",")); identities = [prefix + name for name in sorted(wanted)]
        for identity in identities:
            result.append({"requirement_id": item["requirement_id"], "tuple_kind": "column_acl",
                           "object_class": "column", "object_identity": identity,
                           "owner": table_owner[item["table_identity"]], "grantee": item["grantee"],
                           "privilege": item["privilege"], "grant_option": item["grant_option"],
                           "origin": "explicit", "default_scope": "", "object_kind": "r"})
    for item in defaults:
        kind, state, owner = item["object_kind"], item["expected_row_state"], item["owner"]
        privileges = [(owner, privilege) for privilege in OWNER_PRIVILEGES[kind]]
        if state == "null" and kind in {"f", "T"}:
            privileges += [("PUBLIC", privilege) for privilege in
                           (("EXECUTE",) if kind == "f" else ("USAGE",))]
        for grantee, privilege in privileges:
            result.append({"requirement_id": item["requirement_id"], "tuple_kind": "default_acl",
                           "object_class": "default", "object_identity": f"{owner}:{item['scope']}:{kind}",
                           "owner": owner, "grantee": grantee, "privilege": privilege,
                           "grant_option": "false", "origin": state,
                           "default_scope": item["scope"], "object_kind": kind})
    return result


def reconcile_acl_manifest_set(manifest_root: Path, observed_capture: Path, requirement_id: str,
                               run_id: str, cluster_id: str, attested_query_execution_ids: set[str],
                               manifest_set_digest: str, observed_capture_digest: str,
                               observed_capture_bytes: bytes | None = None) -> dict[str, str]:
    if observed_capture_bytes is None:
        raise AclCatalogError("snapshot-bytes-required")
    observed_rows = _read(observed_capture, OBSERVED_FIELDS, observed_capture_bytes)
    expected_rows = [row for row in expected_rows_from_manifest_set(manifest_root, observed_rows)
                     if row["requirement_id"] == requirement_id]
    if not expected_rows:
        raise AclCatalogError("missing-expected-tuples")
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=EXPECTED_FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(expected_rows)
    expected_bytes = target.getvalue().encode()
    comparison = reconcile_acl_catalog(
        Path("manifest-set"), observed_capture, requirement_id, run_id, cluster_id,
        attested_query_execution_ids, hashlib.sha256(expected_bytes).hexdigest(),
        observed_capture_digest, expected_manifest_bytes=expected_bytes,
        observed_capture_bytes=observed_capture_bytes)
    comparison["manifest_set_digest"] = manifest_set_digest
    return comparison


def reconcile_acl_catalog(expected_manifest: Path, observed_capture: Path, requirement_id: str,
                          run_id: str, cluster_id: str, attested_query_execution_ids: set[str],
                          expected_manifest_digest: str, observed_capture_digest: str,
                          expected_manifest_bytes: bytes | None = None,
                          observed_capture_bytes: bytes | None = None) -> dict[str, str]:
    if expected_manifest_bytes is None or observed_capture_bytes is None:
        raise AclCatalogError("snapshot-bytes-required")
    expected_bytes = expected_manifest_bytes
    observed_bytes = observed_capture_bytes
    if hashlib.sha256(expected_bytes).hexdigest() != expected_manifest_digest:
        raise AclCatalogError("stale-expected-manifest-digest")
    if hashlib.sha256(observed_bytes).hexdigest() != observed_capture_digest:
        raise AclCatalogError("stale-observed-catalog-digest")
    expected_rows = [row for row in _read(expected_manifest, EXPECTED_FIELDS, expected_bytes)
                     if row["requirement_id"] == requirement_id]
    all_observed_rows = _read(observed_capture, OBSERVED_FIELDS, observed_bytes)
    if not expected_rows:
        raise AclCatalogError("missing-expected-tuples")
    if not all_observed_rows:
        raise AclCatalogError("missing-observed-tuples")
    for row in all_observed_rows:
        if row["capture_version"] != "h1-acl-catalog-capture-v1":
            raise AclCatalogError("stale-catalog-capture")
        if row["run_id"] != run_id:
            raise AclCatalogError("wrong-observed-run-identity")
        if row["cluster_id"] != cluster_id:
            raise AclCatalogError("wrong-observed-cluster-identity")
        if row["query_execution_id"] not in attested_query_execution_ids:
            raise AclCatalogError("unattested-catalog-query-execution")
        if not row["query_id"].startswith("pg-catalog-acl-query-"):
            raise AclCatalogError("wrong-catalog-query-identity")
        if row["origin"] not in {"null", "explicit"}:
            raise AclCatalogError("invalid-acl-origin")
        if row["origin"] == "null" and not row["default_acl_source"]:
            raise AclCatalogError("missing-acldefault-source")
        if row["origin"] == "explicit" and not row["raw_acl"]:
            raise AclCatalogError("missing-explicit-raw-acl")
    # The generator captures a requirement-agnostic catalog.  Expected object
    # identities enter only here, at comparison time.  Rows for unrelated
    # objects remain in the immutable capture but cannot influence a result.
    expected_object_keys = {(row["tuple_kind"], row["object_class"], row["object_identity"],
                             row["owner"], row["default_scope"], row["object_kind"])
                            for row in expected_rows}
    observed_rows = [row for row in all_observed_rows
                     if (row["tuple_kind"], row["object_class"], row["object_identity"],
                         row["owner"], row["default_scope"], row["object_kind"])
                     in expected_object_keys]
    if not observed_rows:
        raise AclCatalogError("missing-observed-tuples")
    expected = [_tuple(row) for row in expected_rows]
    observed = [_tuple(row) for row in observed_rows]
    if len(expected) != len(set(expected)):
        raise AclCatalogError("duplicate-expected-tuple")
    if len(observed) != len(set(observed)):
        raise AclCatalogError("duplicate-observed-tuple")
    missing = sorted(set(expected) - set(observed))
    if missing:
        missing_tuple = missing[0]
        candidates = [_tuple(row) for row in all_observed_rows
                      if row["object_class"] == missing_tuple[1] and
                      (row["object_identity"] == missing_tuple[2] or
                       row["object_identity"][:48] == missing_tuple[2][:48])]
        raise AclCatalogError(f"missing-catalog-tuple:{missing_tuple}:observed-candidates:{candidates[:8]}")
    extra = sorted(set(observed) - set(expected))
    if extra:
        raise AclCatalogError(f"extra-catalog-tuple:{extra[0]}")
    canonical = "\n".join("\t".join(row) for row in sorted(expected)).encode()
    return {"requirement_id": requirement_id, "comparison": "equal",
            "tuple_count": str(len(expected)), "canonical_tuple_digest": hashlib.sha256(canonical).hexdigest(),
            "expected_manifest_digest": expected_manifest_digest,
            "observed_capture_digest": observed_capture_digest, "run_id": run_id, "cluster_id": cluster_id}
