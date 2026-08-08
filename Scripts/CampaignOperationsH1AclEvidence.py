#!/usr/bin/env python3
"""Normalize and authenticate retained H1 ACL-origin audit evidence."""
from __future__ import annotations
import csv
import hashlib
import sys
from pathlib import Path

HEADER = ["runtime_format_version", "run_id", "fixture_id", "requirement_id",
          "generator_id", "generator_version", "implementation_path", "entry_point",
          "emitted_runtime_record_id", "output_artifact_id", "raw_artifact_id",
          "object_class", "schema_name", "object_identity", "catalog_acl_column",
          "expected_origin", "actual_origin", "direction", "raw_acl_is_null",
          "raw_acl_text", "owner", "acldefault_type", "expanded_acl_count",
          "canonical_acl_expansion", "effective_acl_expansion_digest",
          "expected_sqlstate", "actual_sqlstate", "expected_diagnostic",
          "actual_diagnostic", "expected_stage", "actual_stage", "cleanup_result",
          "raw_artifact_path", "raw_artifact_digest", "record_digest"]

RUNTIME_VERSION = "h1-acl-origin-runtime-v3"
RAW_VERSION = "h1-acl-origin-raw-v3"
GENERATOR_ID = "GEN-ACL-ORIGIN"
GENERATOR_VERSION = "h1-generator-registry-v2"
IMPLEMENTATION_PATH = "Scripts/CampaignOperationsH1AclEvidence.py"
ENTRY_POINT = "generate"

def fail(key: str) -> None:
    print(f"H1O001 key={key} stage=acl-origin-runtime-reconciliation", file=sys.stderr)
    raise SystemExit(1)

def rows(path: Path) -> list[list[str]]:
    try:
        with path.open(newline="") as source:
            return list(csv.reader(source, delimiter="\t"))
    except OSError:
        fail(f"missing:{path}")

def fixtures(path: Path) -> dict[str, list[str]]:
    data = rows(path)
    if not data or data[0] != ["fixture_id", "requirement_id", "object_class",
        "schema_name", "object_identity", "catalog_acl_column", "expected_origin",
        "actual_origin", "direction"]:
        fail("fixture-header")
    result = {}
    for row in data[1:]:
        if len(row) != 9 or row[0] in result:
            fail(f"fixture:{row[0] if row else 'fields'}")
        result[row[0]] = row
    return result

def parse_raw(path: Path, fixture: list[str], run_id: str) -> tuple[list[str], list[list[str]]]:
    data = rows(path)
    if not data or data[0] != ["format", RAW_VERSION]:
        fail(f"raw-header:{fixture[0]}")
    meta = [row for row in data[1:] if row and row[0] == "meta"]
    acl = [row for row in data[1:] if row and row[0] == "acl"]
    if len(meta) != 1 or len(meta[0]) != 25:
        fail(f"raw-meta:{fixture[0]}")
    if any(len(row) != 4 or row[3] not in {"true", "false"} for row in acl):
        fail(f"raw-acl:{fixture[0]}")
    if (meta[0][1] != run_id or meta[0][2] != f"ART-ACL-RAW-{fixture[0]}" or
            meta[0][3:7] != [GENERATOR_ID, GENERATOR_VERSION,
                              IMPLEMENTATION_PATH, ENTRY_POINT] or
            meta[0][7:16] != fixture):
        fail(f"raw-mapping:{fixture[0]}")
    return meta[0], acl

def normalize(fixture: list[str], raw_path: str, raw_digest: str,
              meta: list[str], acl: list[list[str]], run_id: str) -> list[str]:
    # meta: tag + run/artifact/generator metadata + nine fixture fields + raw-null, raw-text, owner, acldefault,
    # actual SQLSTATE, diagnostic, object, stage, cleanup.
    offset = 6
    if meta[10 + offset] not in {"true", "false"} or (meta[10 + offset] == "true") != (meta[8 + offset] == "null"):
        fail(f"raw-origin:{fixture[0]}")
    if (meta[10 + offset] == "true" and meta[11 + offset]) or (meta[10 + offset] == "false" and not meta[11 + offset]):
        fail(f"raw-acl-text:{fixture[0]}")
    canonical = "".join("|".join(row[1:]) + "\n" for row in sorted(acl, key=lambda r: r[1:]))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if meta[14 + offset] != "42501" or not meta[15 + offset].startswith("H1A006"):
        fail(f"production-audit:{fixture[0]}")
    if fixture[4] not in meta[16 + offset] or fixture[4] not in meta[15 + offset]:
        fail(f"audit-object:{fixture[0]}")
    if meta[17 + offset] != "database-audit" or meta[18 + offset] != "PASS":
        fail(f"audit-stage:{fixture[0]}")
    values = [RUNTIME_VERSION, run_id, fixture[0], fixture[1], GENERATOR_ID,
        GENERATOR_VERSION, IMPLEMENTATION_PATH, ENTRY_POINT, f"RT-{fixture[0]}",
        f"ART-RECORD-{fixture[0]}", f"ART-ACL-RAW-{fixture[0]}"] + fixture[2:] + [meta[10 + offset],
        meta[11 + offset] if meta[11 + offset] else "<NULL>", meta[12 + offset],
        meta[13 + offset], str(len(acl)), canonical.rstrip("\n") or "<EMPTY>", digest,
        "42501", meta[14 + offset], "H1A006", meta[15 + offset].split()[0],
        "database-audit", meta[17 + offset], meta[18 + offset], raw_path, raw_digest]
    return values + [hashlib.sha256("\t".join(values).encode()).hexdigest()]

def generate(arguments: list[str]) -> None:
    if len(arguments) != 5:
        raise SystemExit("usage: generate FIXTURES RAW_ROOT RUNTIME RUN_ID")
    expected = fixtures(Path(arguments[1])); raw_root = Path(arguments[2])
    output = []
    for identifier in sorted(expected):
        path = raw_root / f"{identifier}.tsv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else fail(f"missing-raw:{identifier}")
        meta, acl = parse_raw(path, expected[identifier], arguments[4])
        output.append(normalize(expected[identifier], f"raw-acl-origin/{identifier}.tsv",
                                digest, meta, acl, arguments[4]))
    with Path(arguments[3]).open("w", newline="") as target:
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER); writer.writerows(output)

def validate(arguments: list[str]) -> None:
    if len(arguments) != 5:
        raise SystemExit("usage: validate FIXTURES RUNTIME ROOT RUN_ID")
    expected = fixtures(Path(arguments[1])); data = rows(Path(arguments[2]))
    if not data or data[0] != HEADER:
        fail("runtime-header")
    found = set()
    for values in data[1:]:
        if len(values) != len(HEADER):
            fail(f"fields:{len(values)}")
        row = dict(zip(HEADER, values)); identifier = row["fixture_id"]
        if identifier not in expected or identifier in found:
            fail(f"runtime-key:{identifier}")
        found.add(identifier)
        if row["run_id"] != arguments[4]:
            fail(f"stale-run:{identifier}")
        relative = row["raw_artifact_path"]
        if relative.startswith("/") or ".." in Path(relative).parts:
            fail(f"raw-path:{identifier}")
        path = Path(arguments[3]) / relative
        if not path.is_file():
            fail(f"missing-raw:{identifier}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["raw_artifact_digest"]:
            fail(f"raw-digest:{identifier}")
        meta, acl = parse_raw(path, expected[identifier], arguments[4])
        normalized = normalize(expected[identifier], relative, digest, meta, acl, arguments[4])
        for index, (actual, wanted) in enumerate(zip(values, normalized)):
            if actual != wanted:
                fail(f"normalized:{identifier}:{HEADER[index]}")
    missing = sorted(set(expected) - found)
    if missing: fail(f"missing:{missing[0]}")
    print(f"H1_ACL_ORIGIN_RUNTIME_RECONCILIATION_OK rows={len(found)}")

if __name__ == "__main__":
    if len(sys.argv) == 8 and sys.argv[1:3] == ["generate", "attest"]:
        from CampaignOperationsH1EvidencePayloadGenerator import main as attest
        sys.argv = [sys.argv[0], "generate", *sys.argv[3:]]
        attest()
        raise SystemExit(0)
    if len(sys.argv) < 2 or sys.argv[1] not in {"generate", "validate"}:
        raise SystemExit("expected generate or validate")
    (generate if sys.argv[1] == "generate" else validate)(sys.argv[1:])
