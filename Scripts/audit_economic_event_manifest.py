#!/usr/bin/env python3
"""Audit every shared economic-event manifest entry without opening PostgreSQL."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import json
import pathlib
import subprocess
import tempfile
from collections import defaultdict


INGESTION_COLUMNS = (
    "artifact_path",
    "artifact_sha256",
    "artifact_type",
    "source_url",
    "source_artifact_path",
    "source_artifact_sha256",
    "extractor",
)
CANDIDATE_COLUMNS = (
    "source_agency",
    "raw_event_family",
    "source_event_id",
    "source_url",
    "event_timestamp_unix_micros",
    "timestamp_confidence",
    "source_release_date",
    "source_release_time",
    "source_timezone",
    "reference_period",
)


def load_manifest(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4:
        raise ValueError("manifest has no entries")
    if lines[0] != "manifest_version\t1":
        raise ValueError("unsupported manifest version")
    if tuple(lines[2].split("\t")) != INGESTION_COLUMNS:
        raise ValueError("unsupported manifest columns")
    rows = list(csv.DictReader(lines[2:], delimiter="\t"))
    if not rows:
        raise ValueError("manifest has no entries")
    return lines[:3], rows


def audit_entry(
    audit_binary: pathlib.Path,
    agency: str,
    manifest: pathlib.Path,
    header: list[str],
    row: dict[str, str],
) -> dict[str, str]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=manifest.parent,
        prefix=".phase6a-entry-",
        suffix=".tsv",
        delete=False,
    ) as temporary:
        temporary.write("\n".join(header) + "\n")
        temporary.write("\t".join(row[column] for column in INGESTION_COLUMNS) + "\n")
        temporary_path = pathlib.Path(temporary.name)
    try:
        result = subprocess.run(
            [str(audit_binary), agency, str(temporary_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    finally:
        temporary_path.unlink(missing_ok=True)

    audited = dict(row)
    audited["provenance_status"] = "validated"
    audited["parse_status"] = "not_run"
    audited["validation_status"] = "not_run"
    audited["diagnostic"] = result.stderr.strip().replace("\t", " ").replace("\n", " | ")
    for column in CANDIDATE_COLUMNS:
        if column not in audited:
            audited[column] = ""
    if result.returncode in {0, 3}:
        candidates = list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))
        if len(candidates) != 1:
            raise RuntimeError("one-entry audit did not emit exactly one candidate")
        audited.update(candidates[0])
        audited["parse_status"] = "parsed"
        if result.returncode == 0:
            audited["validation_status"] = "validated"
            audited["diagnostic"] = ""
        else:
            audited["validation_status"] = "failed"
    elif "_manifest_" in audited["diagnostic"]:
        audited["provenance_status"] = "failed"
    return audited


def apply_batch_validation(audited: list[dict[str, str]]) -> None:
    """Mark entries that violate shared whole-batch identity constraints."""
    identities: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    timestamps: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)

    for row in audited:
        if row["validation_status"] != "validated":
            continue
        identities[(row["source_agency"], row["source_event_id"])].append(row)
        timestamps[
            (
                row["source_agency"],
                row["raw_event_family"],
                row["event_timestamp_unix_micros"],
            )
        ].append(row)

    def fail(rows: list[dict[str, str]], diagnostic: str) -> None:
        if len(rows) < 2:
            return
        for row in rows:
            row["validation_status"] = "failed"
            if row["diagnostic"]:
                row["diagnostic"] += " | "
            row["diagnostic"] += diagnostic

    for rows in identities.values():
        fail(rows, "economic_event_duplicate_source_event_id_in_batch")
    for rows in timestamps.values():
        fail(rows, "economic_event_duplicate_agency_family_timestamp_in_batch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-binary", required=True, type=pathlib.Path)
    parser.add_argument(
        "--agency",
        required=True,
        choices=("bea", "census", "dol-eta", "federal-reserve"),
    )
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--output-json", required=True, type=pathlib.Path)
    parser.add_argument("--output-tsv", required=True, type=pathlib.Path)
    parser.add_argument("--accepted-manifest", type=pathlib.Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    if args.jobs <= 0:
        parser.error("--jobs must be positive")

    manifest = args.manifest.resolve()
    audit_binary = args.audit_binary.resolve()
    header, rows = load_manifest(manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        audited = list(executor.map(
            lambda row: audit_entry(
                audit_binary, args.agency, manifest, header, row
            ),
            rows,
        ))
    apply_batch_validation(audited)
    completed = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()
    document = {
        "audit_version": 1,
        "agency": args.agency,
        "source_manifest": str(manifest),
        "source_manifest_completed_at_utc": datetime.datetime.fromtimestamp(
            manifest.stat().st_mtime, datetime.UTC
        ).replace(microsecond=0).isoformat(),
        "audit_completed_at_utc": completed,
        "occurrence_count": len(audited),
        "provenance_validated_count": sum(
            row["provenance_status"] == "validated" for row in audited
        ),
        "parsed_count": sum(row["parse_status"] == "parsed" for row in audited),
        "validated_count": sum(
            row["validation_status"] == "validated" for row in audited
        ),
        "entries": audited,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(INGESTION_COLUMNS) + list(CANDIDATE_COLUMNS) + [
        "provenance_status",
        "parse_status",
        "validation_status",
        "diagnostic",
    ]
    with args.output_tsv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(audited)

    if args.accepted_manifest:
        accepted = [
            row for row in audited
            if row["validation_status"] == "validated"
        ]
        args.accepted_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.accepted_manifest.write_text(
            "\n".join(header) + "\n" +
            "".join(
                "\t".join(row[column] for column in INGESTION_COLUMNS) + "\n"
                for row in accepted
            ),
            encoding="utf-8",
        )

    print(
        "ECONOMIC_EVENT_ENTRY_AUDIT_COMPLETE"
        f" agency={args.agency} occurrences={len(audited)}"
        f" provenance_validated={document['provenance_validated_count']}"
        f" parsed={document['parsed_count']} validated={document['validated_count']}"
        f" failed={len(audited) - document['parsed_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
