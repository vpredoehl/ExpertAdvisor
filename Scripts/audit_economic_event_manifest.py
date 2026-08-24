#!/usr/bin/env python3
"""Audit every shared economic-event manifest entry without opening PostgreSQL."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import tempfile
from collections import defaultdict

from authoritative_acquisition import ACQUISITION_COLUMNS


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


def load_manifest(
    path: pathlib.Path,
) -> tuple[list[str], str, list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4:
        raise ValueError("manifest has no entries")
    if lines[0] != "manifest_version\t1":
        raise ValueError("unsupported manifest version")
    if not lines[1].startswith("parser_version\t"):
        raise ValueError("manifest parser version missing")
    if tuple(lines[2].split("\t")) != INGESTION_COLUMNS:
        raise ValueError("unsupported manifest columns")
    rows = list(csv.DictReader(lines[2:], delimiter="\t"))
    if not rows:
        raise ValueError("manifest has no entries")
    return lines[:3], lines[1].split("\t", 1)[1], rows


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_acquisition_manifest(
    path: pathlib.Path | None,
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    if path is None:
        return {}, []
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ACQUISITION_COLUMNS:
            raise ValueError("unsupported acquisition manifest columns")
        rows = list(reader)
    by_url: dict[str, dict[str, str]] = {}
    for row in rows:
        requested = row["requested_url"]
        if requested in by_url:
            raise ValueError(f"duplicate acquisition requested URL: {requested}")
        by_url[requested] = row
    return by_url, rows


def audit_entry(
    audit_binary: pathlib.Path,
    agency: str,
    manifest: pathlib.Path,
    header: list[str],
    row: dict[str, str],
    acquisition: dict[str, str] | None,
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
    audited["acquisition_status"] = (
        acquisition["acquisition_status"] if acquisition else "not_audited"
    )
    audited["requested_url"] = acquisition["requested_url"] if acquisition else row["source_url"]
    audited["final_url"] = acquisition["final_url"] if acquisition else ""
    audited["occurrence_identity"] = acquisition["occurrence_identity"] if acquisition else ""
    audited["provenance_status"] = "validated"
    audited["parse_status"] = "not_run"
    audited["validation_status"] = "not_run"
    audited["collision_status"] = "not_run"
    audited["acceptance_status"] = "rejected"
    audited["import_status"] = "not_run"
    audited["acquisition_diagnostic"] = acquisition["diagnostic"] if acquisition else ""
    audited["provenance_diagnostic"] = ""
    audited["parse_diagnostic"] = ""
    audited["validation_diagnostic"] = ""
    audited["collision_diagnostic"] = ""
    audited["import_diagnostic"] = ""
    diagnostic = result.stderr.strip().replace("\t", " ").replace("\n", " | ")
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
        else:
            audited["validation_status"] = "failed"
            audited["validation_diagnostic"] = diagnostic
    elif "_manifest_" in diagnostic:
        audited["provenance_status"] = "failed"
        audited["provenance_diagnostic"] = diagnostic
    else:
        audited["parse_status"] = "failed"
        audited["parse_diagnostic"] = diagnostic
    return audited


def apply_batch_validation(audited: list[dict[str, str]]) -> None:
    """Audit identity independently of candidate-level validation."""
    identities: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    timestamps: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)

    for row in audited:
        if row["parse_status"] != "parsed":
            continue
        row["collision_status"] = "validated"
        if row["source_agency"] and row["source_event_id"]:
            identities[(row["source_agency"], row["source_event_id"])].append(row)
        if (
            row["source_agency"]
            and row["raw_event_family"]
            and row["event_timestamp_unix_micros"]
        ):
            timestamps[
                (
                    row["source_agency"],
                    row["raw_event_family"],
                    row["event_timestamp_unix_micros"],
                )
            ].append(row)

    duplicate_evidence: set[int] = set()

    def preferred_duplicate(
        rows: list[dict[str, str]],
    ) -> dict[str, str] | None:
        if not rows:
            return None
        semantic_columns = (
            "source_agency",
            "raw_event_family",
            "source_event_id",
            "event_timestamp_unix_micros",
            "timestamp_confidence",
            "source_release_date",
            "source_release_time",
            "source_timezone",
            "reference_period",
            "artifact_sha256",
            "extractor",
        )
        if len({tuple(row.get(column, "") for column in semantic_columns) for row in rows}) != 1:
            return None
        agencies = {row.get("source_agency") for row in rows}
        if agencies == {"CENSUS"}:
            reference = rows[0].get("reference_period", "").replace("-", "")
            preferred = [
                row for row in rows
                if re.search(
                    rf"_{re.escape(reference)}\.pdf$", row.get("source_url", "")
                )
            ]
        elif agencies == {"DOL_ETA"}:
            preferred = []
            for row in rows:
                match = re.search(
                    r"/press/(?P<year>[0-9]{4})/(?P<date>[0-9]{6})\.(?:asp|pdf)$",
                    row.get("source_url", ""),
                    re.IGNORECASE,
                )
                if match and match.group("year")[-2:] == match.group("date")[-2:]:
                    preferred.append(row)
        else:
            return None
        return preferred[0] if len(preferred) == 1 else None

    for rows in identities.values():
        if len(rows) < 2:
            continue
        preferred = preferred_duplicate(rows)
        if preferred is None:
            continue
        for row in rows:
            if row is preferred:
                continue
            duplicate_evidence.add(id(row))
            row["collision_status"] = "duplicate_evidence"
            row["collision_diagnostic"] = (
                "economic_event_duplicate_authoritative_artifact_alias"
            )

    def active(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        return [row for row in rows if id(row) not in duplicate_evidence]

    def permitted_timestamp_group(rows: list[dict[str, str]]) -> bool:
        return (
            len(rows) == 2
            and {row.get("source_agency") for row in rows} == {"FEDERAL_RESERVE"}
            and {row.get("raw_event_family") for row in rows} == {"FOMC_STATEMENT"}
            and {row.get("timestamp_confidence") for row in rows} == {"date_only"}
            and {row.get("source_release_date") for row in rows} == {"2014-09-17"}
            and {row.get("source_event_id") for row in rows} == {
                "federal_reserve:monetary20140917a",
                "federal_reserve:monetary20140917c",
            }
        )

    def fail(rows: list[dict[str, str]], diagnostic: str) -> None:
        if len(rows) < 2:
            return
        for row in rows:
            row["collision_status"] = "failed"
            if row["collision_diagnostic"]:
                row["collision_diagnostic"] += " | "
            row["collision_diagnostic"] += diagnostic

    for rows in identities.values():
        fail(active(rows), "economic_event_duplicate_source_event_id_in_batch")
    for rows in timestamps.values():
        active_rows = active(rows)
        if not permitted_timestamp_group(active_rows):
            fail(
                active_rows,
                "economic_event_duplicate_agency_family_timestamp_in_batch",
            )

    for row in audited:
        if (
            row["acquisition_status"] in {"succeeded", "not_audited"}
            and row["provenance_status"] == "validated"
            and row["parse_status"] == "parsed"
            and row["validation_status"] == "validated"
            and row["collision_status"] == "validated"
        ):
            row["acceptance_status"] = "accepted"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-binary", required=True, type=pathlib.Path)
    parser.add_argument(
        "--agency",
        required=True,
        choices=("bea", "bls", "census", "dol-eta", "federal-reserve"),
    )
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--output-json", required=True, type=pathlib.Path)
    parser.add_argument("--output-tsv", required=True, type=pathlib.Path)
    parser.add_argument("--accepted-manifest", type=pathlib.Path)
    parser.add_argument("--acquisition-manifest", type=pathlib.Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    if args.jobs <= 0:
        parser.error("--jobs must be positive")

    manifest = args.manifest.resolve()
    audit_binary = args.audit_binary.resolve()
    header, parser_version, rows = load_manifest(manifest)
    acquisitions_by_url, acquisition_rows = load_acquisition_manifest(
        args.acquisition_manifest.resolve() if args.acquisition_manifest else None
    )
    if acquisitions_by_url:
        missing = sorted(
            row["source_url"] for row in rows
            if row["source_url"] not in acquisitions_by_url
        )
        if missing:
            raise ValueError(
                f"acquisition manifest is missing {len(missing)} source URLs"
            )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        audited = list(executor.map(
            lambda row: audit_entry(
                audit_binary,
                args.agency,
                manifest,
                header,
                row,
                acquisitions_by_url.get(row["source_url"]),
            ),
            rows,
        ))
    apply_batch_validation(audited)
    completed = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()
    document = {
        "audit_version": 2,
        "agency": args.agency,
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "source_manifest_filesystem_mtime_utc": datetime.datetime.fromtimestamp(
            manifest.stat().st_mtime, datetime.UTC
        ).replace(microsecond=0).isoformat(),
        "parser_provenance": {
            "parser_version": parser_version,
            "audit_binary_sha256": sha256_file(audit_binary),
            "audit_script_sha256": sha256_file(pathlib.Path(__file__)),
            "shared_ingestion_contract_version": 1,
        },
        "audit_completed_at_utc": completed,
        "enumerated_occurrence_count": len(audited),
        "manifest_candidate_count": len(audited),
        "acquired_resource_count": len(acquisition_rows),
        "acquisition_success_count": sum(
            row["acquisition_status"] == "succeeded" for row in audited
        ),
        "acquisition_failure_count": sum(
            row["acquisition_status"] == "failed" for row in audited
        ),
        "acquisition_resource_failure_count": sum(
            row["acquisition_status"] == "failed" for row in acquisition_rows
        ),
        "acquisition_not_audited_count": 0 if acquisition_rows else len(audited),
        "provenance_validated_count": sum(
            row["provenance_status"] == "validated" for row in audited
        ),
        "parsed_count": sum(row["parse_status"] == "parsed" for row in audited),
        "parse_failure_count": sum(row["parse_status"] == "failed" for row in audited),
        "validation_success_count": sum(
            row["validation_status"] == "validated" for row in audited
        ),
        "validation_failure_count": sum(
            row["validation_status"] == "failed" for row in audited
        ),
        "collision_failure_count": sum(
            row["collision_status"] == "failed" for row in audited
        ),
        "duplicate_evidence_count": sum(
            row["collision_status"] == "duplicate_evidence" for row in audited
        ),
        "accepted_candidate_count": sum(
            row["acceptance_status"] == "accepted" for row in audited
        ),
        "import_success_count": 0,
        "import_failure_count": 0,
        "import_not_run_count": len(audited),
        "entries": audited,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(INGESTION_COLUMNS) + list(CANDIDATE_COLUMNS) + [
        "requested_url",
        "final_url",
        "occurrence_identity",
        "acquisition_status",
        "provenance_status",
        "parse_status",
        "validation_status",
        "collision_status",
        "acceptance_status",
        "import_status",
        "acquisition_diagnostic",
        "provenance_diagnostic",
        "parse_diagnostic",
        "validation_diagnostic",
        "collision_diagnostic",
        "import_diagnostic",
    ]
    with args.output_tsv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(audited)

    if args.accepted_manifest:
        accepted = [
            row for row in audited
            if row["acceptance_status"] == "accepted"
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
        f" agency={args.agency} enumerated={document['enumerated_occurrence_count']}"
        f" acquisition_succeeded={document['acquisition_success_count']}"
        f" acquisition_failed={document['acquisition_failure_count']}"
        f" provenance_validated={document['provenance_validated_count']}"
        f" parse_succeeded={document['parsed_count']}"
        f" parse_failed={document['parse_failure_count']}"
        f" validation_succeeded={document['validation_success_count']}"
        f" validation_failed={document['validation_failure_count']}"
        f" collision_failed={document['collision_failure_count']}"
        f" duplicate_evidence={document['duplicate_evidence_count']}"
        f" accepted={document['accepted_candidate_count']}"
        f" import_succeeded={document['import_success_count']}"
        f" import_failed={document['import_failure_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
