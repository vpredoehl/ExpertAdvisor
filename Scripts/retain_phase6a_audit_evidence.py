#!/usr/bin/env python3
"""Retain compact Phase 6A evidence without copying downloaded source files."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import shutil


AGENCIES = {
    "bea": "bea-union-final",
    "bls": "bls-final",
    "census": "census-final",
    "dol-eta": "dol-eta-final",
    "federal-reserve": "federal-reserve-final",
}

DATABASE_VERIFICATION = re.compile(
    r"PHASE6A_DATABASE_VERIFICATION,row_count=(?P<row_count>[0-9]+),"
    r"duplicate_source_identity_count=(?P<duplicate_source_identity_count>[0-9]+),"
    r"disallowed_same_time_count=(?P<disallowed_same_time_count>[0-9]+)"
)

DATABASE_DROPPED = re.compile(
    r"PHASE6A_DATABASE_DROPPED,database=(?P<database>[^,]+),dropped=true"
)

IMPORT_SUMMARY = re.compile(
    r"ECONOMIC_EVENT_IMPORT_SUMMARY,agency=(?P<agency>[^,]+),"
    r"mode=(?P<mode>[^,]+),inserted=(?P<inserted>[0-9]+),"
    r"unchanged=(?P<unchanged>[0-9]+),updated=(?P<updated>[0-9]+),"
    r"rejected=(?P<rejected>[0-9]+)"
)


def read_tsv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def write_tsv(
    path: pathlib.Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_import(path: pathlib.Path) -> dict[str, object]:
    match = IMPORT_SUMMARY.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"import summary missing: {path}")
    result: dict[str, object] = {
        "agency": match.group("agency"),
        "mode": match.group("mode"),
    }
    for name in ("inserted", "unchanged", "updated", "rejected"):
        result[name] = int(match.group(name))
    return result




def parse_database_verification(path: pathlib.Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    verification = DATABASE_VERIFICATION.search(text)
    dropped = DATABASE_DROPPED.search(text)
    if verification is None:
        raise RuntimeError(f"database verification summary missing: {path}")
    if dropped is None:
        raise RuntimeError(f"database drop evidence missing: {path}")
    return {
        "post_repeat_database_row_count": int(verification.group("row_count")),
        "duplicate_source_identity_count": int(
            verification.group("duplicate_source_identity_count")
        ),
        "disallowed_same_time_count": int(
            verification.group("disallowed_same_time_count")
        ),
        "disposable_database": dropped.group("database"),
        "disposable_database_dropped": True,
    }


def parse_build_verification(path: pathlib.Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "build_succeeded": "** BUILD SUCCEEDED **" in text,
        "development_derived_data": "DerivedData/Development" in text,
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-area", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--database-verification-output", required=True, type=pathlib.Path
    )
    parser.add_argument("--build-output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    work = args.work_area.resolve()
    output = args.output_dir.resolve()
    database_verification_output = args.database_verification_output.resolve()
    build_output = args.build_output.resolve()
    if not database_verification_output.is_file():
        raise RuntimeError(
            f"database verification output missing: {database_verification_output}"
        )
    if not build_output.is_file():
        raise RuntimeError(f"build output missing: {build_output}")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output}")

    enumerated: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    collisions: list[dict[str, object]] = []
    parser_provenance: dict[str, object] = {}
    audit_counts: dict[str, object] = {}
    first_imports: list[dict[str, object]] = []
    repeat_imports: list[dict[str, object]] = []

    for agency, directory in AGENCIES.items():
        area = work / directory
        rows = read_tsv(area / "audit.tsv")
        audit = json.loads((area / "audit.json").read_text(encoding="utf-8"))
        parser_provenance[agency] = parser_provenance_entry = dict(
            audit["parser_provenance"]
        )
        parser_provenance_entry["source_manifest_sha256"] = audit[
            "source_manifest_sha256"
        ]
        audit_counts[agency] = {
            name: audit[name]
            for name in (
                "enumerated_occurrence_count",
                "acquisition_success_count",
                "acquisition_failure_count",
                "parsed_count",
                "parse_failure_count",
                "validation_success_count",
                "validation_failure_count",
                "collision_failure_count",
                "duplicate_evidence_count",
                "accepted_candidate_count",
            )
        }

        for row in rows:
            common = {
                "agency": agency,
                "artifact_path": row["artifact_path"],
                "artifact_sha256": row["artifact_sha256"],
                "source_url": row["source_url"],
            }
            enumerated.append(common | {
                "requested_url": row["requested_url"],
                "final_url": row["final_url"],
                "occurrence_identity": row["occurrence_identity"],
                "acquisition_status": row["acquisition_status"],
                "provenance_status": row["provenance_status"],
            })
            if row["acceptance_status"] == "accepted":
                accepted.append(common | {
                    "event_family": row["raw_event_family"],
                    "source_event_id": row["source_event_id"],
                    "event_timestamp_unix_micros": row[
                        "event_timestamp_unix_micros"
                    ],
                    "timestamp_confidence": row["timestamp_confidence"],
                    "source_release_date": row["source_release_date"],
                    "source_release_time": row["source_release_time"],
                    "source_timezone": row["source_timezone"],
                    "reference_period": row["reference_period"],
                })
            else:
                rejected.append(common | {
                    "acquisition_status": row["acquisition_status"],
                    "parse_status": row["parse_status"],
                    "validation_status": row["validation_status"],
                    "collision_status": row["collision_status"],
                    "diagnostic": " | ".join(filter(None, (
                        row["acquisition_diagnostic"],
                        row["provenance_diagnostic"],
                        row["parse_diagnostic"],
                        row["validation_diagnostic"],
                        row["collision_diagnostic"],
                    ))),
                })
            if row["collision_status"] in {"failed", "duplicate_evidence"}:
                collisions.append(common | {
                    "source_event_id": row["source_event_id"],
                    "collision_status": row["collision_status"],
                    "diagnostic": row["collision_diagnostic"],
                })

        first_imports.append(parse_import(area / "import-first.txt"))
        repeat_imports.append(parse_import(area / "import-repeat.txt"))

    write_tsv(
        output / "enumerated-source-manifest.tsv",
        ["agency", "artifact_path", "artifact_sha256", "source_url",
         "requested_url", "final_url", "occurrence_identity",
         "acquisition_status", "provenance_status"],
        enumerated,
    )
    write_tsv(
        output / "accepted-event-manifest.tsv",
        ["agency", "event_family", "source_event_id",
         "event_timestamp_unix_micros", "timestamp_confidence",
         "source_release_date", "source_release_time", "source_timezone",
         "reference_period", "source_url", "artifact_path", "artifact_sha256"],
        accepted,
    )
    write_tsv(
        output / "rejected-event-report.tsv",
        ["agency", "artifact_path", "artifact_sha256", "source_url",
         "acquisition_status", "parse_status", "validation_status",
         "collision_status", "diagnostic"],
        rejected,
    )
    write_tsv(
        output / "collision-report.tsv",
        ["agency", "source_event_id", "collision_status", "source_url",
         "artifact_path", "artifact_sha256", "diagnostic"],
        collisions,
    )
    (output / "parser-provenance.json").write_text(
        json.dumps(parser_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "audit-counts.json").write_text(
        json.dumps(audit_counts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    database_verification = parse_database_verification(
        database_verification_output
    )
    build_verification = parse_build_verification(build_output)
    if not build_verification["build_succeeded"]:
        raise RuntimeError(f"successful build marker missing: {build_output}")
    if not build_verification["development_derived_data"]:
        raise RuntimeError(f"Development DerivedData evidence missing: {build_output}")

    import_summary = {
        "first_import": first_imports,
        "repeat_import": repeat_imports,
        "first_import_total_inserted": sum(
            int(row["inserted"]) for row in first_imports
        ),
        "repeat_import_total_unchanged": sum(
            int(row["unchanged"]) for row in repeat_imports
        ),
        **database_verification,
    }
    (output / "import-summary.json").write_text(
        json.dumps(import_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "build-verification.json").write_text(
        json.dumps(build_verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(
        database_verification_output, output / "database-verification-raw.txt"
    )
    shutil.copy2(build_output, output / "build-output.txt")
    print(
        "PHASE6A_EVIDENCE_RETAINED "
        f"enumerated={len(enumerated)} accepted={len(accepted)} "
        f"rejected={len(rejected)} collisions={len(collisions)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
