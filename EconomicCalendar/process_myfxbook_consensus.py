#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_CAPTURE = (
    REPOSITORY_ROOT
    / "EconomicCalendar/raw/myfxbook/myfxbook_consensus_history.json"
)
DEFAULT_OANDA_MATCHES = (
    REPOSITORY_ROOT
    / "EconomicCalendar/raw/oanda/oanda_consensus_matches.csv"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "EconomicCalendar/raw/myfxbook/derived"
)

NORMALIZED_FILENAME = "myfxbook_consensus_normalized.csv"
GAP_FILL_FILENAME = "myfxbook_oanda_gap_fill_candidates.csv"
OANDA_RECONCILIATION_FILENAME = "myfxbook_oanda_blank_reconciliation.csv"
EXCLUSIONS_FILENAME = "myfxbook_consensus_exclusions.csv"
SUMMARY_FILENAME = "myfxbook_processing_summary.json"

SOURCE_TO_CANONICAL = {
    "CPI_MOM": "CPI",
    "PPI_MOM": "PPI",
    "DURABLE_GOODS_MOM": "DURABLE_GOODS",
    "GDP": "GDP",
    "PCE": "PCE",
    "EMPLOYMENT_NFP": "EMPLOYMENT",
    "FOMC_RATE": "FOMC",
    "RETAIL_SALES": "RETAIL_SALES",
    "JOLTS": "JOLTS",
}

EXPECTED_EVENT_IDS = {
    "CPI_MOM": -265326876,
    "PPI_MOM": 1438001284,
    "DURABLE_GOODS_MOM": 1842891375,
    "GDP": -2050894417,
    "PCE": 486275182,
    "EMPLOYMENT_NFP": 1201719106,
    "FOMC_RATE": -1731853807,
    "RETAIL_SALES": 1066687555,
    "JOLTS": 906938645,
}

# Inclusive windows established by the read-only OANDA coverage audit.  These
# are deliberately explicit inputs to candidate classification, not inferred
# from today's date or from a mutable database.
OANDA_REMAINING_COVERAGE_WINDOWS = {
    "CPI": (date(2010, 1, 1), date(2011, 7, 14)),
    "PPI": (date(2010, 1, 1), date(2011, 7, 13)),
    "DURABLE_GOODS": (date(2010, 1, 1), date(2011, 7, 26)),
    "GDP": (date(2010, 1, 1), date(2011, 7, 28)),
    "PCE": (date(2010, 1, 1), date(2011, 8, 1)),
    "EMPLOYMENT": (date(2010, 1, 1), date(2011, 8, 4)),
    "FOMC": (date(2010, 1, 1), date(2011, 8, 8)),
    "RETAIL_SALES": (date(2010, 1, 1), date(2011, 8, 11)),
    "JOLTS": (date(2010, 1, 1), date(2023, 12, 4)),
}

MANUAL_REVIEW_KEYS = {
    ("CPI", date(2024, 2, 9)): "known_cpi_2024_02_09_anomaly",
}

EMERGENCY_FOMC_NULL_DATES = {
    date(2020, 3, 3),
    date(2020, 3, 15),
}

COUNT_VALUE_FAMILIES = {
    "EMPLOYMENT",
    "JOLTS",
}

NORMALIZED_FIELDS = [
    "event_family",
    "myfxbook_source_family",
    "myfxbook_event_id",
    "release_date",
    "myfxbook_actual",
    "myfxbook_consensus",
    "consensus_present",
    "family_date_observation_count",
    "classification",
    "automatic_candidate_eligible",
    "automatic_exclusion_reason",
    "source",
    "source_endpoint",
    "source_years_back",
    "source_capture_extracted_at",
    "source_capture_filename",
    "source_capture_sha256",
    "source_series_ordinal",
    "source_observation_ordinal",
]

GAP_FILL_FIELDS = [
    "event_family",
    "release_date",
    "myfxbook_source_family",
    "myfxbook_event_id",
    "myfxbook_actual",
    "myfxbook_consensus",
    "consensus_source",
    "candidate_classification",
    "target_coverage_source",
    "oanda_gap_start_date",
    "oanda_gap_end_date",
    "myfxbook_capture_source",
    "myfxbook_capture_endpoint",
    "myfxbook_capture_years_back",
    "myfxbook_capture_filename",
    "myfxbook_capture_sha256",
    "source_series_ordinal",
    "source_observation_ordinal",
]

OANDA_RECONCILIATION_FIELDS = [
    "event_family",
    "official_event_timestamp_utc",
    "official_release_date",
    "official_source_agency",
    "official_reference_period",
    "target_match_source",
    "oanda_date",
    "oanda_event",
    "oanda_report_id",
    "oanda_event_id",
    "oanda_period",
    "oanda_priority",
    "oanda_actual",
    "original_oanda_forecast",
    "original_oanda_forecast_present",
    "oanda_previous",
    "oanda_timestamp",
    "oanda_previous_present",
    "oanda_actual_present",
    "oanda_match_rule",
    "oanda_source_file",
    "oanda_matches_filename",
    "oanda_matches_sha256",
    "myfxbook_match_count",
    "myfxbook_source_families",
    "myfxbook_event_ids",
    "myfxbook_actual_values",
    "myfxbook_consensus_values",
    "myfxbook_row_classifications",
    "myfxbook_capture_source",
    "myfxbook_capture_endpoint",
    "myfxbook_capture_years_back",
    "myfxbook_capture_filename",
    "myfxbook_capture_sha256",
    "candidate_myfxbook_event_id",
    "candidate_myfxbook_actual",
    "candidate_myfxbook_consensus",
    "candidate_consensus_source",
    "merge_classification",
    "merge_exclusion_reason",
    "oanda_match_row_number",
]

EXCLUSION_FIELDS = NORMALIZED_FIELDS + [
    "anomaly_type",
    "anomaly_detail",
]


def repository_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_text(value: Any, field: str, *, decimal_family: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number or null")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if decimal_family and isinstance(value, int):
        return f"{value}.0"
    return str(value)


def parse_myfxbook_date(value: Any) -> date:
    if not isinstance(value, str):
        raise ValueError("Myfxbook row date must be a string")
    try:
        return datetime.strptime(value, "%b %d, '%y").date()
    except ValueError as exc:
        raise ValueError(f"invalid Myfxbook row date: {value!r}") from exc


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(buffer.getvalue().encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_bytes(rendered.encode("utf-8"))


def load_normalized_evidence(raw_capture: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    try:
        payload = json.loads(raw_capture.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read Myfxbook capture {raw_capture}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Myfxbook capture root must be an object")
    if payload.get("source") != "myfxbook":
        raise ValueError("Myfxbook capture source must be 'myfxbook'")
    if payload.get("endpoint") != "calendar-chart-years-back.json":
        raise ValueError("unexpected Myfxbook capture endpoint")
    if payload.get("yearsBack") != 20:
        raise ValueError("Myfxbook capture yearsBack must be 20")
    if not isinstance(payload.get("extractedAt"), str) or not payload["extractedAt"]:
        raise ValueError("Myfxbook capture extractedAt is missing")

    series_values = payload.get("series")
    if not isinstance(series_values, list):
        raise ValueError("Myfxbook capture series must be an array")

    observed_families = [
        series.get("family") if isinstance(series, dict) else None
        for series in series_values
    ]
    if len(observed_families) != len(set(observed_families)):
        raise ValueError("Myfxbook capture contains duplicate source series")
    if set(observed_families) != set(SOURCE_TO_CANONICAL):
        raise ValueError(
            "Myfxbook capture source families differ from the expected nine-series contract"
        )

    capture_filename = repository_display_path(raw_capture)
    capture_sha256 = sha256_file(raw_capture)
    evidence: list[dict[str, str]] = []

    for series_ordinal, series in enumerate(series_values, start=1):
        if not isinstance(series, dict):
            raise ValueError("Myfxbook series entries must be objects")
        source_family = series["family"]
        event_id = series.get("eventId")
        if event_id != EXPECTED_EVENT_IDS[source_family]:
            raise ValueError(f"unexpected Myfxbook event ID for {source_family}")
        observations = series.get("rows")
        if not isinstance(observations, list):
            raise ValueError(f"Myfxbook rows for {source_family} must be an array")

        for observation_ordinal, observation in enumerate(observations, start=1):
            if not isinstance(observation, dict):
                raise ValueError(f"Myfxbook observation for {source_family} must be an object")
            if set(observation) != {"date", "actual", "consensus"}:
                raise ValueError(f"unexpected Myfxbook observation fields for {source_family}")
            release_date = parse_myfxbook_date(observation["date"])
            canonical_family = SOURCE_TO_CANONICAL[source_family]
            decimal_family = canonical_family not in COUNT_VALUE_FAMILIES
            consensus = numeric_text(
                observation["consensus"],
                "consensus",
                decimal_family=decimal_family,
            )
            evidence.append(
                {
                    "event_family": canonical_family,
                    "myfxbook_source_family": source_family,
                    "myfxbook_event_id": str(event_id),
                    "release_date": release_date.isoformat(),
                    "myfxbook_actual": numeric_text(
                        observation["actual"],
                        "actual",
                        decimal_family=decimal_family,
                    ),
                    "myfxbook_consensus": consensus,
                    "consensus_present": "1" if observation["consensus"] is not None else "0",
                    "family_date_observation_count": "",
                    "classification": "",
                    "automatic_candidate_eligible": "",
                    "automatic_exclusion_reason": "",
                    "source": payload["source"],
                    "source_endpoint": payload["endpoint"],
                    "source_years_back": str(payload["yearsBack"]),
                    "source_capture_extracted_at": payload["extractedAt"],
                    "source_capture_filename": capture_filename,
                    "source_capture_sha256": capture_sha256,
                    "source_series_ordinal": str(series_ordinal),
                    "source_observation_ordinal": str(observation_ordinal),
                }
            )

    key_counts = Counter((row["event_family"], row["release_date"]) for row in evidence)
    for row in evidence:
        key = (row["event_family"], row["release_date"])
        count = key_counts[key]
        row["family_date_observation_count"] = str(count)
        manual_reason = MANUAL_REVIEW_KEYS.get((key[0], date.fromisoformat(key[1])))
        if count > 1:
            row["classification"] = "ambiguous_duplicate"
            row["automatic_candidate_eligible"] = "0"
            row["automatic_exclusion_reason"] = "duplicate_family_date"
        elif manual_reason:
            row["classification"] = "manual_review"
            row["automatic_candidate_eligible"] = "0"
            row["automatic_exclusion_reason"] = manual_reason
        elif row["consensus_present"] == "0":
            row["classification"] = "unique_null"
            row["automatic_candidate_eligible"] = "0"
            row["automatic_exclusion_reason"] = "consensus_missing"
        else:
            row["classification"] = "unique_populated"
            row["automatic_candidate_eligible"] = "1"
            row["automatic_exclusion_reason"] = ""

    evidence.sort(
        key=lambda row: (
            row["event_family"],
            row["release_date"],
            int(row["source_series_ordinal"]),
            int(row["source_observation_ordinal"]),
        )
    )
    metadata = {
        "source": payload["source"],
        "endpoint": payload["endpoint"],
        "years_back": payload["yearsBack"],
        "capture_extracted_at": payload["extractedAt"],
        "capture_filename": capture_filename,
        "capture_sha256": capture_sha256,
    }
    return metadata, evidence


def build_gap_fill_candidates(evidence: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for row in evidence:
        if row["automatic_candidate_eligible"] != "1":
            continue
        release_date = date.fromisoformat(row["release_date"])
        gap_start, gap_end = OANDA_REMAINING_COVERAGE_WINDOWS[row["event_family"]]
        if not gap_start <= release_date <= gap_end:
            continue
        candidates.append(
            {
                "event_family": row["event_family"],
                "release_date": row["release_date"],
                "myfxbook_source_family": row["myfxbook_source_family"],
                "myfxbook_event_id": row["myfxbook_event_id"],
                "myfxbook_actual": row["myfxbook_actual"],
                "myfxbook_consensus": row["myfxbook_consensus"],
                "consensus_source": "myfxbook",
                "candidate_classification": "myfxbook_unique_populated_gap_fill",
                "target_coverage_source": "oanda",
                "oanda_gap_start_date": gap_start.isoformat(),
                "oanda_gap_end_date": gap_end.isoformat(),
                "myfxbook_capture_source": row["source"],
                "myfxbook_capture_endpoint": row["source_endpoint"],
                "myfxbook_capture_years_back": row["source_years_back"],
                "myfxbook_capture_filename": row["source_capture_filename"],
                "myfxbook_capture_sha256": row["source_capture_sha256"],
                "source_series_ordinal": row["source_series_ordinal"],
                "source_observation_ordinal": row["source_observation_ordinal"],
            }
        )
    candidates.sort(key=lambda row: (row["event_family"], row["release_date"]))
    return candidates


def anomaly_for(row: dict[str, str]) -> tuple[str, str] | None:
    release_date = date.fromisoformat(row["release_date"])
    family = row["event_family"]
    if row["classification"] == "ambiguous_duplicate":
        return (
            "duplicate_family_date",
            f"{row['family_date_observation_count']} raw observations share this family/date key; no row is selected",
        )
    if (family, release_date) in MANUAL_REVIEW_KEYS:
        return (
            "manual_review",
            "known CPI date anomaly retained as evidence and excluded from automatic candidates",
        )
    if row["consensus_present"] != "0":
        return None
    if family == "PCE":
        return (
            "pce_null_consensus",
            "row-level PCE null retained; no prefix assumption, interpolation, or adjacent-release fill",
        )
    if family == "GDP" and release_date == date(2013, 3, 28):
        return (
            "gdp_null_consensus",
            "known GDP null consensus retained and excluded from automatic candidates",
        )
    if family == "FOMC" and release_date in EMERGENCY_FOMC_NULL_DATES:
        return (
            "emergency_fomc_null_consensus",
            "emergency FOMC event has no source consensus; no forecast is inferred",
        )
    return (
        "null_consensus",
        "source consensus is null and is excluded from automatic candidates",
    )


def build_exclusions(evidence: list[dict[str, str]]) -> list[dict[str, str]]:
    exclusions: list[dict[str, str]] = []
    for row in evidence:
        anomaly = anomaly_for(row)
        if anomaly is None:
            continue
        exclusions.append({**row, "anomaly_type": anomaly[0], "anomaly_detail": anomaly[1]})
    return exclusions


def joined_evidence_values(rows: list[dict[str, str]], field: str) -> str:
    return " | ".join(row[field] if row[field] != "" else "<null>" for row in rows)


def build_oanda_blank_reconciliation(
    evidence: list[dict[str, str]],
    oanda_matches: Path,
) -> list[dict[str, str]]:
    evidence_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in evidence:
        evidence_by_key[(row["event_family"], row["release_date"])].append(row)

    oanda_filename = repository_display_path(oanda_matches)
    oanda_sha256 = sha256_file(oanda_matches)
    with oanda_matches.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "event_family", "official_event_timestamp_utc", "official_release_date",
            "official_source_agency", "official_reference_period", "oanda_date",
            "oanda_event", "oanda_report_id", "oanda_event_id", "oanda_period",
            "oanda_priority", "oanda_actual", "oanda_forecast", "oanda_previous",
            "oanda_timestamp", "forecast_present", "previous_present", "actual_present",
            "match_rule", "source_file",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("OANDA match CSV does not satisfy the expected schema")
        blank_rows = [
            (row_number, row)
            for row_number, row in enumerate(reader, start=2)
            if row["forecast_present"] == "0"
        ]

    reconciled: list[dict[str, str]] = []
    for row_number, oanda in blank_rows:
        if oanda["oanda_forecast"] != "":
            raise ValueError(
                f"OANDA row {row_number} marks forecast absent but retains a value"
            )
        matched = evidence_by_key.get(
            (oanda["event_family"], oanda["official_release_date"]), []
        )
        candidate: dict[str, str] | None = None
        if not matched:
            merge_classification = "myfxbook_no_match"
            merge_exclusion_reason = "no_myfxbook_family_date_match"
        elif len(matched) > 1 or matched[0]["classification"] == "ambiguous_duplicate":
            merge_classification = "myfxbook_ambiguous"
            merge_exclusion_reason = "multiple_myfxbook_family_date_observations"
        elif matched[0]["classification"] == "manual_review":
            merge_classification = "myfxbook_manual_review"
            merge_exclusion_reason = matched[0]["automatic_exclusion_reason"]
        elif matched[0]["consensus_present"] == "0":
            merge_classification = "myfxbook_null"
            merge_exclusion_reason = "myfxbook_consensus_missing"
        else:
            candidate = matched[0]
            merge_classification = "myfxbook_populated_candidate"
            merge_exclusion_reason = ""

        first = evidence[0]
        reconciled.append(
            {
                "event_family": oanda["event_family"],
                "official_event_timestamp_utc": oanda["official_event_timestamp_utc"],
                "official_release_date": oanda["official_release_date"],
                "official_source_agency": oanda["official_source_agency"],
                "official_reference_period": oanda["official_reference_period"],
                "target_match_source": "oanda",
                "oanda_date": oanda["oanda_date"],
                "oanda_event": oanda["oanda_event"],
                "oanda_report_id": oanda["oanda_report_id"],
                "oanda_event_id": oanda["oanda_event_id"],
                "oanda_period": oanda["oanda_period"],
                "oanda_priority": oanda["oanda_priority"],
                "oanda_actual": oanda["oanda_actual"],
                "original_oanda_forecast": oanda["oanda_forecast"],
                "original_oanda_forecast_present": oanda["forecast_present"],
                "oanda_previous": oanda["oanda_previous"],
                "oanda_timestamp": oanda["oanda_timestamp"],
                "oanda_previous_present": oanda["previous_present"],
                "oanda_actual_present": oanda["actual_present"],
                "oanda_match_rule": oanda["match_rule"],
                "oanda_source_file": oanda["source_file"],
                "oanda_matches_filename": oanda_filename,
                "oanda_matches_sha256": oanda_sha256,
                "myfxbook_match_count": str(len(matched)),
                "myfxbook_source_families": joined_evidence_values(matched, "myfxbook_source_family") if matched else "",
                "myfxbook_event_ids": joined_evidence_values(matched, "myfxbook_event_id") if matched else "",
                "myfxbook_actual_values": joined_evidence_values(matched, "myfxbook_actual") if matched else "",
                "myfxbook_consensus_values": joined_evidence_values(matched, "myfxbook_consensus") if matched else "",
                "myfxbook_row_classifications": joined_evidence_values(matched, "classification") if matched else "",
                "myfxbook_capture_source": first["source"],
                "myfxbook_capture_endpoint": first["source_endpoint"],
                "myfxbook_capture_years_back": first["source_years_back"],
                "myfxbook_capture_filename": first["source_capture_filename"],
                "myfxbook_capture_sha256": first["source_capture_sha256"],
                "candidate_myfxbook_event_id": candidate["myfxbook_event_id"] if candidate else "",
                "candidate_myfxbook_actual": candidate["myfxbook_actual"] if candidate else "",
                "candidate_myfxbook_consensus": candidate["myfxbook_consensus"] if candidate else "",
                "candidate_consensus_source": "myfxbook" if candidate else "",
                "merge_classification": merge_classification,
                "merge_exclusion_reason": merge_exclusion_reason,
                "oanda_match_row_number": str(row_number),
            }
        )

    reconciled.sort(
        key=lambda row: (
            row["official_release_date"],
            row["event_family"],
            int(row["oanda_match_row_number"]),
        )
    )
    return reconciled


def counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def process_inputs(
    raw_capture: Path = DEFAULT_RAW_CAPTURE,
    oanda_matches: Path = DEFAULT_OANDA_MATCHES,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    raw_capture = Path(raw_capture)
    oanda_matches = Path(oanda_matches)
    output_directory = Path(output_directory)
    metadata, evidence = load_normalized_evidence(raw_capture)
    gap_candidates = build_gap_fill_candidates(evidence)
    exclusions = build_exclusions(evidence)
    reconciliation = build_oanda_blank_reconciliation(evidence, oanda_matches)

    duplicate_counts = Counter(
        (row["event_family"], row["release_date"]) for row in evidence
    )
    duplicate_keys = sorted(
        f"{family} {release_date}"
        for (family, release_date), count in duplicate_counts.items()
        if count > 1
    )
    gap_counts = Counter(row["event_family"] for row in gap_candidates)
    gap_counts_by_family = {
        family: gap_counts.get(family, 0)
        for family in sorted(OANDA_REMAINING_COVERAGE_WINDOWS)
    }
    source_counts = Counter(row["myfxbook_source_family"] for row in evidence)
    canonical_counts = Counter(row["event_family"] for row in evidence)
    summary: dict[str, Any] = {
        "artifacts": {
            "exclusions": EXCLUSIONS_FILENAME,
            "gap_fill_candidates": GAP_FILL_FILENAME,
            "normalized_evidence": NORMALIZED_FILENAME,
            "oanda_blank_reconciliation": OANDA_RECONCILIATION_FILENAME,
        },
        "inputs": {
            "myfxbook_capture": metadata,
            "oanda_matches": {
                "filename": repository_display_path(oanda_matches),
                "sha256": sha256_file(oanda_matches),
            },
        },
        "normalized_evidence": {
            "rows": len(evidence),
            "by_canonical_family": dict(sorted(canonical_counts.items())),
            "by_source_family": dict(sorted(source_counts.items())),
            "classification_counts": counter_dict(row["classification"] for row in evidence),
            "source_to_canonical_mapping": dict(sorted(SOURCE_TO_CANONICAL.items())),
        },
        "duplicates": {
            "family_date_key_count": len(duplicate_keys),
            "observation_count": sum(count for count in duplicate_counts.values() if count > 1),
            "family_date_keys": duplicate_keys,
        },
        "gap_fill_candidates": {
            "rows": len(gap_candidates),
            "by_family": gap_counts_by_family,
        },
        "exclusions": {
            "rows": len(exclusions),
            "by_anomaly_type": counter_dict(row["anomaly_type"] for row in exclusions),
        },
        "oanda_blank_reconciliation": {
            "rows": len(reconciliation),
            "by_merge_classification": counter_dict(
                row["merge_classification"] for row in reconciliation
            ),
        },
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(output_directory / NORMALIZED_FILENAME, NORMALIZED_FIELDS, evidence)
    write_csv(output_directory / GAP_FILL_FILENAME, GAP_FILL_FIELDS, gap_candidates)
    write_csv(
        output_directory / OANDA_RECONCILIATION_FILENAME,
        OANDA_RECONCILIATION_FIELDS,
        reconciliation,
    )
    write_csv(output_directory / EXCLUSIONS_FILENAME, EXCLUSION_FIELDS, exclusions)
    write_json(output_directory / SUMMARY_FILENAME, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize immutable Myfxbook consensus evidence and create file-only "
            "OANDA reconciliation candidates. No database or network is used."
        )
    )
    parser.add_argument("--raw-capture", type=Path, default=DEFAULT_RAW_CAPTURE)
    parser.add_argument("--oanda-matches", type=Path, default=DEFAULT_OANDA_MATCHES)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    summary = process_inputs(args.raw_capture, args.oanda_matches, args.output_directory)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
