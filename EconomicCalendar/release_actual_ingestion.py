#!/usr/bin/env python3
"""Fail-closed authoritative economic release-actual ingestion primitives.

Phase 9 intentionally supports only Census advance RETAIL_SALES and
DURABLE_GOODS artifacts.  The module is local-artifact-only; PostgreSQL and
command-line orchestration live in import_economic_event_release_actual.py.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
import re
import subprocess
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Callable, Iterable, Mapping, Sequence


UTC = dt.timezone.utc
SUPPORTED_FAMILIES = ("DURABLE_GOODS", "RETAIL_SALES")
SEMANTIC_CONTRACT = "census_advance_release_headline_mom_percent_v1"


def canonical_instant(value: str | dt.datetime) -> str:
    parsed = value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp_missing_timezone")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_decimal(value: Decimal | str) -> str:
    parsed = value if isinstance(value, Decimal) else Decimal(value)
    if not parsed.is_finite():
        raise ValueError("numeric_value_not_finite")
    text = format(parsed, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def month_key(reference_period: str) -> str:
    try:
        value = dt.datetime.strptime(reference_period, "%B %Y")
    except ValueError as error:
        raise ValueError("reference_period_unsupported") from error
    return value.strftime("%Y-%m")


def previous_reference_period(reference_period: str) -> str:
    value = dt.datetime.strptime(reference_period, "%B %Y")
    if value.month == 1:
        value = value.replace(year=value.year - 1, month=12)
    else:
        value = value.replace(month=value.month - 1)
    return value.strftime("%B %Y")


def _clean_text(value: str) -> str:
    translations = str.maketrans({"\u2212": "-", "\u2013": "-", "\u2014": "-", "\u00a0": " "})
    return " ".join(value.translate(translations).split())


@dataclasses.dataclass(frozen=True)
class CensusArtifact:
    event_family: str
    reference_period: str
    source_url: str
    path: pathlib.Path
    repository_path: str
    source_event_id: str
    available_at: str
    retrieved_at: str
    archive_commit: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class ReleaseActualCandidate:
    source_family: str
    candidate_event_family: str
    candidate_source_event_id: str
    candidate_reference_period: str
    source_agency: str
    source_observation_id: str
    publication_state: str
    revision_sequence: int
    available_at: str
    retrieved_at: str
    source_url: str
    source_artifact_path: str
    source_artifact_sha256: str
    semantic_contract: str
    source_provenance: Mapping[str, object]
    actual_raw: str
    actual_value_kind: str
    actual_value_low: str
    actual_value_high: str | None
    actual_canonical_value_low: str
    actual_canonical_value_high: str | None
    actual_unit: str
    actual_scale: str
    actual_qualifier: str | None

    def __post_init__(self) -> None:
        if self.publication_state == "initial":
            if self.revision_sequence != 0:
                raise ValueError("initial_revision_sequence_invalid")
        elif self.publication_state == "revision":
            if self.revision_sequence <= 0:
                raise ValueError("revision_sequence_invalid")
        else:
            raise ValueError("publication_state_invalid")
        if self.actual_value_kind != "scalar" or self.actual_value_high is not None:
            raise ValueError("census_value_shape_unsupported")
        raw = Decimal(self.actual_value_low)
        scale = Decimal(self.actual_scale)
        canonical = Decimal(self.actual_canonical_value_low)
        if scale <= 0 or raw * scale != canonical:
            raise ValueError("raw_canonical_scaling_invalid")
        if self.actual_unit != "percent" or self.actual_qualifier != "m/m":
            raise ValueError("census_semantics_invalid")
        if canonical_instant(self.retrieved_at) < canonical_instant(self.available_at):
            raise ValueError("retrieved_at_predates_available_at")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_artifact_sha256):
            raise ValueError("source_sha256_invalid")
        if not re.match(r"^https://www2?\.census\.gov/", self.source_url):
            raise ValueError("source_url_not_authoritative")

    def persisted_values(self, economic_event_id: int) -> dict[str, object]:
        return {
            "economic_event_id": economic_event_id,
            "source_agency": self.source_agency,
            "source_observation_id": self.source_observation_id,
            "publication_state": self.publication_state,
            "revision_sequence": self.revision_sequence,
            "available_at": self.available_at,
            "retrieved_at": self.retrieved_at,
            "source_url": self.source_url,
            "source_artifact_path": self.source_artifact_path,
            "source_artifact_sha256": self.source_artifact_sha256,
            "semantic_contract": self.semantic_contract,
            "source_provenance": dict(self.source_provenance),
            "actual_raw": self.actual_raw,
            "actual_value_kind": self.actual_value_kind,
            "actual_value_low": self.actual_value_low,
            "actual_value_high": self.actual_value_high,
            "actual_canonical_value_low": self.actual_canonical_value_low,
            "actual_canonical_value_high": self.actual_canonical_value_high,
            "actual_unit": self.actual_unit,
            "actual_scale": self.actual_scale,
            "actual_qualifier": self.actual_qualifier,
        }


@dataclasses.dataclass(frozen=True)
class Rejection:
    source_family: str
    artifact: str
    candidate_event_family: str | None
    candidate_reference_period: str | None
    decision: str
    rejection_reason: str


@dataclasses.dataclass(frozen=True)
class EconomicEvent:
    economic_event_id: int
    source_agency: str
    event_family: str
    event_timestamp_utc: str
    source_event_id: str
    source_url: str
    reference_period: str | None


@dataclasses.dataclass(frozen=True)
class Consensus:
    economic_event_id: int
    value_kind: str
    unit: str
    qualifier: str | None


@dataclasses.dataclass(frozen=True)
class ImportDecision:
    candidate: ReleaseActualCandidate
    decision: str
    matched_economic_event_id: int | None
    rejection_reason: str | None

    def report(self) -> dict[str, object]:
        return {
            "source_family": self.candidate.source_family,
            "artifact": self.candidate.source_artifact_path,
            "candidate_event": {
                "event_family": self.candidate.candidate_event_family,
                "source_event_id": self.candidate.candidate_source_event_id,
                "reference_period": self.candidate.candidate_reference_period,
            },
            "matched_economic_event_id": self.matched_economic_event_id,
            "publication_state": self.candidate.publication_state,
            "revision_sequence": self.candidate.revision_sequence,
            "raw_actual": self.candidate.actual_raw,
            "canonical_actual": self.candidate.actual_canonical_value_low,
            "unit": self.candidate.actual_unit,
            "qualifier": self.candidate.actual_qualifier,
            "available_at": self.candidate.available_at,
            "retrieved_at": self.candidate.retrieved_at,
            "source_provenance": dict(self.candidate.source_provenance),
            "decision": self.decision,
            "rejection_reason": self.rejection_reason,
        }


def read_artifact_text(path: pathlib.Path, pdftotext: str = "pdftotext") -> tuple[str, str]:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8"), "text_fixture_v1"
    if path.suffix.lower() != ".pdf":
        raise ValueError("unsupported_document_type")
    result = subprocess.run(
        [pdftotext, "-raw", str(path), "-"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("pdf_extraction_failed:" + " ".join(result.stderr.split()))
    if not result.stdout.strip():
        raise ValueError("pdf_extraction_empty")
    version = subprocess.run(
        [pdftotext, "-v"], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    ).stdout.splitlines()
    extractor = " ".join(version[0].split()) if version else "pdftotext_unknown"
    return result.stdout, extractor


def _signed(direction: str | None, numeric: str) -> Decimal:
    try:
        value = Decimal(numeric)
    except InvalidOperation as error:
        raise ValueError("actual_numeric_invalid") from error
    if direction and direction.lower().startswith("decreas"):
        value = -value
    return value


def _initial_actual(family: str, reference_period: str, text: str) -> tuple[Decimal, str, str]:
    normalized = _clean_text(text)
    month, year = reference_period.split()
    if "ADVANCE" not in normalized.upper():
        raise ValueError("missing_initial_provenance:advance_release_marker_missing")

    if family == "RETAIL_SALES":
        prose = re.search(
            rf"advance estimates of U\.S\. retail and food services sales for {month}.*?"
            rf"were \$[0-9,.]+ billion, (?P<raw>(?:an? )?(?P<direction>increase|decrease) "
            rf"of (?P<value>[0-9.]+) percent).*?from the previous month",
            normalized,
            re.IGNORECASE,
        )
        card = re.search(
            rf"(?P<raw>{month} {year}\s+\$[0-9,.]+ billion\s+(?P<value>[+\-]?[0-9.]+)%)",
            normalized,
            re.IGNORECASE,
        )
    elif family == "DURABLE_GOODS":
        prose = re.search(
            rf"New orders for manufactured durable goods in {month}\s+"
            rf"(?P<raw>(?P<direction>increased|decreased) \$[0-9,.]+ billion or "
            rf"(?P<value>[0-9.]+) percent)",
            normalized,
            re.IGNORECASE,
        )
        card = re.search(
            rf"(?P<raw>{month} {year}\s+\$[0-9,.]+ billion\s+(?P<value>[+\-]?[0-9.]+)%[°*]?)",
            normalized,
            re.IGNORECASE,
        )
    else:
        raise ValueError("unsupported_source_family")

    match = prose or card
    if not match:
        raise ValueError("unsupported_semantics:headline_mom_actual_not_proven")
    value = _signed(match.groupdict().get("direction"), match.group("value"))
    return value, match.group("raw"), match.group(0)


def _revision_actual(
    family: str, reference_period: str, text: str
) -> tuple[str, Decimal, str, str] | None:
    normalized = _clean_text(text)
    previous = previous_reference_period(reference_period)
    previous_month, previous_year = previous.split()

    if family == "RETAIL_SALES":
        prose = re.search(
            rf"The [A-Z][a-z]+(?: [0-9]{{4}})? to {previous_month} {previous_year} percent change "
            rf"was (?P<raw>(?:revised from .*? to|unrevised from)\s*"
            rf"(?P<value>[+\-]?[0-9.]+) percent)",
            normalized,
            re.IGNORECASE,
        )
    else:
        prose = None
    card_after = re.search(
        rf"(?P<raw>{previous_month} {previous_year}\s+\$[0-9,.]+ billion\s+"
        rf"(?P<value>[+\-]?[0-9.]+)%[°*]?\s*\(revised\))",
        normalized,
        re.IGNORECASE,
    )
    card_before = re.search(
        rf"(?P<raw>{previous_month} {previous_year}\s*\(revised\)\s+\$[0-9,.]+ billion\s+"
        rf"(?P<value>[+\-]?[0-9.]+)%[°*]?)",
        normalized,
        re.IGNORECASE,
    )
    match = prose or card_after or card_before
    if not match:
        return None
    return previous, Decimal(match.group("value")), match.group("raw"), match.group(0)


def _candidate(
    artifact: CensusArtifact,
    reference_period: str,
    source_event_id: str,
    publication_state: str,
    revision_sequence: int,
    value: Decimal,
    raw: str,
    evidence: str,
    extractor: str,
) -> ReleaseActualCandidate:
    normalized = canonical_decimal(value)
    observation = (
        f"census:{artifact.event_family.lower()}:{month_key(reference_period)}:"
        f"{publication_state}:{artifact.sha256[:20]}"
    )
    provenance = {
        "adapter": "census_advance_release_actual_v1",
        "archive_admission_commit": artifact.archive_commit,
        "artifact_sha256": artifact.sha256,
        "evidence_excerpt": evidence,
        "extractor": extractor,
        "headline_statistic": (
            "retail_and_food_services_sales" if artifact.event_family == "RETAIL_SALES"
            else "durable_goods_new_orders"
        ),
        "publication_evidence": (
            "current_period_advance_release" if publication_state == "initial"
            else "explicit_prior_period_revised_label"
        ),
        "reference_period": reference_period,
        "retrieved_at_basis": "first_git_archive_admission_committer_timestamp",
    }
    return ReleaseActualCandidate(
        source_family="CENSUS",
        candidate_event_family=artifact.event_family,
        candidate_source_event_id=source_event_id,
        candidate_reference_period=reference_period,
        source_agency="CENSUS",
        source_observation_id=observation,
        publication_state=publication_state,
        revision_sequence=revision_sequence,
        available_at=artifact.available_at,
        retrieved_at=artifact.retrieved_at,
        source_url=artifact.source_url,
        source_artifact_path=artifact.repository_path,
        source_artifact_sha256=artifact.sha256,
        semantic_contract=SEMANTIC_CONTRACT,
        source_provenance=provenance,
        actual_raw=raw,
        actual_value_kind="scalar",
        actual_value_low=normalized,
        actual_value_high=None,
        actual_canonical_value_low=normalized,
        actual_canonical_value_high=None,
        actual_unit="percent",
        actual_scale="1",
        actual_qualifier="m/m",
    )


def extract_census_candidates(
    artifact: CensusArtifact,
    source_ids_by_reference: Mapping[tuple[str, str], str],
    text_reader: Callable[[pathlib.Path], tuple[str, str]] = read_artifact_text,
) -> tuple[list[ReleaseActualCandidate], list[Rejection]]:
    try:
        text, extractor = text_reader(artifact.path)
        initial_value, initial_raw, initial_evidence = _initial_actual(
            artifact.event_family, artifact.reference_period, text
        )
        candidates = [_candidate(
            artifact, artifact.reference_period, artifact.source_event_id,
            "initial", 0, initial_value, initial_raw, initial_evidence, extractor,
        )]
        revision = _revision_actual(artifact.event_family, artifact.reference_period, text)
        if revision:
            reference, value, raw, evidence = revision
            revision_source_id = source_ids_by_reference.get((artifact.event_family, reference))
            if revision_source_id:
                candidates.append(_candidate(
                    artifact, reference, revision_source_id, "revision", 1,
                    value, raw, evidence, extractor,
                ))
        return candidates, []
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        message = str(error)
        decision = "unsupported"
        if message.startswith("missing_initial_provenance"):
            decision = "missing_initial_provenance"
        elif message.startswith("unsupported_semantics"):
            decision = "invalid_semantics"
        return [], [Rejection(
            source_family="CENSUS",
            artifact=artifact.repository_path,
            candidate_event_family=artifact.event_family,
            candidate_reference_period=artifact.reference_period,
            decision=decision,
            rejection_reason=message,
        )]


def match_candidates(
    candidates: Iterable[ReleaseActualCandidate],
    events: Sequence[EconomicEvent],
    existing_rows: Sequence[Mapping[str, object]] = (),
) -> list[ImportDecision]:
    by_source_id: dict[str, list[EconomicEvent]] = defaultdict(list)
    for event in events:
        by_source_id[event.source_event_id].append(event)

    existing_by_observation: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    existing_by_revision: dict[tuple[int, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in existing_rows:
        existing_by_observation[(str(row["source_agency"]), str(row["source_observation_id"]))].append(row)
        existing_by_revision[(int(row["economic_event_id"]), int(row["revision_sequence"]))].append(row)

    decisions: list[ImportDecision] = []
    for candidate in sorted(candidates, key=lambda item: (
        item.available_at, item.candidate_event_family,
        item.candidate_reference_period, item.revision_sequence,
        item.source_observation_id,
    )):
        matches = by_source_id.get(candidate.candidate_source_event_id, [])
        if len(matches) > 1:
            decisions.append(ImportDecision(candidate, "ambiguous", None, "duplicate_source_event_id"))
            continue
        if not matches:
            decisions.append(ImportDecision(candidate, "unmatched", None, "source_event_id_not_found"))
            continue
        event = matches[0]
        if event.source_agency != candidate.source_agency:
            decisions.append(ImportDecision(candidate, "invalid_semantics", None, "source_agency_mismatch"))
            continue
        if event.event_family != candidate.candidate_event_family or event.reference_period != candidate.candidate_reference_period:
            decisions.append(ImportDecision(candidate, "invalid_semantics", None, "canonical_event_identity_mismatch"))
            continue
        if event.source_url != candidate.source_url and candidate.publication_state == "initial":
            decisions.append(ImportDecision(candidate, "invalid_semantics", None, "canonical_source_url_mismatch"))
            continue
        if canonical_instant(candidate.available_at) < canonical_instant(event.event_timestamp_utc):
            decisions.append(ImportDecision(candidate, "invalid_semantics", None, "available_at_predates_event"))
            continue
        if (
            candidate.publication_state == "initial"
            and canonical_instant(candidate.available_at) != canonical_instant(event.event_timestamp_utc)
        ):
            decisions.append(ImportDecision(candidate, "invalid_semantics", None, "initial_available_at_event_mismatch"))
            continue

        persisted = candidate.persisted_values(event.economic_event_id)
        conflicts = (
            existing_by_observation.get((candidate.source_agency, candidate.source_observation_id), [])
            + existing_by_revision.get((event.economic_event_id, candidate.revision_sequence), [])
        )
        unique_conflicts = {int(row.get("economic_event_release_actual_id", index)): row for index, row in enumerate(conflicts)}
        if unique_conflicts:
            identical = all(_persisted_equal(persisted, row) for row in unique_conflicts.values())
            decision = "duplicate_identical" if identical else "conflict"
            reason = "identical_observation_already_persisted" if identical else "immutable_identity_conflict"
            decisions.append(ImportDecision(candidate, decision, event.economic_event_id, reason))
            continue
        decisions.append(ImportDecision(candidate, "matched", event.economic_event_id, None))
    return decisions


def _json_object(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _persisted_equal(expected: Mapping[str, object], actual: Mapping[str, object]) -> bool:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if key in {"available_at", "retrieved_at"}:
            if canonical_instant(str(actual_value)) != canonical_instant(str(expected_value)):
                return False
        elif key == "source_provenance":
            if _json_object(actual_value) != expected_value:
                return False
        elif key in {
            "actual_value_low", "actual_value_high", "actual_canonical_value_low",
            "actual_canonical_value_high", "actual_scale",
        }:
            if actual_value is None or expected_value is None:
                if actual_value is not expected_value:
                    return False
            elif Decimal(str(actual_value)) != Decimal(str(expected_value)):
                return False
        elif str(actual_value) != str(expected_value):
            return False
    return True


def deterministic_json_lines(
    decisions: Sequence[ImportDecision], rejections: Sequence[Rejection]
) -> str:
    rows = [decision.report() for decision in decisions]
    rows.extend({
        "source_family": rejection.source_family,
        "artifact": rejection.artifact,
        "candidate_event": {
            "event_family": rejection.candidate_event_family,
            "source_event_id": None,
            "reference_period": rejection.candidate_reference_period,
        },
        "matched_economic_event_id": None,
        "publication_state": None,
        "revision_sequence": None,
        "raw_actual": None,
        "canonical_actual": None,
        "unit": None,
        "qualifier": None,
        "available_at": None,
        "retrieved_at": None,
        "source_provenance": None,
        "decision": rejection.decision,
        "rejection_reason": rejection.rejection_reason,
    } for rejection in rejections)
    rows.sort(key=lambda row: (
        str(row["artifact"]),
        str(row["publication_state"]),
        int(row["revision_sequence"] or 0),
        str(row["decision"]),
    ))
    return "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)


def coverage_audit(
    events: Sequence[EconomicEvent],
    consensus: Sequence[Consensus],
    decisions: Sequence[ImportDecision],
    rejections: Sequence[Rejection],
    start: str = "2010-01-01T00:00:00Z",
    end: str = "2025-01-01T00:00:00Z",
) -> dict[str, object]:
    start_at, end_at = canonical_instant(start), canonical_instant(end)
    consensus_by_id = {row.economic_event_id: row for row in consensus}
    accepted = [row for row in decisions if row.decision in {"matched", "duplicate_identical"}]

    def in_period(event: EconomicEvent) -> bool:
        timestamp = canonical_instant(event.event_timestamp_utc)
        return start_at <= timestamp < end_at

    def summarize(family: str, period_events: list[EconomicEvent]) -> dict[str, object]:
        ids = {event.economic_event_id for event in period_events}
        family_accepted = [row for row in accepted if row.matched_economic_event_id in ids]
        initials = [row for row in family_accepted if row.candidate.publication_state == "initial"]
        revisions = [row for row in family_accepted if row.candidate.publication_state == "revision"]
        selected = [consensus_by_id[event_id] for event_id in ids if event_id in consensus_by_id]
        scalar = [row for row in selected if row.value_kind == "scalar"]
        initial_by_id = {row.matched_economic_event_id: row for row in initials}
        joint = [event_id for event_id in ids if event_id in initial_by_id and event_id in consensus_by_id]
        compatible = [
            event_id for event_id in joint
            if consensus_by_id[event_id].value_kind == "scalar"
            and consensus_by_id[event_id].unit == initial_by_id[event_id].candidate.actual_unit
            and consensus_by_id[event_id].qualifier == initial_by_id[event_id].candidate.actual_qualifier
        ]
        artifacts = {row.candidate.source_artifact_path for row in family_accepted}
        family_rejections = [row for row in rejections if row.candidate_event_family == family]
        initial_times = sorted(row.candidate.available_at for row in initials)
        total = len(period_events)
        return {
            "authoritative_events": total,
            "source_artifacts": len(artifacts),
            "matched_events": len({row.matched_economic_event_id for row in initials}),
            "certified_initial_actuals": len(initials),
            "certified_revisions": len(revisions),
            "events_lacking_initial_provenance": total - len(initials),
            "unmatched_artifacts": sum(row.decision == "unmatched" for row in decisions if row.candidate.candidate_event_family == family),
            "ambiguous_matches": sum(row.decision == "ambiguous" for row in decisions if row.candidate.candidate_event_family == family),
            "unsupported_semantic_cases": sum(row.decision in {"invalid_semantics", "unsupported"} for row in family_rejections),
            "conflicts": sum(row.decision == "conflict" for row in decisions if row.candidate.candidate_event_family == family),
            "actual_coverage_percent": round(100.0 * len(initials) / total, 4) if total else 0.0,
            "selected_consensus_rows": len(selected),
            "selected_scalar_consensus_rows": len(scalar),
            "rows_having_both": len(joint),
            "jointly_usable_surprise_rows": len(compatible),
            "usable_surprise_coverage_percent": round(100.0 * len(compatible) / total, 4) if total else 0.0,
            "missing_consensus": len(initials) - len(joint),
            "incompatible_consensus_actual_semantics": len(joint) - len(compatible),
            "earliest_certified_initial_actual": initial_times[0] if initial_times else None,
            "latest_certified_initial_actual": initial_times[-1] if initial_times else None,
        }

    families = sorted({event.event_family for event in events})
    target_events = [event for event in events if in_period(event)]
    post_events = [event for event in events if canonical_instant(event.event_timestamp_utc) >= end_at]
    target = {family: summarize(family, [event for event in target_events if event.event_family == family]) for family in families}
    post = {family: summarize(family, [event for event in post_events if event.event_family == family]) for family in families}

    by_year: dict[str, dict[str, object]] = {}
    for event in target_events:
        year = canonical_instant(event.event_timestamp_utc)[:4]
        key = f"{year}:{event.event_family}"
        if key not in by_year:
            year_events = [candidate for candidate in target_events if canonical_instant(candidate.event_timestamp_utc)[:4] == year and candidate.event_family == event.event_family]
            by_year[key] = summarize(event.event_family, year_events)

    return {
        "contract": "economic_event_release_actual_phase9_coverage_v1",
        "generated_from": "deterministic_local_artifacts_plus_read_only_catalog",
        "target_period": {"start_inclusive": start_at, "end_exclusive": end_at},
        "families": target,
        "by_year": dict(sorted(by_year.items())),
        "post_target_period": post,
    }


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_census_artifacts(
    repo_root: pathlib.Path,
    manifest_path: pathlib.Path,
    prepared_path: pathlib.Path,
    admissions: Mapping[str, tuple[str, str]],
) -> tuple[list[CensusArtifact], dict[tuple[str, str], str]]:
    with prepared_path.open(newline="", encoding="utf-8-sig") as source:
        prepared = list(csv.DictReader(source))
    prepared_by_key = {(row["event_family"], row["reference_period"]): row for row in prepared}
    source_ids = {key: row["source_event_id"] for key, row in prepared_by_key.items()}

    with manifest_path.open(newline="", encoding="utf-8-sig") as source:
        manifest = list(csv.DictReader(source))
    artifacts: list[CensusArtifact] = []
    for row in manifest:
        family = row["event_family"]
        if family not in SUPPORTED_FAMILIES:
            continue
        key = (family, row["reference_period"])
        canonical = prepared_by_key.get(key)
        if canonical is None or canonical["filename"] != row["filename"]:
            raise ValueError("census_manifest_canonical_identity_mismatch")
        subdirectory = "retail_sales" if family == "RETAIL_SALES" else "durable_goods"
        artifact_path = (manifest_path.parent / "releases" / subdirectory / row["filename"]).resolve()
        raw_root = (manifest_path.parent / "releases").resolve()
        if raw_root not in artifact_path.parents or not artifact_path.is_file():
            raise ValueError("census_artifact_path_invalid")
        repository_path = artifact_path.relative_to(repo_root.resolve()).as_posix()
        admission = admissions.get(repository_path)
        if admission is None:
            raise ValueError("census_artifact_archive_admission_missing:" + repository_path)
        commit, retrieved_at = admission
        artifacts.append(CensusArtifact(
            event_family=family,
            reference_period=row["reference_period"],
            source_url=row["url"],
            path=artifact_path,
            repository_path=repository_path,
            source_event_id=canonical["source_event_id"],
            available_at=canonical_instant(canonical["event_timestamp_utc"]),
            retrieved_at=canonical_instant(retrieved_at),
            archive_commit=commit,
            sha256=sha256_file(artifact_path),
        ))
    artifacts.sort(key=lambda row: (row.available_at, row.event_family, row.reference_period))
    return artifacts, source_ids
