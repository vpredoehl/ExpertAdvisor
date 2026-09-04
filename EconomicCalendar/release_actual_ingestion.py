#!/usr/bin/env python3
"""Fail-closed authoritative economic release-actual ingestion primitives.

Phase 10 extends the proven Phase 9 Census path with source-specific BEA GDP
and PCE adapters.  The module remains local-artifact-only; PostgreSQL and
command-line orchestration live in import_economic_event_release_actual.py.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import html
import json
import pathlib
import re
import subprocess
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


UTC = dt.timezone.utc
CENSUS_SUPPORTED_FAMILIES = ("DURABLE_GOODS", "RETAIL_SALES")
BEA_SUPPORTED_FAMILIES = ("GDP", "PCE")
BLS_SUPPORTED_FAMILIES = ("CPI", "EMPLOYMENT", "PPI", "JOLTS")
SUPPORTED_FAMILIES = CENSUS_SUPPORTED_FAMILIES
SEMANTIC_CONTRACT = "census_advance_release_headline_mom_percent_v1"
BEA_GDP_SEMANTIC_CONTRACT = "bea_real_gdp_annualized_quarterly_percent_v1"
BEA_PCE_SEMANTIC_CONTRACT = "bea_current_dollar_pce_mom_percent_v1"
BLS_CPI_SEMANTIC_CONTRACT = "bls_cpi_u_all_items_sa_mom_percent_v1"
BLS_EMPLOYMENT_SEMANTIC_CONTRACT = "bls_total_nonfarm_payroll_sa_change_v1"
BLS_PPI_FINISHED_GOODS_SEMANTIC_CONTRACT = "bls_ppi_finished_goods_sa_mom_percent_v1"
BLS_PPI_FINAL_DEMAND_SEMANTIC_CONTRACT = "bls_ppi_final_demand_sa_mom_percent_v1"
BLS_JOLTS_SEMANTIC_CONTRACT = "bls_jolts_total_nonfarm_job_openings_sa_level_v1"


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
class BeaArtifact:
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
    title: str


@dataclasses.dataclass(frozen=True)
class BlsArtifact:
    event_family: str
    reference_period: str
    release_date: str
    source_url: str
    canonical_source_url: str
    path: pathlib.Path
    repository_path: str
    source_event_id: str
    available_at: str
    retrieved_at: str
    first_archive_admission_at: str
    sha256: str
    source_release_identity: str
    release_timestamp_evidence: str


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
            raise ValueError("actual_value_shape_unsupported")
        raw = Decimal(self.actual_value_low)
        scale = Decimal(self.actual_scale)
        canonical = Decimal(self.actual_canonical_value_low)
        if scale <= 0 or raw * scale != canonical:
            raise ValueError("raw_canonical_scaling_invalid")
        expected_semantics = {
            SEMANTIC_CONTRACT: ("CENSUS", "percent", "m/m"),
            BEA_GDP_SEMANTIC_CONTRACT: ("BEA", "percent", None),
            BEA_PCE_SEMANTIC_CONTRACT: ("BEA", "percent", "m/m"),
            BLS_CPI_SEMANTIC_CONTRACT: ("BLS", "percent", "m/m"),
            BLS_EMPLOYMENT_SEMANTIC_CONTRACT: ("BLS", "count", None),
            BLS_PPI_FINISHED_GOODS_SEMANTIC_CONTRACT: ("BLS", "percent", "m/m"),
            BLS_PPI_FINAL_DEMAND_SEMANTIC_CONTRACT: ("BLS", "percent", "m/m"),
            BLS_JOLTS_SEMANTIC_CONTRACT: ("BLS", "count", None),
        }
        expected = expected_semantics.get(self.semantic_contract)
        if expected is None:
            raise ValueError("semantic_contract_unsupported")
        if (self.source_agency, self.actual_unit, self.actual_qualifier) != expected:
            error = (
                "census_semantics_invalid"
                if self.semantic_contract == SEMANTIC_CONTRACT
                else "bls_semantics_invalid"
                if self.source_agency == "BLS"
                else "bea_semantics_invalid"
            )
            raise ValueError(error)
        if canonical_instant(self.retrieved_at) < canonical_instant(self.available_at):
            raise ValueError("retrieved_at_predates_available_at")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_artifact_sha256):
            raise ValueError("source_sha256_invalid")
        authoritative_url = (
            re.match(r"^https://www2?\.census\.gov/", self.source_url)
            if self.source_agency == "CENSUS"
            else re.match(r"^https://www\.bea\.gov/", self.source_url)
            if self.source_agency == "BEA"
            else re.match(r"^https://www\.bls\.gov/news\.release/archives/", self.source_url)
            if self.source_agency == "BLS"
            else None
        )
        if not authoritative_url:
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

    def provenance_observation_values(
        self, economic_event_id: int
    ) -> dict[str, object]:
        """Return the migration-090 append-only evidence representation.

        ``available_at`` has already passed the source-specific direct
        publication checks in this module.  It is therefore source
        publication evidence here, while ``retrieved_at`` remains the first
        system observation time.  Database ingestion time is intentionally
        left to its independent column default.
        """
        return {
            "economic_event_id": economic_event_id,
            "source_name": self.source_agency,
            "source_role": "authoritative",
            "source_native_event_id": self.candidate_source_event_id,
            "source_observation_id": self.source_observation_id,
            "evidence_key": (
                "artifact-sha256:" + self.source_artifact_sha256
                + ":source-observation:" + self.source_observation_id
            ),
            "observation_kind": self.publication_state,
            "revision_sequence": self.revision_sequence,
            "source_publication_at": self.available_at,
            "source_publication_time_status": "exact",
            "observed_at": self.retrieved_at,
            "availability_proof": "source_publication",
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


def read_bea_artifact_text(path: pathlib.Path) -> tuple[str, str]:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8"), "text_fixture_v1"
    if path.suffix.lower() not in {".htm", ".html"}:
        raise ValueError("unsupported_document_type")
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("html_extraction_empty")
    return normalized, "python_html_text_v1"


def _signed(direction: str | None, numeric: str) -> Decimal:
    try:
        value = Decimal(numeric)
    except InvalidOperation as error:
        raise ValueError("actual_numeric_invalid") from error
    if direction and re.search(r"\b(?:decreas|declin|fell|down)\w*\b", direction.lower()):
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


def _bea_estimate_identity(reference_period: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"(Q[1-4] [0-9]{4}) (Advance|Initial|Second|Third|Updated)",
        reference_period,
    )
    if not match:
        raise ValueError("reference_period_unsupported")
    return match.group(1), match.group(2).lower()


def _bea_gdp_actual(
    reference_period: str, text: str
) -> tuple[Decimal, str, str, str]:
    quarter, estimate = _bea_estimate_identity(reference_period)
    expected_marker = {
        "advance": "advance",
        "initial": "initial",
        "second": "second",
        "third": "third",
        "updated": "updated",
    }[estimate]
    normalized = _clean_text(text)
    pattern = re.compile(
        r"Real gross domestic product(?: \(GDP\))?.{0,260}?"
        r"(?P<direction>increased|decreased) at an annual rate of "
        r"(?P<value>[0-9]+(?:\.[0-9]+)?) percent.{0,260}?"
        rf"(?:the [\"']?{expected_marker}[\"']? estimate|according to the "
        rf"[\"']?{expected_marker}[\"']? estimate)",
        re.IGNORECASE,
    )
    match = pattern.search(normalized)
    if not match:
        raise ValueError("unsupported_semantics:real_gdp_annualized_headline_not_proven")
    value = _signed(match.group("direction"), match.group("value"))
    raw = f"{match.group('direction')} at an annual rate of {match.group('value')} percent"
    return value, raw, match.group(0), quarter


def _bea_pce_actual(reference_period: str, text: str) -> tuple[Decimal, str, str]:
    try:
        dt.datetime.strptime(reference_period, "%B %Y")
    except ValueError as error:
        raise ValueError("unsupported_semantics:pce_single_month_reference_not_proven") from error
    normalized = _clean_text(text)
    pattern = re.compile(
        r"personal consumption expenditures(?: \(PCE\))? "
        r"(?P<direction>increased|decreased) "
        r"\$[0-9,.]+ (?:billion|trillion)(?:,? or)?\s*"
        r"(?:\((?P<paren>[+\-]?[0-9]+(?:\.[0-9]+)?) percent\)|"
        r"(?P<plain>[+\-]?[0-9]+(?:\.[0-9]+)?) percent)",
        re.IGNORECASE,
    )
    match = pattern.search(normalized)
    if not match:
        if re.search(
            r"personal consumption expenditures \(PCE\).{0,120}?less than 0\.1 percent",
            normalized,
            re.IGNORECASE,
        ):
            raise ValueError("unsupported_semantics:pce_less_than_scalar_not_exact")
        raise ValueError("unsupported_semantics:current_dollar_pce_mom_headline_not_proven")
    numeric = match.group("paren") or match.group("plain")
    direction = match.group("direction")
    if numeric.startswith("-") and not direction.lower().startswith("decreas"):
        raise ValueError("unsupported_semantics:pce_direction_sign_conflict")
    if numeric.startswith("+") and direction.lower().startswith("decreas"):
        raise ValueError("unsupported_semantics:pce_direction_sign_conflict")
    value = _signed(direction, numeric.lstrip("+-"))
    return value, match.group(0), match.group(0)


def _bea_candidate(
    artifact: BeaArtifact,
    candidate_reference_period: str,
    candidate_source_event_id: str,
    publication_state: str,
    revision_sequence: int,
    value: Decimal,
    raw: str,
    evidence: str,
    extractor: str,
    statistic: str,
) -> ReleaseActualCandidate:
    normalized = canonical_decimal(value)
    contract = (
        BEA_GDP_SEMANTIC_CONTRACT
        if artifact.event_family == "GDP"
        else BEA_PCE_SEMANTIC_CONTRACT
    )
    qualifier = None if artifact.event_family == "GDP" else "m/m"
    period_key = candidate_reference_period.lower().replace(" ", "-")
    observation = (
        f"bea:{artifact.event_family.lower()}:{period_key}:"
        f"{publication_state}:{revision_sequence}:{artifact.sha256[:20]}"
    )
    provenance = {
        "adapter": "bea_release_actual_v1",
        "archive_admission_commit": artifact.archive_commit,
        "artifact_sha256": artifact.sha256,
        "evidence_excerpt": evidence,
        "extractor": extractor,
        "headline_statistic": statistic,
        "publication_evidence": (
            "advance_or_initial_estimate_headline"
            if publication_state == "initial"
            else "later_estimate_headline_for_same_quarter"
        ),
        "reference_period": candidate_reference_period,
        "source_release_reference_period": artifact.reference_period,
        "retrieved_at_basis": "first_git_archive_admission_committer_timestamp",
    }
    return ReleaseActualCandidate(
        source_family="BEA",
        candidate_event_family=artifact.event_family,
        candidate_source_event_id=candidate_source_event_id,
        candidate_reference_period=candidate_reference_period,
        source_agency="BEA",
        source_observation_id=observation,
        publication_state=publication_state,
        revision_sequence=revision_sequence,
        available_at=artifact.available_at,
        retrieved_at=artifact.retrieved_at,
        source_url=artifact.source_url,
        source_artifact_path=artifact.repository_path,
        source_artifact_sha256=artifact.sha256,
        semantic_contract=contract,
        source_provenance=provenance,
        actual_raw=raw,
        actual_value_kind="scalar",
        actual_value_low=normalized,
        actual_value_high=None,
        actual_canonical_value_low=normalized,
        actual_canonical_value_high=None,
        actual_unit="percent",
        actual_scale="1",
        actual_qualifier=qualifier,
    )


def extract_bea_candidates(
    artifact: BeaArtifact,
    gdp_initial_by_quarter: Mapping[str, tuple[str, str]],
    text_reader: Callable[[pathlib.Path], tuple[str, str]] = read_bea_artifact_text,
) -> tuple[list[ReleaseActualCandidate], list[Rejection]]:
    try:
        text, extractor = text_reader(artifact.path)
        if artifact.event_family == "PCE":
            value, raw, evidence = _bea_pce_actual(artifact.reference_period, text)
            return [_bea_candidate(
                artifact,
                artifact.reference_period,
                artifact.source_event_id,
                "initial",
                0,
                value,
                raw,
                evidence,
                extractor,
                "current_dollar_personal_consumption_expenditures",
            )], []

        value, raw, evidence, quarter = _bea_gdp_actual(
            artifact.reference_period, text
        )
        _, estimate = _bea_estimate_identity(artifact.reference_period)
        if estimate in {"advance", "initial"}:
            target_reference = artifact.reference_period
            target_source_id = artifact.source_event_id
            state, sequence = "initial", 0
        else:
            target = gdp_initial_by_quarter.get(quarter)
            if target is None:
                raise ValueError("missing_initial_provenance:gdp_advance_event_missing")
            target_reference, target_source_id = target
            state = "revision"
            sequence = {"second": 1, "updated": 1, "third": 2}[estimate]
        return [_bea_candidate(
            artifact,
            target_reference,
            target_source_id,
            state,
            sequence,
            value,
            raw,
            evidence,
            extractor,
            "real_gdp_annualized_quarter_over_quarter",
        )], []
    except (OSError, ValueError) as error:
        message = str(error)
        decision = "unsupported"
        if message.startswith("missing_initial_provenance"):
            decision = "missing_initial_provenance"
        elif message.startswith("unsupported_semantics"):
            decision = "invalid_semantics"
        return [], [Rejection(
            source_family="BEA",
            artifact=artifact.repository_path,
            candidate_event_family=artifact.event_family,
            candidate_reference_period=artifact.reference_period,
            decision=decision,
            rejection_reason=message,
        )]


def _bls_release_identity(artifact: BlsArtifact, text: str) -> str:
    normalized = _clean_text(text)
    release_date = dt.date.fromisoformat(artifact.release_date)
    date_text = f"{release_date.strftime('%B')} {release_date.day}, {release_date.year}"
    expected_time = "10:00 a.m." if artifact.event_family == "JOLTS" else "8:30 a.m."
    evidence = artifact.release_timestamp_evidence
    identity_text = lambda value: " ".join(re.findall(r"[a-z0-9:]+", value.lower()))
    if (
        identity_text(expected_time) not in identity_text(evidence)
        or identity_text(date_text) not in identity_text(evidence)
    ):
        raise ValueError("missing_initial_provenance:manifest_release_timestamp_not_proven")
    release_prefix = normalized[:20000]
    if (
        identity_text(expected_time) not in identity_text(release_prefix)
        or identity_text(date_text) not in identity_text(release_prefix)
    ):
        raise ValueError("missing_initial_provenance:artifact_release_timestamp_not_proven")
    if artifact.reference_period.lower() not in release_prefix.lower():
        raise ValueError("missing_initial_provenance:artifact_reference_period_not_proven")
    return f"{expected_time} ET {date_text}"


def _unique_bls_value(
    matches: Sequence[tuple[Decimal, str, str]], rejection: str
) -> tuple[Decimal, str, str]:
    if not matches:
        raise ValueError("unsupported_semantics:" + rejection + "_not_proven")
    values = {value for value, _, _ in matches}
    if len(values) != 1:
        raise ValueError("unsupported_semantics:" + rejection + "_ambiguous")
    return matches[0]


def _bls_cpi_actual(reference_period: str, text: str) -> tuple[Decimal, str, str]:
    normalized = _clean_text(text)
    headline = re.search(
        r"(?P<headline>(?:On a seasonally adjusted basis, )?"
        r"(?:The )?(?:[A-Z][a-z]+ )?Consumer Price Index for All Urban Consumers \(CPI-U\) "
        r".{0,320}?the U\.S\. Bureau of Labor Statistics reported today\.)",
        normalized,
        re.IGNORECASE,
    )
    if not headline or "seasonally adjusted basis" not in headline.group("headline").lower():
        raise ValueError("unsupported_semantics:cpi_u_all_items_sa_mom_headline_not_proven")
    movement = re.compile(
        r"(?P<direction>increased|rose|declined|decreased|fell) "
        r"(?P<value>[0-9]+(?:\.[0-9]+)?) percent",
        re.IGNORECASE,
    )
    unchanged = re.compile(r"was unchanged", re.IGNORECASE)
    matches: list[tuple[Decimal, str, str]] = []
    for match in movement.finditer(headline.group("headline")):
        value = _signed(match.group("direction"), match.group("value"))
        matches.append((value, match.group(0), headline.group("headline")))
    for match in unchanged.finditer(headline.group("headline")):
        matches.append((Decimal("0"), match.group(0), headline.group("headline")))
    return _unique_bls_value(matches, "cpi_u_all_items_sa_mom_headline")


def _bls_employment_actual(
    reference_period: str, text: str
) -> tuple[Decimal, str, str]:
    month, year = map(re.escape, reference_period.split())
    normalized = _clean_text(text)
    if re.search(
        r"BLS reissued this news release",
        normalized[:30000],
        re.IGNORECASE,
    ):
        raise ValueError("missing_initial_provenance:employment_release_reissued")
    release = re.search(
        rf"THE EMPLOYMENT SITUATION\s*-+\s*{month} {year} "
        r"(?P<headline>.{0,6000}?)(?:Household Survey Data|"
        r"This news release presents statistics from two monthly surveys)",
        normalized,
        re.IGNORECASE,
    )
    if not release:
        raise ValueError("unsupported_semantics:total_nonfarm_payroll_current_month_not_proven")
    headline = release.group("headline")
    movement = re.compile(
        r"(?:Total )?[Nn]onfarm payroll employment "
        r"(?P<direction>rose|increased|declined|decreased|fell|grew|"
        r"edged up|edged down) "
        r"(?:by )?(?P<value>[0-9][0-9,.]*)"
        r"(?: (?P<unit>million|thousand))?\b",
        re.IGNORECASE,
    )
    matches: list[tuple[Decimal, str, str]] = []
    for match in movement.finditer(headline):
        value = Decimal(match.group("value").replace(",", ""))
        unit = (match.group("unit") or "count").lower()
        if unit == "million":
            value *= Decimal("1000")
        elif unit == "count":
            value /= Decimal("1000")
        value = _signed(match.group("direction"), canonical_decimal(value))
        matches.append((value, match.group(0), release.group(0)))

    parenthetical = re.compile(
        r"(?:Total )?[Nn]onfarm payroll employment.{0,120}?"
        r"\((?P<signed>[+\-]?[0-9,]+)\)",
        re.IGNORECASE,
    )
    for match in parenthetical.finditer(headline):
        value = Decimal(match.group("signed").replace(",", "")) / Decimal("1000")
        matches.append((value, match.group(0), release.group(0)))
    return _unique_bls_value(matches, "total_nonfarm_payroll_current_month")


def _bls_ppi_actual(
    artifact: BlsArtifact, text: str
) -> tuple[Decimal, str, str, str]:
    month = re.escape(artifact.reference_period.split()[0])
    normalized = _clean_text(text)
    headline = re.search(
        r"(?P<headline>The Producer Price Index for "
        r"(?P<statistic>final demand|finished goods) .{0,320}?"
        r"the U\.S\. Bureau of Labor Statistics reported today\.)",
        normalized,
        re.IGNORECASE,
    )
    if not headline or "seasonally adjusted" not in headline.group("headline").lower():
        raise ValueError("unsupported_semantics:ppi_headline_sa_mom_not_proven")
    movement = re.compile(
        r"(?P<direction>rose|advanced|increased|fell|declined|decreased|"
        r"moved up|moved down|edged up|edged down|inched up|inched down) "
        r"(?P<value>[0-9]+(?:\.[0-9]+)?) percent",
        re.IGNORECASE,
    )
    unchanged = re.compile(r"was unchanged", re.IGNORECASE)
    matches: list[tuple[Decimal, str, str, str]] = []
    for match in movement.finditer(headline.group("headline")):
        value = _signed(match.group("direction"), match.group("value"))
        matches.append((value, match.group(0), headline.group("headline"), headline.group("statistic").lower()))
    for match in unchanged.finditer(headline.group("headline")):
        matches.append((Decimal("0"), match.group(0), headline.group("headline"), headline.group("statistic").lower()))
    if not matches:
        raise ValueError("unsupported_semantics:ppi_headline_sa_mom_not_proven")
    identities = {(row[0], row[3]) for row in matches}
    if len(identities) != 1:
        raise ValueError("unsupported_semantics:ppi_headline_sa_mom_ambiguous")
    value, raw, evidence, statistic = matches[0]
    expected = "final demand" if artifact.release_date >= "2014-02-19" else "finished goods"
    if statistic != expected:
        raise ValueError("unsupported_semantics:ppi_historical_regime_mismatch")
    contract = (
        BLS_PPI_FINAL_DEMAND_SEMANTIC_CONTRACT
        if statistic == "final demand"
        else BLS_PPI_FINISHED_GOODS_SEMANTIC_CONTRACT
    )
    return value, raw, evidence, contract


def _bls_jolts_actual(
    reference_period: str, text: str
) -> tuple[Decimal, str, str]:
    normalized = _clean_text(text)
    month, year = reference_period.split()
    headline = re.search(
        rf"(?:There were (?P<old_value>[0-9]+(?:\.[0-9]+)?) million job openings "
        rf"on the last business day of {re.escape(month)}(?: {year})?|"
        r"The number of job openings.{0,100}?"
        r"(?P<new_value>[0-9]+(?:\.[0-9]+)?) million "
        rf"(?:.{{0,50}}?on the last business day of|in) {re.escape(month)})"
        r".{0,100}?the U\.S\. Bureau of Labor Statistics reported today",
        normalized,
        re.IGNORECASE,
    )
    table = re.search(
        r"Table A\. Job openings, hires, and total separations by industry, "
        r"seasonally adjusted.{0,1800}?LEVELS?\s*(?:BY INDUSTRY)?\s*"
        r"[^a-z0-9]{0,300}\(in thousands\).{0,500}?"
        r"Total(?:\([a-z0-9]+\))?(?: nonfarm)?"
        r"(?:\.{2,}\|)?\s*"
        r"(?P<prior_year>[0-9,]+)(?:\s*\|)?\s*(?P<prior_month>[0-9,]+)"
        r"(?:\s*\|)?\s*"
        r"(?P<current>[0-9,]+)",
        normalized,
        re.IGNORECASE,
    )
    if not table:
        raise ValueError("unsupported_semantics:jolts_table_a_current_level_not_proven")
    table_context = table.group(0)
    if month[:3].lower() not in table_context.lower() or year not in table_context:
        raise ValueError("unsupported_semantics:jolts_table_a_reference_period_mismatch")
    current_thousands = Decimal(table.group("current").replace(",", ""))
    if headline:
        headline_millions = Decimal(headline.group("old_value") or headline.group("new_value"))
        if (current_thousands / Decimal("1000")).quantize(Decimal("0.1")) != headline_millions:
            raise ValueError("unsupported_semantics:jolts_headline_table_value_mismatch")
    raw = table.group("current") + " thousand job openings"
    evidence = ((headline.group(0) + " | ") if headline else "") + table.group(0)
    return current_thousands, raw, evidence


def _bls_candidate(
    artifact: BlsArtifact,
    value: Decimal,
    raw: str,
    evidence: str,
    contract: str,
    statistic: str,
    release_identity_evidence: str,
) -> ReleaseActualCandidate:
    scale = Decimal("1000") if artifact.event_family in {"EMPLOYMENT", "JOLTS"} else Decimal("1")
    normalized = canonical_decimal(value)
    canonical = canonical_decimal(value * scale)
    period_key = artifact.reference_period.lower().replace(" ", "-")
    observation = (
        f"bls:{artifact.event_family.lower()}:{period_key}:initial:0:"
        f"{artifact.sha256[:20]}"
    )
    provenance = {
        "adapter": "bls_archived_release_actual_v1",
        "artifact_sha256": artifact.sha256,
        "canonical_event_source_url": artifact.canonical_source_url,
        "evidence_excerpt": evidence,
        "first_archive_admission_at": artifact.first_archive_admission_at,
        "headline_statistic": statistic,
        "publication_evidence": "official_bls_archived_release_current_reference_period",
        "reference_period": artifact.reference_period,
        "release_identity_evidence": release_identity_evidence,
        "source_release_identity": artifact.source_release_identity,
        "retrieved_at_basis": "browser_capture_manifest_first_archive_admission",
    }
    return ReleaseActualCandidate(
        source_family="BLS",
        candidate_event_family=artifact.event_family,
        candidate_source_event_id=artifact.source_event_id,
        candidate_reference_period=artifact.reference_period,
        source_agency="BLS",
        source_observation_id=observation,
        publication_state="initial",
        revision_sequence=0,
        available_at=artifact.available_at,
        retrieved_at=artifact.retrieved_at,
        source_url=artifact.source_url,
        source_artifact_path=artifact.repository_path,
        source_artifact_sha256=artifact.sha256,
        semantic_contract=contract,
        source_provenance=provenance,
        actual_raw=raw,
        actual_value_kind="scalar",
        actual_value_low=normalized,
        actual_value_high=None,
        actual_canonical_value_low=canonical,
        actual_canonical_value_high=None,
        actual_unit="count" if artifact.event_family in {"EMPLOYMENT", "JOLTS"} else "percent",
        actual_scale=canonical_decimal(scale),
        actual_qualifier=None if artifact.event_family in {"EMPLOYMENT", "JOLTS"} else "m/m",
    )


def extract_bls_candidates(
    artifact: BlsArtifact,
    text_reader: Callable[[pathlib.Path], tuple[str, str]] = read_bea_artifact_text,
) -> tuple[list[ReleaseActualCandidate], list[Rejection]]:
    try:
        text, _ = text_reader(artifact.path)
        release_identity = _bls_release_identity(artifact, text)
        if artifact.event_family == "CPI":
            value, raw, evidence = _bls_cpi_actual(artifact.reference_period, text)
            contract = BLS_CPI_SEMANTIC_CONTRACT
            statistic = "cpi_u_all_items_seasonally_adjusted_month_over_month"
        elif artifact.event_family == "EMPLOYMENT":
            value, raw, evidence = _bls_employment_actual(artifact.reference_period, text)
            contract = BLS_EMPLOYMENT_SEMANTIC_CONTRACT
            statistic = "total_nonfarm_payroll_employment_seasonally_adjusted_change"
        elif artifact.event_family == "PPI":
            value, raw, evidence, contract = _bls_ppi_actual(artifact, text)
            statistic = (
                "ppi_final_demand_seasonally_adjusted_month_over_month"
                if contract == BLS_PPI_FINAL_DEMAND_SEMANTIC_CONTRACT
                else "ppi_finished_goods_seasonally_adjusted_month_over_month"
            )
        elif artifact.event_family == "JOLTS":
            value, raw, evidence = _bls_jolts_actual(artifact.reference_period, text)
            contract = BLS_JOLTS_SEMANTIC_CONTRACT
            statistic = "jolts_total_nonfarm_job_openings_seasonally_adjusted_level"
        else:
            raise ValueError("unsupported_semantics:bls_family_not_supported")
        return [
            _bls_candidate(
                artifact, value, raw, evidence, contract, statistic, release_identity
            )
        ], []
    except (OSError, ValueError) as error:
        message = str(error)
        decision = "unsupported"
        if message.startswith("missing_initial_provenance"):
            decision = "missing_initial_provenance"
        elif message.startswith("unsupported_semantics"):
            decision = "invalid_semantics"
        return [], [Rejection(
            source_family="BLS",
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
            bls_archive_identity = (
                candidate.source_agency == "BLS"
                and re.match(
                    r"^https://www\.bls\.gov/news\.release/archives/",
                    candidate.source_url,
                )
                and candidate.source_provenance.get("canonical_event_source_url")
                == event.source_url
            )
            if not bls_archive_identity:
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
        "contract": "economic_event_release_actual_phase10_coverage_v1",
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


def load_bea_artifacts(
    repo_root: pathlib.Path,
    canonical_path: pathlib.Path,
    prepared_path: pathlib.Path,
    admissions: Mapping[str, tuple[str, str]],
) -> tuple[list[BeaArtifact], dict[str, tuple[str, str]]]:
    with prepared_path.open(newline="", encoding="utf-8-sig") as source:
        prepared = list(csv.DictReader(source))
    prepared_by_key = {
        (row["event_family"], canonical_instant(row["event_timestamp_utc"])): row
        for row in prepared
    }

    with canonical_path.open(newline="", encoding="utf-8-sig") as source:
        canonical = list(csv.DictReader(source))
    artifacts: list[BeaArtifact] = []
    initial_by_quarter: dict[str, tuple[str, str]] = {}
    for row in canonical:
        family = row["event_family"]
        if family not in BEA_SUPPORTED_FAMILIES:
            continue
        key = (family, canonical_instant(row["event_timestamp_utc"]))
        prepared_row = prepared_by_key.get(key)
        if prepared_row is None or prepared_row["source_url"] != row["url"]:
            raise ValueError("bea_canonical_identity_mismatch")
        release_directory = (
            canonical_path.parent / "releases"
            if row["source_set"] == "base"
            else canonical_path.parent / "recovery_v2" / "releases"
        )
        artifact_path = (release_directory / row["filename"]).resolve()
        if release_directory.resolve() not in artifact_path.parents or not artifact_path.is_file():
            raise ValueError("bea_artifact_path_invalid")
        repository_path = artifact_path.relative_to(repo_root.resolve()).as_posix()
        admission = admissions.get(repository_path)
        if admission is None:
            raise ValueError("bea_artifact_archive_admission_missing:" + repository_path)
        commit, retrieved_at = admission
        artifact = BeaArtifact(
            event_family=family,
            reference_period=prepared_row["reference_period"],
            source_url=row["url"],
            path=artifact_path,
            repository_path=repository_path,
            source_event_id=prepared_row["source_event_id"],
            available_at=canonical_instant(prepared_row["event_timestamp_utc"]),
            retrieved_at=canonical_instant(retrieved_at),
            archive_commit=commit,
            sha256=sha256_file(artifact_path),
            title=row["title"],
        )
        artifacts.append(artifact)
        if family == "GDP":
            try:
                quarter, estimate = _bea_estimate_identity(artifact.reference_period)
            except ValueError:
                continue
            if estimate in {"advance", "initial"}:
                if quarter in initial_by_quarter:
                    raise ValueError("bea_gdp_initial_identity_ambiguous:" + quarter)
                initial_by_quarter[quarter] = (
                    artifact.reference_period,
                    artifact.source_event_id,
                )
    artifacts.sort(key=lambda row: (row.available_at, row.event_family, row.reference_period))
    return artifacts, initial_by_quarter


def load_bls_artifacts(
    repo_root: pathlib.Path,
    manifest_path: pathlib.Path,
    events: Sequence[EconomicEvent],
) -> list[BlsArtifact]:
    event_by_identity: dict[tuple[str, str, str], list[EconomicEvent]] = defaultdict(list)
    eastern = ZoneInfo("America/New_York")
    for event in events:
        if event.source_agency != "BLS" or event.event_family not in BLS_SUPPORTED_FAMILIES:
            continue
        instant = dt.datetime.fromisoformat(event.event_timestamp_utc.replace("Z", "+00:00"))
        release_date = instant.astimezone(eastern).date().isoformat()
        event_by_identity[(event.event_family, release_date, event.reference_period or "")].append(event)

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    artifacts: list[BlsArtifact] = []
    seen_urls: set[str] = set()
    raw_root = (manifest_path.parent / "releases").resolve()
    for row in rows:
        if row.get("artifact_kind") != "release":
            continue
        family = str(row.get("bls_family", ""))
        if family not in BLS_SUPPORTED_FAMILIES:
            raise ValueError("bls_manifest_family_invalid")
        source_url = str(row.get("source_url", ""))
        if source_url in seen_urls:
            raise ValueError("bls_manifest_source_url_duplicate")
        seen_urls.add(source_url)
        if not re.match(
            rf"^https://www\.bls\.gov/news\.release/archives/"
            rf"(?:cpi|empsit|ppi|jolts)_\d{{8}}\.htm$",
            source_url,
        ):
            raise ValueError("bls_manifest_source_url_invalid")
        release_date = str(row.get("release_date", ""))
        reference_period = str(row.get("reference_period", ""))
        try:
            dt.date.fromisoformat(release_date)
            dt.datetime.strptime(reference_period, "%B %Y")
        except ValueError as error:
            raise ValueError("bls_manifest_release_identity_invalid") from error
        matches = event_by_identity.get((family, release_date, reference_period), [])
        if len(matches) != 1:
            reason = "missing" if not matches else "ambiguous"
            raise ValueError(f"bls_canonical_event_identity_{reason}:{family}:{release_date}")
        event = matches[0]
        repository_path = str(row.get("immutable_local_path", ""))
        artifact_path = (repo_root / repository_path).resolve()
        if raw_root not in artifact_path.parents or not artifact_path.is_file():
            raise ValueError("bls_artifact_path_invalid:" + repository_path)
        digest = sha256_file(artifact_path)
        if digest != row.get("sha256"):
            raise ValueError("bls_artifact_hash_mismatch:" + repository_path)
        retrieved_at = canonical_instant(str(row.get("retrieved_at", "")))
        first_admission = canonical_instant(str(row.get("first_archive_admission_at", "")))
        if first_admission > retrieved_at:
            raise ValueError("bls_first_archive_admission_after_retrieval")
        timestamp_evidence = row.get("release_timestamp_evidence")
        if not isinstance(timestamp_evidence, str) or not timestamp_evidence:
            raise ValueError("bls_release_timestamp_evidence_missing")
        artifacts.append(BlsArtifact(
            event_family=family,
            reference_period=reference_period,
            release_date=release_date,
            source_url=source_url,
            canonical_source_url=event.source_url,
            path=artifact_path,
            repository_path=repository_path,
            source_event_id=event.source_event_id,
            available_at=canonical_instant(event.event_timestamp_utc),
            retrieved_at=retrieved_at,
            first_archive_admission_at=first_admission,
            sha256=digest,
            source_release_identity=str(row.get("source_release_identity", "")),
            release_timestamp_evidence=timestamp_evidence,
        ))
    artifacts.sort(key=lambda row: (row.available_at, row.event_family, row.reference_period))
    return artifacts
