#!/usr/bin/env python3
"""DOL/ETA Weekly Claims first-release actual evidence adapter.

The adapter is local-artifact-only.  It verifies acquisition-manifest hashes,
extracts the embedded embargo publication boundary and the seasonally adjusted
advance initial-claims level, and maps only exact DOL/ETA catalog identities.
The separately labelled prior-week revised level is retained as revision 1;
it is never substituted for the current initial value.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import html
import json
import pathlib
import re
from collections import defaultdict
from decimal import Decimal
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from release_actual_ingestion import (
    DOL_ETA_WEEKLY_CLAIMS_SEMANTIC_CONTRACT,
    EconomicEvent,
    Rejection,
    ReleaseActualCandidate,
    canonical_instant,
)


PARSER_VERSION = "dol_eta_weekly_claims_actual_v1"
UTC = dt.timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
CANONICAL_URL = re.compile(
    r"^https://oui\.doleta\.gov/press/(?P<year>[0-9]{4})/"
    r"(?P<date>[0-9]{6})\.(?P<extension>asp|pdf)$",
    re.IGNORECASE,
)
KNOWN_FILENAME_YEAR_EXCEPTION = (
    "https://oui.doleta.gov/press/2019/010318.pdf"
)
MONTHS = {
    name.lower(): number
    for number, name in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November",
            "December",
        ),
        start=1,
    )
}


@dataclasses.dataclass(frozen=True)
class DolEtaArtifact:
    release_date: str
    source_url: str
    source_artifact_identity: str
    source_path: pathlib.Path
    source_repository_path: str
    source_sha256: str
    parser_path: pathlib.Path
    parser_repository_path: str
    parser_sha256: str
    extractor: str
    retrieved_at: str


@dataclasses.dataclass(frozen=True)
class ParsedWeeklyClaimsRelease:
    release_date: str
    publication_at: str
    publication_evidence: str
    reference_period: str
    initial_value: str
    initial_raw: str
    revision_reference_period: str | None
    revision_value: str | None
    revision_raw: str | None


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: pathlib.Path, relative: str) -> pathlib.Path:
    candidate = pathlib.Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("dol_eta_manifest_path_invalid")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("dol_eta_manifest_path_escapes_root") from error
    return resolved


def _visible_text(raw: str) -> str:
    value = html.unescape(re.sub(r"<[^>]*>", " ", raw))
    return " ".join(value.replace("\u2212", "-").split())


def _date(month: str, day: str, year: str) -> dt.date:
    key = month.rstrip(".").lower()
    if len(key) in {3, 4}:
        matches = [name for name in MONTHS if name.startswith(key)]
        if len(matches) != 1:
            raise ValueError("dol_eta_release_month_invalid")
        key = matches[0]
    try:
        return dt.date(int(year), MONTHS[key], int(day))
    except (KeyError, ValueError) as error:
        raise ValueError("dol_eta_release_date_invalid") from error


def _reference_period(date: dt.date) -> str:
    return "week ending " + date.isoformat()


def _decimal_count(value: str) -> str:
    cleaned = value.replace(",", "")
    if not re.fullmatch(r"[0-9]+", cleaned):
        raise ValueError("dol_eta_initial_claims_value_invalid")
    return str(Decimal(cleaned).quantize(Decimal("1")))


def _url_release_date(source_url: str) -> str:
    match = CANONICAL_URL.fullmatch(source_url)
    if not match:
        raise ValueError("dol_eta_source_url_not_canonical_first_party")
    compact = match.group("date")
    # DOL's 2019-01-03 archive filename carries the prior two-digit year.
    # The embedded embargo date is still required to match this result.
    if (
        compact[4:] != match.group("year")[2:]
        and source_url.lower() != KNOWN_FILENAME_YEAR_EXCEPTION
    ):
        raise ValueError("dol_eta_artifact_identity_year_mismatch")
    return f"{match.group('year')}-{compact[:2]}-{compact[2:4]}"


def _artifact_identity(source_url: str) -> str:
    match = CANONICAL_URL.fullmatch(source_url)
    if not match:
        raise ValueError("dol_eta_source_url_not_canonical_first_party")
    return (
        f"dol_eta:press:{match.group('year')}:"
        f"{match.group('date').lower()}.{match.group('extension').lower()}"
    )


def load_dol_eta_artifacts(
    repo_root: pathlib.Path,
    manifest_path: pathlib.Path,
) -> list[DolEtaArtifact]:
    """Load successful, hash-verified acquisition rows in stable URL order."""
    root = repo_root.resolve()
    manifest_root = manifest_path.resolve().parent
    try:
        manifest_root.relative_to(root)
    except ValueError as error:
        raise ValueError("dol_eta_manifest_outside_repository") from error
    rows: list[DolEtaArtifact] = []
    seen: set[str] = set()
    with manifest_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"dol_eta_manifest_json_invalid:{line_number}"
                ) from error
            if row.get("manifest_version") != 1:
                raise ValueError("dol_eta_actual_manifest_version_invalid")
            if row.get("parser_version") != PARSER_VERSION:
                raise ValueError("dol_eta_actual_manifest_parser_version_invalid")
            source_url = str(row.get("source_url", ""))
            url_match = CANONICAL_URL.fullmatch(source_url)
            if source_url in seen:
                raise ValueError("dol_eta_actual_manifest_duplicate_url")
            seen.add(source_url)
            if row.get("http_acquisition_result") != "succeeded":
                continue
            release_date = _url_release_date(source_url)
            if row.get("release_date") != release_date:
                raise ValueError("dol_eta_manifest_release_date_mismatch")
            source_path = _inside(root, str(row.get("local_artifact_path", "")))
            parser_path = _inside(root, str(row.get("parser_artifact_path", "")))
            if not source_path.is_file() or not parser_path.is_file():
                raise ValueError("dol_eta_manifest_artifact_missing")
            if row.get("requested_url") != source_url or row.get("final_url") != source_url:
                raise ValueError("dol_eta_manifest_resource_identity_mismatch")
            if row.get("source_artifact_identity") != _artifact_identity(source_url):
                raise ValueError("dol_eta_manifest_artifact_identity_mismatch")
            assert url_match is not None
            expected_source = manifest_root / url_match.group("year") / (
                f"{url_match.group('date').lower()}."
                f"{url_match.group('extension').lower()}"
            )
            if source_path != expected_source:
                raise ValueError("dol_eta_manifest_source_path_mismatch")
            extractor = str(row.get("extractor", ""))
            if url_match.group("extension").lower() == "asp":
                if parser_path != source_path or extractor != "none":
                    raise ValueError("dol_eta_manifest_html_provenance_invalid")
            elif (
                parser_path != source_path.with_suffix(".txt")
                or not extractor.startswith("pdftotext-")
            ):
                raise ValueError("dol_eta_manifest_pdf_provenance_invalid")
            source_sha = str(row.get("source_artifact_sha256", ""))
            parser_sha = str(row.get("parser_artifact_sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", source_sha):
                raise ValueError("dol_eta_manifest_source_sha256_invalid")
            if not re.fullmatch(r"[0-9a-f]{64}", parser_sha):
                raise ValueError("dol_eta_manifest_parser_sha256_invalid")
            if _sha256(source_path) != source_sha:
                raise ValueError("dol_eta_manifest_source_sha256_mismatch")
            if _sha256(parser_path) != parser_sha:
                raise ValueError("dol_eta_manifest_parser_sha256_mismatch")
            rows.append(DolEtaArtifact(
                release_date=release_date,
                source_url=source_url,
                source_artifact_identity=_artifact_identity(source_url),
                source_path=source_path,
                source_repository_path=source_path.relative_to(root).as_posix(),
                source_sha256=source_sha,
                parser_path=parser_path,
                parser_repository_path=parser_path.relative_to(root).as_posix(),
                parser_sha256=parser_sha,
                extractor=extractor,
                retrieved_at=canonical_instant(str(row["retrieved_at"])),
            ))
    return sorted(rows, key=lambda row: (row.release_date, row.source_url))


def parse_dol_eta_weekly_claims_actual(
    artifact: DolEtaArtifact,
) -> ParsedWeeklyClaimsRelease:
    text = _visible_text(artifact.parser_path.read_text(
        encoding="utf-8", errors="strict"
    ))
    if not re.search(
        r"UNEMPLOYMENT INSURANCE WEEKLY CLAIMS(?: REPORT)?", text,
        re.IGNORECASE,
    ):
        raise ValueError("dol_eta_weekly_claims_title_missing")

    embargo = re.search(
        r"(?:TRANSMISSION OF MATERIALS? IN THIS RELEASE IS )?"
        r"EMBARGOED UNTIL.{0,500}?"
        r"(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{2})\s*"
        r"(?P<meridiem>[AP])\.?M\.?\s*\((?P<zone>[^)]+)\)"
        r".{0,250}?(?P<month>[A-Za-z]+\.?)\s+"
        r"(?P<day>[0-9]{1,2})\s*,?\s*(?P<year>[0-9]{4})",
        text,
        re.IGNORECASE,
    )
    if not embargo:
        raise ValueError("publication_time_unproven:dol_eta_embargo_missing")
    hour = int(embargo.group("hour"))
    minute = int(embargo.group("minute"))
    if not 1 <= hour <= 12 or not 0 <= minute <= 59:
        raise ValueError("publication_time_unproven:dol_eta_embargo_invalid")
    meridiem = embargo.group("meridiem").upper()
    hour = hour % 12 + (12 if meridiem == "P" else 0)
    release_date = _date(
        embargo.group("month"), embargo.group("day"), embargo.group("year")
    )
    if release_date.isoformat() != artifact.release_date:
        raise ValueError("publication_time_unproven:release_date_url_mismatch")
    zone = embargo.group("zone").strip().upper()
    if zone not in {"EST", "EDT", "EASTERN"}:
        raise ValueError("publication_time_unproven:timezone_unsupported")
    local = dt.datetime.combine(
        release_date, dt.time(hour, minute), tzinfo=NEW_YORK
    )
    if zone in {"EST", "EDT"} and local.tzname() != zone:
        raise ValueError("publication_time_unproven:timezone_contradiction")
    publication_at = canonical_instant(local)
    publication_evidence = embargo.group(0)

    initial = re.search(
        r"(?P<raw>In the week ending\s+"
        r"(?P<month>[A-Za-z]+\.?)\s+(?P<day>[0-9]{1,2})"
        r"(?:\s*,\s*(?P<year>[0-9]{4}))?\s*,?\s*"
        r"the advance figure for seasonally adjusted initial claims was\s+"
        r"(?P<value>[0-9][0-9,]*))",
        text,
        re.IGNORECASE,
    )
    if not initial:
        raise ValueError(
            "actual_value_unavailable:seasonally_adjusted_advance_missing"
        )
    reference_year = initial.group("year") or str(release_date.year)
    reference_date = _date(
        initial.group("month"), initial.group("day"), reference_year
    )
    if initial.group("year") is None and reference_date > release_date:
        reference_date = reference_date.replace(year=reference_date.year - 1)
    if reference_date > release_date or (release_date - reference_date).days > 14:
        raise ValueError("actual_value_ambiguous:reference_week_invalid")

    sentence_end = text.find(".", initial.end())
    sentence_end = len(text) if sentence_end < 0 else sentence_end + 1
    initial_sentence = text[initial.start():sentence_end]
    prior_date = reference_date - dt.timedelta(days=7)
    revision_value: str | None = None
    revision_raw: str | None = None
    revision_window = text[initial.start():initial.start() + 1000]
    explicit_revision = re.search(
        r"(?P<raw>The previous week's (?:level|figure) was revised(?:\s+"
        r"(?:up|down) by\s+[0-9][0-9,]*)?\s+from\s+"
        r"[0-9][0-9,]*\s+to\s+(?P<value>[0-9][0-9,]*))",
        revision_window,
        re.IGNORECASE,
    )
    if explicit_revision:
        revision_value = _decimal_count(explicit_revision.group("value"))
        revision_raw = explicit_revision.group("raw")
    else:
        labelled_revision = re.search(
            r"(?P<raw>previous week's revised (?:figure|level) of\s+"
            r"(?P<value>[0-9][0-9,]*))",
            revision_window,
            re.IGNORECASE,
        )
        if labelled_revision:
            revision_value = _decimal_count(labelled_revision.group("value"))
            revision_raw = labelled_revision.group("raw")

    return ParsedWeeklyClaimsRelease(
        release_date=release_date.isoformat(),
        publication_at=publication_at,
        publication_evidence=publication_evidence,
        reference_period=_reference_period(reference_date),
        initial_value=_decimal_count(initial.group("value")),
        initial_raw=initial.group("raw"),
        revision_reference_period=(
            _reference_period(prior_date) if revision_value is not None else None
        ),
        revision_value=revision_value,
        revision_raw=revision_raw,
    )


def _candidate(
    artifact: DolEtaArtifact,
    parsed: ParsedWeeklyClaimsRelease,
    event: EconomicEvent,
    *,
    publication_state: str,
    revision_sequence: int,
    value: str,
    raw: str,
    reference_period: str,
) -> ReleaseActualCandidate:
    provenance: Mapping[str, object] = {
        "adapter": PARSER_VERSION,
        "artifact_identity": artifact.source_artifact_identity,
        "artifact_sha256": artifact.source_sha256,
        "parser_artifact_path": artifact.parser_repository_path,
        "parser_artifact_sha256": artifact.parser_sha256,
        "extractor": artifact.extractor,
        "publication_timestamp_evidence": parsed.publication_evidence,
        "publication_timestamp_basis": "embedded_release_embargo_date_time_zone",
        "actual_evidence": raw,
        "statistic": "seasonally_adjusted_initial_claims_advance_level",
        "reference_period": reference_period,
        "revision_semantics": (
            "current_reporting_week_advance_first_published_value"
            if publication_state == "initial"
            else "next_release_explicit_previous_week_revised_value"
        ),
        "retrieved_at_basis": "acquisition_manifest_system_observation",
    }
    observation_id = (
        f"dol_eta:weekly_claims:{artifact.release_date}:"
        f"{reference_period.removeprefix('week ending ')}:{publication_state}"
    )
    return ReleaseActualCandidate(
        source_family="DOL_ETA",
        candidate_event_family="WEEKLY_CLAIMS",
        candidate_source_event_id=event.source_event_id,
        candidate_reference_period=reference_period,
        source_agency="DOL_ETA",
        source_observation_id=observation_id,
        publication_state=publication_state,
        revision_sequence=revision_sequence,
        available_at=parsed.publication_at,
        retrieved_at=artifact.retrieved_at,
        source_url=artifact.source_url,
        source_artifact_path=artifact.source_repository_path,
        source_artifact_sha256=artifact.source_sha256,
        semantic_contract=DOL_ETA_WEEKLY_CLAIMS_SEMANTIC_CONTRACT,
        source_provenance=provenance,
        actual_raw=raw,
        actual_value_kind="scalar",
        actual_value_low=value,
        actual_value_high=None,
        actual_canonical_value_low=value,
        actual_canonical_value_high=None,
        actual_unit="count",
        actual_scale="1",
        actual_qualifier=None,
    )


def extract_dol_eta_candidates(
    artifact: DolEtaArtifact,
    events: Sequence[EconomicEvent],
) -> tuple[list[ReleaseActualCandidate], list[Rejection]]:
    """Parse and exact-map one occurrence; every ambiguity fails closed."""
    try:
        parsed = parse_dol_eta_weekly_claims_actual(artifact)
    except (OSError, UnicodeError, ValueError) as error:
        message = str(error)
        if message.startswith("publication_time_unproven"):
            decision = "publication_time_unproven"
        elif message.startswith("actual_value_unavailable"):
            decision = "actual_value_unavailable"
        elif message.startswith("actual_value_ambiguous"):
            decision = "actual_value_ambiguous"
        else:
            decision = "unsupported_source_format"
        return [], [Rejection(
            source_family="DOL_ETA",
            artifact=artifact.source_repository_path,
            candidate_event_family="WEEKLY_CLAIMS",
            candidate_reference_period=None,
            decision=decision,
            rejection_reason=message,
        )]

    by_url = [
        event for event in events
        if event.source_agency == "DOL_ETA"
        and event.event_family == "WEEKLY_CLAIMS"
        and event.source_url == artifact.source_url
    ]
    if len(by_url) != 1:
        decision = "event_mapping_ambiguous" if by_url else "missing_event_mapping"
        return [], [Rejection(
            "DOL_ETA", artifact.source_repository_path, "WEEKLY_CLAIMS",
            parsed.reference_period, decision,
            "exact_source_url_mapping_not_unique",
        )]
    current = by_url[0]
    if (
        current.reference_period != parsed.reference_period
        or canonical_instant(current.event_timestamp_utc)
        != canonical_instant(parsed.publication_at)
    ):
        return [], [Rejection(
            "DOL_ETA", artifact.source_repository_path, "WEEKLY_CLAIMS",
            parsed.reference_period, "event_mapping_ambiguous",
            "catalog_release_or_reference_identity_mismatch",
        )]

    candidates = [_candidate(
        artifact, parsed, current,
        publication_state="initial", revision_sequence=0,
        value=parsed.initial_value, raw=parsed.initial_raw,
        reference_period=parsed.reference_period,
    )]
    rejections: list[Rejection] = []
    if parsed.revision_value is not None:
        assert parsed.revision_reference_period is not None
        assert parsed.revision_raw is not None
        prior = [
            event for event in events
            if event.source_agency == "DOL_ETA"
            and event.event_family == "WEEKLY_CLAIMS"
            and event.reference_period == parsed.revision_reference_period
        ]
        if len(prior) == 1:
            candidates.append(_candidate(
                artifact, parsed, prior[0],
                publication_state="revision", revision_sequence=1,
                value=parsed.revision_value, raw=parsed.revision_raw,
                reference_period=parsed.revision_reference_period,
            ))
        else:
            rejections.append(Rejection(
                "DOL_ETA", artifact.source_repository_path, "WEEKLY_CLAIMS",
                parsed.revision_reference_period,
                "event_mapping_ambiguous" if prior else "missing_event_mapping",
                "prior_week_reference_mapping_not_unique",
            ))
    return candidates, rejections


def extract_dol_eta_corpus(
    artifacts: Iterable[DolEtaArtifact],
    events: Sequence[EconomicEvent],
) -> tuple[list[ReleaseActualCandidate], list[Rejection]]:
    candidates: list[ReleaseActualCandidate] = []
    rejections: list[Rejection] = []
    for artifact in artifacts:
        found, rejected = extract_dol_eta_candidates(artifact, events)
        candidates.extend(found)
        rejections.extend(rejected)
    return candidates, rejections
