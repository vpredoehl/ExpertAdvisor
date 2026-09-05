#!/usr/bin/env python3
"""Deterministic Myfxbook pre-release Weekly Claims evidence processing."""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import html
import io
import json
import pathlib
import re
import urllib.parse
from collections import defaultdict
from decimal import Decimal
from typing import Iterable, Mapping


PARSER_VERSION = "myfxbook_weekly_claims_pre_release_snapshot_v1"
PROOF = "internet_archive_pre_release_capture"
UTC = dt.timezone.utc
ROW_RE = re.compile(
    r'<tr\b(?=[^>]*\bid=["\']calRow(?P<id>\d+)["\'])[^>]*>'
    r'(?P<body>.*?)</tr>', re.IGNORECASE | re.DOTALL,
)
META_TIMESTAMP_RE = re.compile(
    r'<meta\s+name=["\']server-timestamp["\']\s+content=["\'](\d+)["\']',
    re.IGNORECASE,
)
TIME_RE = re.compile(r'\btime=["\'](\d{10,13})["\']', re.IGNORECASE)
FORECAST_CELL_RE = re.compile(
    r'<td\b(?=[^>]*(?:data-concensus=|id=["\']concensus))[^>]*>'
    r'(?P<body>.*?)</td>', re.IGNORECASE | re.DOTALL,
)
ACTUAL_CELL_RE = re.compile(
    r'<td\b(?=[^>]*(?:data-actual=|id=["\']actualTip))[^>]*>'
    r'(?P<body>.*?)</td>', re.IGNORECASE | re.DOTALL,
)
VALUE_RE = re.compile(
    r'(?<![\w.])(?:[+-]?\d+(?:[.,]\d+)?\s*[Kk]\b|'
    r'[+-]?\d{1,3}(?:,\d{3})+\b|[+-]?\d{4,}\b)'
)
REFERENCE_RE = re.compile(
    r'>\s*Initial Jobless Claims\s*</a>\s*<span>\s*\(([^<]+)\)\s*</span>',
    re.IGNORECASE,
)


def canonical_instant(value: dt.datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp_missing_timezone")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def instant_from_epoch_millis(value: int) -> str:
    return canonical_instant(dt.datetime.fromtimestamp(value / 1000, tz=UTC))


def instant_from_cdx(value: str) -> str:
    if not re.fullmatch(r"\d{14}", value):
        raise ValueError("archive_capture_timestamp_invalid")
    parsed = dt.datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return canonical_instant(parsed)


def parse_instant(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _visible_text(value: str) -> str:
    value = re.sub(r"<script\b.*?</script>", " ", value,
                   flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(value).replace("\xa0", " ").split())


def _count_value(cell: str) -> str | None:
    match = VALUE_RE.search(_visible_text(cell))
    if not match:
        return None
    return match.group(0).strip()


def normalize_forecast(raw: str) -> tuple[str, str]:
    compact = raw.strip()
    k_match = re.fullmatch(r"([+-]?\d+(?:[.,]\d+)?)\s*[Kk]", compact)
    if k_match:
        numeric = Decimal(k_match.group(1).replace(",", ".")) * Decimal(1000)
    elif re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", compact):
        numeric = Decimal(compact.replace(",", ""))
    elif re.fullmatch(r"[+-]?\d+", compact):
        numeric = Decimal(compact)
    else:
        raise ValueError("weekly_claims_forecast_not_count")
    canonical = numeric
    if canonical != canonical.to_integral_value():
        raise ValueError("weekly_claims_forecast_fractional_count")
    canonical_text = str(int(canonical))
    return canonical_text, canonical_text


@dataclasses.dataclass(frozen=True)
class CatalogEvent:
    economic_event_id: int
    event_timestamp_utc: str
    source_event_id: str
    reference_period: str
    source_release_date: str
    official_source_url: str


@dataclasses.dataclass(frozen=True)
class SnapshotIdentity:
    capture_timestamp: str
    original_url: str

    def __post_init__(self) -> None:
        instant_from_cdx(self.capture_timestamp)
        parsed = urllib.parse.urlsplit(self.original_url)
        allowed_paths = {
            "/forex-economic-calendar",
            "/forex-economic-calendar/united-states",
            "/forex-economic-calendar/united-states/initial-jobless-claims",
            "/forex-economic-calendar/category/initial-jobless-claims",
        }
        if (parsed.scheme not in {"http", "https"} or
                parsed.hostname not in {"myfxbook.com", "www.myfxbook.com"} or
                parsed.port not in {None, 80, 443} or
                parsed.path.rstrip("/") not in allowed_paths or
                parsed.query or parsed.fragment):
            raise ValueError("myfxbook_archive_original_url_invalid")

    @property
    def available_at(self) -> str:
        return instant_from_cdx(self.capture_timestamp)

    @property
    def replay_url(self) -> str:
        return (
            f"https://web.archive.org/web/{self.capture_timestamp}id_/"
            f"{self.original_url}"
        )


@dataclasses.dataclass(frozen=True)
class ParsedForecast:
    myfxbook_event_id: int
    release_at: str
    reference_period: str | None
    forecast_raw: str | None
    actual_raw: str | None


@dataclasses.dataclass(frozen=True)
class SnapshotArtifact:
    identity: SnapshotIdentity
    provider_observed_at: str
    retrieved_at: str
    repository_path: str
    sha256: str
    rows: tuple[ParsedForecast, ...]

    @property
    def forecast_available_at(self) -> str:
        return max(self.identity.available_at, self.provider_observed_at)


def load_catalog(path: pathlib.Path) -> list[CatalogEvent]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "economic_event_id", "event_timestamp_utc", "source_event_id",
        "reference_period", "source_release_date", "official_source_url",
        "event_family", "source_agency", "currency",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("weekly_claims_catalog_contract_mismatch")
    events: list[CatalogEvent] = []
    for row in rows:
        if (row["event_family"], row["source_agency"], row["currency"]) != (
            "WEEKLY_CLAIMS", "DOL_ETA", "USD"
        ):
            raise ValueError("weekly_claims_catalog_identity_mismatch")
        events.append(CatalogEvent(
            int(row["economic_event_id"]), row["event_timestamp_utc"],
            row["source_event_id"], row["reference_period"],
            row["source_release_date"], row["official_source_url"],
        ))
    if len({event.economic_event_id for event in events}) != len(events):
        raise ValueError("weekly_claims_catalog_duplicate_event_id")
    return sorted(events, key=lambda item: item.event_timestamp_utc)


def parse_snapshot(
    data: bytes, identity: SnapshotIdentity, retrieved_at: str,
    repository_path: str,
) -> SnapshotArtifact:
    text = data.decode("utf-8", errors="replace")
    meta = META_TIMESTAMP_RE.search(text)
    if not meta:
        raise ValueError("myfxbook_server_timestamp_missing")
    provider_observed_at = instant_from_epoch_millis(int(meta.group(1)))
    capture = parse_instant(identity.available_at)
    observed = parse_instant(provider_observed_at)
    retrieved = parse_instant(retrieved_at)
    if observed - capture > dt.timedelta(minutes=5) or \
            capture - observed > dt.timedelta(hours=12):
        raise ValueError("myfxbook_archive_provider_clock_conflict")
    if retrieved < max(capture, observed):
        raise ValueError("myfxbook_retrieval_predates_historical_snapshot")

    parsed: list[ParsedForecast] = []
    for match in ROW_RE.finditer(text):
        body = match.group("body")
        if not re.search(r"(?:>|&nbsp;)\s*Initial Jobless Claims\s*(?:<|$)",
                         body, re.IGNORECASE):
            continue
        modern_us = "United States" in body and re.search(
            r">\s*USD\s*<", body, re.IGNORECASE
        )
        legacy_us = re.search(
            r'class=["\']US["\']|currencies/US\.png', body, re.IGNORECASE
        )
        if not modern_us and not legacy_us:
            raise ValueError("weekly_claims_provider_country_ambiguous")
        release = TIME_RE.search(body)
        if not release:
            raise ValueError("weekly_claims_provider_release_time_missing")
        epoch = int(release.group(1))
        if epoch < 10_000_000_000:
            epoch *= 1000
        forecast_cell = FORECAST_CELL_RE.search(body)
        actual_cell = ACTUAL_CELL_RE.search(body)
        reference = REFERENCE_RE.search(body)
        parsed.append(ParsedForecast(
            int(match.group("id")), instant_from_epoch_millis(epoch),
            reference.group(1).strip() if reference else None,
            _count_value(forecast_cell.group("body")) if forecast_cell else None,
            _count_value(actual_cell.group("body")) if actual_cell else None,
        ))
    return SnapshotArtifact(
        identity, provider_observed_at, retrieved_at, repository_path,
        sha256_bytes(data), tuple(parsed),
    )


def _reference_matches(provider: str | None, canonical: str) -> bool:
    if provider is None:
        return True
    match = re.search(r"week ending (\d{4})-(\d{2})-(\d{2})$", canonical)
    if not match:
        return False
    expected = (int(match.group(2)), int(match.group(3)))
    try:
        actual = dt.datetime.strptime(
            f"{match.group(1)}/{provider.strip()}", "%Y/%b/%d"
        )
    except ValueError:
        return False
    return (actual.month, actual.day) == expected


def prepare(
    events: Iterable[CatalogEvent], artifacts: Iterable[SnapshotArtifact],
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    by_release: dict[str, list[CatalogEvent]] = defaultdict(list)
    for event in events:
        by_release[event.event_timestamp_utc].append(event)

    possible: dict[int, list[tuple[SnapshotArtifact, ParsedForecast]]] = defaultdict(list)
    rejected: list[dict[str, object]] = []
    for artifact in artifacts:
        for row in artifact.rows:
            base: dict[str, object] = {
                "capture_timestamp": artifact.identity.capture_timestamp,
                "provider_source_url": artifact.identity.original_url,
                "myfxbook_event_id": row.myfxbook_event_id,
                "provider_release_at": row.release_at,
                "forecast_raw": row.forecast_raw,
                "actual_raw": row.actual_raw,
            }
            matches = by_release.get(row.release_at, [])
            if not matches:
                rejected.append({**base, "decision": "unmapped_release_timestamp"})
                continue
            if len(matches) != 1:
                rejected.append({**base, "decision": "ambiguous_release_timestamp"})
                continue
            event = matches[0]
            if row.reference_period and not _reference_matches(
                row.reference_period, event.reference_period
            ):
                rejected.append({**base, "decision": "reference_period_mismatch"})
                continue
            if row.actual_raw is not None:
                rejected.append({**base, "decision": "post_release_actual_present"})
                continue
            if row.forecast_raw is None:
                rejected.append({**base, "decision": "forecast_missing"})
                continue
            if artifact.forecast_available_at >= event.event_timestamp_utc:
                rejected.append({**base, "decision": "archive_not_pre_release"})
                continue
            possible[event.economic_event_id].append((artifact, row))

    eligible: list[dict[str, str]] = []
    event_by_id = {event.economic_event_id: event for event in events}
    for event_id, values in sorted(possible.items()):
        latest_at = max(artifact.forecast_available_at for artifact, _ in values)
        latest = [(artifact, row) for artifact, row in values
                  if artifact.forecast_available_at == latest_at]
        normalized = {normalize_forecast(row.forecast_raw or "")[1]
                      for _, row in latest}
        if len(normalized) != 1:
            for artifact, row in latest:
                rejected.append({
                    "capture_timestamp": artifact.identity.capture_timestamp,
                    "provider_source_url": artifact.identity.original_url,
                    "myfxbook_event_id": row.myfxbook_event_id,
                    "provider_release_at": row.release_at,
                    "forecast_raw": row.forecast_raw,
                    "actual_raw": row.actual_raw,
                    "decision": "conflicting_latest_provider_snapshots",
                })
            continue
        chosen_artifact, chosen_row = min(
            latest, key=lambda item: (
                item[0].identity.original_url, item[1].myfxbook_event_id
            )
        )
        source_value, canonical_value = normalize_forecast(
            chosen_row.forecast_raw or ""
        )
        event = event_by_id[event_id]
        observation_id = (
            f"myfxbook:calendar-row:{chosen_row.myfxbook_event_id}:"
            f"release:{event.event_timestamp_utc}:"
            f"archive:{chosen_artifact.identity.capture_timestamp}"
        )
        provenance = {
            "archive_capture_timestamp":
                chosen_artifact.identity.capture_timestamp,
            "archive_replay_url": chosen_artifact.identity.replay_url,
            "myfxbook_event_id": chosen_row.myfxbook_event_id,
            "parser_version": PARSER_VERSION,
            "provider": "MYFXBOOK",
            "provider_release_timestamp": chosen_row.release_at,
            "provider_source_url": chosen_artifact.identity.original_url,
            "reference_period_raw": chosen_row.reference_period,
        }
        eligible.append({
            "economic_event_id": str(event.economic_event_id),
            "event_family": "WEEKLY_CLAIMS",
            "event_timestamp_utc": event.event_timestamp_utc,
            "source_agency": "DOL_ETA",
            "source_event_id": event.source_event_id,
            "reference_period": event.reference_period,
            "source_release_date": event.source_release_date,
            "consensus_source": "MYFXBOOK",
            "myfxbook_event_id": str(chosen_row.myfxbook_event_id),
            "archive_capture_timestamp":
                chosen_artifact.identity.capture_timestamp,
            "provider_source_url": chosen_artifact.identity.original_url,
            "provider_release_timestamp": chosen_row.release_at,
            "provider_reference_period": chosen_row.reference_period or "",
            "source_observation_id": observation_id,
            "source_event_name": "Initial Jobless Claims",
            "source_artifact_path": chosen_artifact.repository_path,
            "source_artifact_sha256": chosen_artifact.sha256,
            "candidate_classification":
                "myfxbook_weekly_claims_pre_release_snapshot",
            "match_rule": "exact_family_currency_release_timestamp",
            "semantic_contract": PARSER_VERSION,
            "provider_observed_at": chosen_artifact.provider_observed_at,
            "forecast_available_at": chosen_artifact.forecast_available_at,
            "source_retrieved_at": chosen_artifact.retrieved_at,
            "forecast_availability_proof": PROOF,
            "provider_provenance": json.dumps(
                provenance, sort_keys=True, separators=(",", ":")
            ),
            "forecast_raw": chosen_row.forecast_raw or "",
            "forecast_parse_status": "parsed",
            "forecast_value_kind": "scalar",
            "forecast_value_low": source_value,
            "forecast_canonical_value_low": canonical_value,
            "forecast_unit": "count",
            "forecast_scale": "1",
            "forecast_qualifier": "",
            "pre_release_actual_raw": "",
        })
        for artifact, row in values:
            if (artifact, row) != (chosen_artifact, chosen_row):
                rejected.append({
                    "capture_timestamp": artifact.identity.capture_timestamp,
                    "provider_source_url": artifact.identity.original_url,
                    "myfxbook_event_id": row.myfxbook_event_id,
                    "provider_release_at": row.release_at,
                    "forecast_raw": row.forecast_raw,
                    "actual_raw": row.actual_raw,
                    "decision": (
                        "duplicate_identical_snapshot"
                        if normalize_forecast(row.forecast_raw or "")[1] ==
                           canonical_value
                        else "superseded_pre_release_forecast"
                    ),
                })
    return eligible, sorted(rejected, key=lambda row: (
        str(row.get("capture_timestamp", "")),
        int(row.get("myfxbook_event_id", 0)),
        str(row.get("decision", "")),
    ))


def csv_bytes(rows: list[Mapping[str, str]]) -> bytes:
    if not rows:
        raise ValueError("no_eligible_weekly_claims_consensus")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def json_lines(rows: Iterable[Mapping[str, object]]) -> bytes:
    return "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
