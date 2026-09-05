#!/usr/bin/env python3
"""Acquire and prepare causal Myfxbook Weekly Claims archive snapshots."""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import gzip
import json
import pathlib
import sys
import tempfile
import time
import urllib.parse
import urllib.error
import urllib.request
from collections import Counter


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "EconomicCalendar"))

from myfxbook_weekly_claims_consensus import (  # noqa: E402
    PARSER_VERSION,
    SnapshotIdentity,
    canonical_instant,
    csv_bytes,
    json_lines,
    load_catalog,
    parse_instant,
    parse_snapshot,
    prepare,
    sha256_bytes,
)


ORIGINAL_URLS = (
    "https://www.myfxbook.com/forex-economic-calendar",
    "https://www.myfxbook.com/forex-economic-calendar/united-states",
    "https://www.myfxbook.com/forex-economic-calendar/united-states/initial-jobless-claims",
    "https://www.myfxbook.com/forex-economic-calendar/category/initial-jobless-claims",
)
USER_AGENT = "ExpertAdvisor causal economic provenance audit/1.0"


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = pathlib.Path(stream.name)
        stream.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch(url: str, timeout: int = 90, attempts: int = 3) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
                if data.startswith(b"\x1f\x8b"):
                    data = gzip.decompress(data)
                return data, response.geturl()
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def fetch_snapshot(identity: SnapshotIdentity) -> bytes:
    data, final_url = fetch(identity.replay_url)
    final = urllib.parse.urlsplit(final_url)
    expected_marker = f"/web/{identity.capture_timestamp}id_/"
    if (final.scheme != "https" or final.hostname != "web.archive.org" or
            expected_marker not in final.path):
        raise ValueError("myfxbook_archive_replay_redirect_invalid")
    return data


def acquire_cdx(path: pathlib.Path) -> list[dict[str, str]]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or payload.get("urls") != list(ORIGINAL_URLS):
            raise ValueError("myfxbook_cdx_index_contract_mismatch")
        return payload["captures"]

    captures: list[dict[str, str]] = []
    for original in ORIGINAL_URLS:
        parameters = urllib.parse.urlencode({
            "url": original,
            "output": "json",
            "filter": ["statuscode:200", "mimetype:text/html"],
            "collapse": "digest",
            "fl": "timestamp,original,statuscode,digest,length",
        }, doseq=True)
        data, _ = fetch("https://web.archive.org/cdx/search/cdx?" + parameters)
        rows = json.loads(data)
        if not rows or rows[0] != [
            "timestamp", "original", "statuscode", "digest", "length"
        ]:
            raise ValueError("myfxbook_cdx_response_contract_mismatch")
        captures.extend(dict(zip(rows[0], row)) for row in rows[1:])
    captures.sort(key=lambda row: (row["timestamp"], row["original"]))
    payload = {
        "version": 1,
        "urls": list(ORIGINAL_URLS),
        "captures": captures,
    }
    atomic_write(path, json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n")
    return captures


def select_captures(captures, events, window_days: int):
    instants = [parse_instant(event.event_timestamp_utc) for event in events]
    selected = {}
    for capture in captures:
        timestamp = dt.datetime.strptime(
            capture["timestamp"], "%Y%m%d%H%M%S"
        ).replace(tzinfo=dt.timezone.utc)
        index = bisect.bisect_right(instants, timestamp)
        if index == len(events):
            continue
        release = instants[index]
        if release - timestamp > dt.timedelta(days=window_days):
            continue
        key = (capture["original"], events[index].economic_event_id)
        if key not in selected or capture["timestamp"] > selected[key]["timestamp"]:
            selected[key] = capture
    return sorted(selected.values(), key=lambda row: (
        row["timestamp"], row["original"]
    ))


def artifact_name(identity: SnapshotIdentity) -> str:
    suffix = identity.original_url.removeprefix(
        "https://www.myfxbook.com/forex-economic-calendar"
    ).strip("/").replace("/", "-") or "calendar"
    return f"{identity.capture_timestamp}-{suffix}.html"


def run(args) -> dict[str, object]:
    repo = ROOT.resolve()
    raw = args.raw.resolve()
    audit = args.audit.resolve()
    for target in (raw, audit):
        target.relative_to(repo)
    events = load_catalog(args.catalog)
    cdx_path = raw / "cdx_index.json"
    captures = acquire_cdx(cdx_path)
    selected = select_captures(captures, events, args.window_days)
    retrieved_at = args.retrieved_at or canonical_instant(
        dt.datetime.now(tz=dt.timezone.utc)
    )

    manifest_path = raw / "acquisition_manifest.jsonl"
    prior = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prior[(row["capture_timestamp"], row["provider_source_url"])] = row
    manifest = []
    artifacts = []
    requested_captures = set(args.capture_timestamp or [])
    available_captures = {row["timestamp"] for row in selected}
    missing_requested = requested_captures - available_captures
    if missing_requested:
        raise ValueError(
            "myfxbook_requested_capture_not_selected:" +
            ",".join(sorted(missing_requested))
        )
    for capture in selected:
        identity = SnapshotIdentity(capture["timestamp"], capture["original"])
        key = (identity.capture_timestamp, identity.original_url)
        existing = prior.get(key)
        name = artifact_name(identity)
        path = raw / name
        repository_path = path.relative_to(repo).as_posix()
        if existing and existing["retrieval_result"] != "failed":
            if existing["retrieval_result"] == "retained":
                if not path.is_file() or \
                        sha256_bytes(path.read_bytes()) != existing["sha256"]:
                    raise ValueError("myfxbook_retained_artifact_conflict")
                artifact = parse_snapshot(
                    path.read_bytes(), identity, existing["retrieved_at"],
                    repository_path,
                )
                artifacts.append(artifact)
            manifest.append(existing)
            continue
        if not existing and path.is_file():
            data = path.read_bytes()
            artifact = parse_snapshot(
                data, identity, retrieved_at, repository_path
            )
            if not artifact.rows:
                raise ValueError("myfxbook_orphan_artifact_has_no_weekly_claims")
            artifacts.append(artifact)
            manifest.append({
                "manifest_version": 1,
                "parser_version": PARSER_VERSION,
                "capture_timestamp": identity.capture_timestamp,
                "provider_source_url": identity.original_url,
                "archive_replay_url": identity.replay_url,
                "retrieved_at": retrieved_at,
                "retrieval_result": "retained",
                "local_artifact_path": repository_path,
                "sha256": artifact.sha256,
                "weekly_claims_rows": len(artifact.rows),
                "diagnostic": None,
            })
            continue
        if requested_captures and \
                identity.capture_timestamp not in requested_captures:
            if existing:
                manifest.append(existing)
            continue
        row = {
            "manifest_version": 1,
            "parser_version": PARSER_VERSION,
            "capture_timestamp": identity.capture_timestamp,
            "provider_source_url": identity.original_url,
            "archive_replay_url": identity.replay_url,
            "retrieved_at": retrieved_at,
            "retrieval_result": "failed",
            "local_artifact_path": None,
            "sha256": None,
            "weekly_claims_rows": 0,
            "diagnostic": None,
        }
        try:
            time.sleep(args.delay_seconds)
            data = fetch_snapshot(identity)
            artifact = parse_snapshot(
                data, identity, retrieved_at, repository_path
            )
            if artifact.rows:
                atomic_write(path, data)
                artifacts.append(artifact)
                row.update({
                    "retrieval_result": "retained",
                    "local_artifact_path": repository_path,
                    "sha256": artifact.sha256,
                    "weekly_claims_rows": len(artifact.rows),
                })
            else:
                row["retrieval_result"] = "no_weekly_claims_row"
                row["sha256"] = sha256_bytes(data)
        except Exception as error:  # every failed source remains audited
            row["diagnostic"] = str(error)
        manifest.append(row)
        atomic_write(manifest_path, json_lines(sorted(manifest, key=lambda item: (
            item["capture_timestamp"], item["provider_source_url"]
        ))))

    atomic_write(manifest_path, json_lines(sorted(manifest, key=lambda item: (
        item["capture_timestamp"], item["provider_source_url"]
    ))))

    eligible, rejected = prepare(events, artifacts)
    eligible_path = audit / "myfxbook-weekly-claims-consensus-import.csv"
    rejected_path = audit / "myfxbook-weekly-claims-classifications.jsonl"
    atomic_write(eligible_path, csv_bytes(eligible))
    atomic_write(rejected_path, json_lines(rejected))
    decisions = Counter(row["decision"] for row in rejected)
    summary = {
        "parser_version": PARSER_VERSION,
        "production_catalog_events": len(events),
        "production_catalog_date_min": events[0].source_release_date,
        "production_catalog_date_max": events[-1].source_release_date,
        "cdx_captures": len(captures),
        "selected_capture_attempts": len(selected),
        "successful_retained_artifacts": len(artifacts),
        "retained_provider_rows": sum(len(item.rows) for item in artifacts),
        "eligible_consensus_observations": len(eligible),
        "eligible_unique_economic_event_ids": len({
            row["economic_event_id"] for row in eligible
        }),
        "eligible_date_min": min(
            (row["source_release_date"] for row in eligible), default=None
        ),
        "eligible_date_max": max(
            (row["source_release_date"] for row in eligible), default=None
        ),
        "classification_counts": dict(sorted(decisions.items())),
        "import_csv_sha256": sha256_bytes(eligible_path.read_bytes()),
        "classifications_sha256": sha256_bytes(rejected_path.read_bytes()),
        "raw_manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "cdx_index_sha256": sha256_bytes(cdx_path.read_bytes()),
    }
    atomic_write(audit / "myfxbook-weekly-claims-summary.json", json.dumps(
        summary, indent=2, sort_keys=True
    ).encode() + b"\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--raw", type=pathlib.Path, required=True)
    parser.add_argument("--audit", type=pathlib.Path, required=True)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--delay-seconds", type=float, default=0.75)
    parser.add_argument("--retrieved-at")
    parser.add_argument("--capture-timestamp", action="append")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
