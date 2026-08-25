#!/usr/bin/env python3

from __future__ import annotations

import csv
import html
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("EconomicCalendar/raw/bea")

BASE_MANIFEST = ROOT / "manifest_filtered.csv"
BASE_RELEASES = ROOT / "releases"

RECOVERY_MANIFEST = ROOT / "recovery_v2/recovered.csv"
RECOVERY_RELEASES = ROOT / "recovery_v2/releases"

OUTPUT = ROOT / "bea_canonical_events.csv"
DUPLICATES = ROOT / "bea_canonical_duplicates.csv"
FAILURES = ROOT / "bea_canonical_parse_failures.csv"

EASTERN = ZoneInfo("America/New_York")

MONTH_WORD = (
    r"January|February|March|April|May|June|"
    r"July|August|September|October|November|December"
)

RELEASE_RE = re.compile(
    r'EMBARGOED\s+'
    r'(?:UNTIL\s+RELEASE\s+AT|FOR\s+RELEASE:)\s+'
    r'(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*'
    r'(?P<ampm>A\.?M\.?|P\.?M\.?)\s*,?\s*'
    r'(?P<tz>EST|EDT)?\s*,?\s*'
    r'(?:(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY),?\s*)?'
    rf'(?P<month>{MONTH_WORD})\s+'
    r'(?P<day>\d{1,2}),\s+'
    r'(?P<year>\d{4})',
    re.I,
)


def strip_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_timestamp(page_text: str):
    text = strip_html(page_text)
    m = RELEASE_RE.search(text)

    if not m:
        return None, None

    hour = int(m.group("hour"))
    minute = int(m.group("minute"))

    ampm = m.group("ampm").replace(".", "").upper()

    if ampm == "PM" and hour != 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0

    dt_local = datetime.strptime(
        f"{m.group('month')} "
        f"{m.group('day')} "
        f"{m.group('year')} "
        f"{hour:02d}:{minute:02d}",
        "%B %d %Y %H:%M",
    ).replace(tzinfo=EASTERN)

    return dt_local, dt_local.astimezone(timezone.utc)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    base = read_csv(BASE_MANIFEST)
    recovery = read_csv(RECOVERY_MANIFEST)

    candidates = []

    for r in base:
        candidates.append({
            "family": r["family"],
            "title": r["title"],
            "url": r["url"],
            "filename": r["filename"],
            "source_set": "base",
            "path": BASE_RELEASES / r["filename"],
        })

    for r in recovery:
        candidates.append({
            "family": r["family"],
            "title": r["title"],
            "url": r["url"],
            "filename": r["filename"],
            "source_set": "recovery",
            "path": RECOVERY_RELEASES / r["filename"],
        })

    parsed = []
    failures = []

    for c in candidates:
        path = c["path"]

        if not path.exists():
            failures.append({
                "family": c["family"],
                "title": c["title"],
                "url": c["url"],
                "filename": c["filename"],
                "source_set": c["source_set"],
                "error": "missing_html",
            })
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        local_dt, utc_dt = parse_timestamp(text)

        if local_dt is None:
            failures.append({
                "family": c["family"],
                "title": c["title"],
                "url": c["url"],
                "filename": c["filename"],
                "source_set": c["source_set"],
                "error": "timestamp_not_found",
            })
            continue

        parsed.append({
            "source_agency": "BEA",
            "event_family": c["family"],
            "event_timestamp_utc": utc_dt.isoformat(),
            "source_local_date": local_dt.date().isoformat(),
            "source_local_time": local_dt.time().isoformat(),
            "source_timezone": "America/New_York",
            "title": c["title"],
            "url": c["url"],
            "filename": c["filename"],
            "source_set": c["source_set"],
        })

    #
    # Canonical event identity:
    # one BEA family event at one actual release instant.
    #
    groups = defaultdict(list)

    for r in parsed:
        groups[
            (
                r["source_agency"],
                r["event_family"],
                r["event_timestamp_utc"],
            )
        ].append(r)

    canonical = []
    duplicates = []

    for key in sorted(groups):
        rows = groups[key]

        #
        # Prefer base over recovery when the same event appears in both.
        # Otherwise choose deterministically by URL.
        #
        rows = sorted(
            rows,
            key=lambda r: (
                0 if r["source_set"] == "base" else 1,
                r["url"],
            ),
        )

        canonical.append(rows[0])

        if len(rows) > 1:
            for r in rows:
                duplicates.append(r)

    fields = [
        "source_agency",
        "event_family",
        "event_timestamp_utc",
        "source_local_date",
        "source_local_time",
        "source_timezone",
        "title",
        "url",
        "filename",
        "source_set",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(canonical)

    with DUPLICATES.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(duplicates)

    failure_fields = [
        "family",
        "title",
        "url",
        "filename",
        "source_set",
        "error",
    ]

    with FAILURES.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=failure_fields)
        w.writeheader()
        w.writerows(failures)

    counts = defaultdict(int)

    for r in canonical:
        counts[r["event_family"]] += 1

    print("BEA canonical build")
    print()
    print(f"Base candidates     : {len(base)}")
    print(f"Recovery candidates : {len(recovery)}")
    print(f"Total candidates    : {len(candidates)}")
    print(f"Parsed              : {len(parsed)}")
    print(f"Parse failures      : {len(failures)}")
    print(f"Duplicate rows      : {len(duplicates)}")
    print(f"Canonical events    : {len(canonical)}")
    print()
    print(f"GDP canonical       : {counts['GDP']}")
    print(f"PCE canonical       : {counts['PCE']}")
    print()
    print(f"Output              : {OUTPUT}")
    print(f"Duplicates          : {DUPLICATES}")
    print(f"Failures            : {FAILURES}")


if __name__ == "__main__":
    main()
