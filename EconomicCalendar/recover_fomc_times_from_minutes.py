#!/usr/bin/env python3

from __future__ import annotations

import csv
import html
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

BASE = "https://www.federalreserve.gov"

ROOT = Path("EconomicCalendar/raw/federal_reserve")

FAILURES_INPUT = ROOT / "fomc_timestamp_parse_failures.csv"
EXTRACTED_INPUT = ROOT / "fomc_events_extracted.csv"

MINUTES_DIR = ROOT / "minutes"
RECOVERED_OUTPUT = ROOT / "fomc_times_recovered_from_minutes.csv"
RECOVERY_FAILURES = ROOT / "fomc_minutes_recovery_failures.csv"

CANONICAL_OUTPUT = ROOT / "fomc_canonical_events.csv"
CANONICAL_DUPLICATES = ROOT / "fomc_canonical_duplicates.csv"

EASTERN = ZoneInfo("America/New_York")

REQUEST_DELAY = 0.20

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
    "ExpertAdvisor EconomicCalendar FOMC minutes recovery"
)

URL_DATE_RE = re.compile(
    r"(?:monetary)?"
    r"(?P<year>20\d{2})"
    r"(?P<month>\d{2})"
    r"(?P<day>\d{2})"
    r"[a-z]?\.(?:htm|html)$",
    re.I,
)

#
# The minutes use wording such as:
#
#   The vote encompassed approval of the statement below
#   to be released at 2:15 p.m.:
#
# or:
#
#   ...statement below to be released at 2:00 p.m.:
#
# Keep this tightly tied to "statement" and "released" so that
# meeting start times, press-conference times, etc. cannot match.
#
STATEMENT_RELEASE_RE = re.compile(
    r"""
    statement
    (?:
        .{0,500}?
    )
    (?:to\s+be\s+released|for\s+release)
    \s+at\s+
    (?P<hour>\d{1,2})
    (?:
        :(?P<minute>\d{2})
    )?
    \s*
    (?P<ampm>a\.?m\.?|p\.?m\.?)
    (?:\s+(?P<tz>EST|EDT))?
    """,
    re.I | re.X | re.S,
)


def fetch(url: str):
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    try:
        with urlopen(req, timeout=30) as r:
            return r.status, r.geturl(), r.read()
    except HTTPError as exc:
        return exc.code, url, exc.read()
    except URLError as exc:
        raise RuntimeError(f"{url}: {exc}") from exc


def strip_html(text: str) -> str:
    text = html.unescape(text)

    text = re.sub(
        r"</?(?:p|div|br|li|h1|h2|h3|h4|tr|td|span|blockquote)[^>]*>",
        " ",
        text,
        flags=re.I,
    )

    text = re.sub(r"<[^>]+>", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def release_date_from_statement_url(url: str):
    filename = url.rstrip("/").split("/")[-1]

    m = URL_DATE_RE.search(filename)

    if not m:
        return None

    return datetime(
        int(m.group("year")),
        int(m.group("month")),
        int(m.group("day")),
    ).date()


def minutes_url(release_date):
    return (
        f"{BASE}/monetarypolicy/"
        f"fomcminutes{release_date:%Y%m%d}.htm"
    )


def minutes_filename(release_date):
    return f"fomcminutes{release_date:%Y%m%d}.html"


def clock_to_24h(hour: int, minute: int, ampm: str):
    ampm = ampm.replace(".", "").upper()

    if not 1 <= hour <= 12:
        raise ValueError(f"invalid hour {hour}")

    if not 0 <= minute <= 59:
        raise ValueError(f"invalid minute {minute}")

    if ampm == "PM" and hour != 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0

    return hour, minute


def parse_statement_release_time(minutes_html: str):
    text = strip_html(minutes_html)

    matches = list(
        STATEMENT_RELEASE_RE.finditer(text)
    )

    if not matches:
        return None, "statement_release_time_not_found"

    parsed = []

    for m in matches:
        hour = int(m.group("hour"))
        minute = int(m.group("minute") or "0")

        hour, minute = clock_to_24h(
            hour,
            minute,
            m.group("ampm"),
        )

        parsed.append({
            "hour": hour,
            "minute": minute,
            "timezone_token": (
                m.group("tz") or ""
            ).upper(),
            "matched_text": re.sub(
                r"\s+",
                " ",
                m.group(0),
            ).strip(),
        })

    #
    # Multiple textual occurrences are acceptable only if they all
    # identify the same release clock time.
    #
    unique_times = {
        (
            x["hour"],
            x["minute"],
        )
        for x in parsed
    }

    if len(unique_times) != 1:
        return (
            None,
            "multiple_distinct_statement_release_times:"
            + ",".join(
                f"{h:02d}:{m:02d}"
                for h, m in sorted(unique_times)
            ),
        )

    return parsed[0], None


def timestamp_for(release_date, parsed):
    local_dt = datetime(
        release_date.year,
        release_date.month,
        release_date.day,
        parsed["hour"],
        parsed["minute"],
        tzinfo=EASTERN,
    )

    return (
        local_dt,
        local_dt.astimezone(timezone.utc),
    )


def read_csv(path: Path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def main():
    MINUTES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    failures = read_csv(FAILURES_INPUT)
    existing = read_csv(EXTRACTED_INPUT)

    target_failures = [
        r for r in failures
        if r["error"] == "release_time_not_found"
    ]

    print("FOMC timestamp recovery from official minutes")
    print()
    print(f"Existing extracted : {len(existing)}")
    print(f"Recovery targets   : {len(target_failures)}")
    print()

    recovered = []
    unresolved = []

    for i, row in enumerate(
        target_failures,
        start=1,
    ):
        statement_url = row["url"]

        release_date = release_date_from_statement_url(
            statement_url
        )

        if release_date is None:
            unresolved.append({
                **row,
                "minutes_url": "",
                "recovery_error":
                    "statement_date_not_derivable",
            })
            continue

        url = minutes_url(release_date)
        path = MINUTES_DIR / minutes_filename(
            release_date
        )

        try:
            status, final_url, body = fetch(url)
        except Exception as exc:
            unresolved.append({
                **row,
                "minutes_url": url,
                "recovery_error": str(exc),
            })

            print(
                f"[{i:02d}/{len(target_failures):02d}] "
                f"ERROR {release_date}: {exc}"
            )
            continue

        if status != 200:
            unresolved.append({
                **row,
                "minutes_url": url,
                "recovery_error":
                    f"minutes_http_{status}",
            })

            print(
                f"[{i:02d}/{len(target_failures):02d}] "
                f"HTTP {status} {release_date}"
            )
            continue

        path.write_bytes(body)

        text = body.decode(
            "utf-8",
            errors="replace",
        )

        parsed, error = parse_statement_release_time(
            text
        )

        if error:
            unresolved.append({
                **row,
                "minutes_url": final_url,
                "recovery_error": error,
            })

            print(
                f"[{i:02d}/{len(target_failures):02d}] "
                f"UNRESOLVED {release_date}: {error}"
            )
            continue

        local_dt, utc_dt = timestamp_for(
            release_date,
            parsed,
        )

        recovered.append({
            "source_agency":
                "FEDERAL_RESERVE",
            "event_family":
                "FOMC",
            "event_timestamp_utc":
                utc_dt.isoformat(),
            "source_local_date":
                local_dt.date().isoformat(),
            "source_local_time":
                local_dt.time().isoformat(),
            "source_timezone":
                "America/New_York",
            "release_timezone_token":
                parsed["timezone_token"],
            "title":
                row["page_title"],
            "url":
                statement_url,
            "filename":
                row["filename"],
            "discovery_year":
                row["discovery_year"],
            "discovery_source":
                "fomc_minutes_recovery",
            "timestamp_parser":
                "fomc_minutes_statement_release",
            "release_text":
                parsed["matched_text"],
            "timestamp_source_url":
                final_url,
        })

        print(
            f"[{i:02d}/{len(target_failures):02d}] "
            f"RECOVERED "
            f"{release_date} "
            f"{local_dt.time().isoformat()} "
            f"{row['page_title']}"
        )

        time.sleep(REQUEST_DELAY)

    recovered_fields = [
        "source_agency",
        "event_family",
        "event_timestamp_utc",
        "source_local_date",
        "source_local_time",
        "source_timezone",
        "release_timezone_token",
        "title",
        "url",
        "filename",
        "discovery_year",
        "discovery_source",
        "timestamp_parser",
        "release_text",
        "timestamp_source_url",
    ]

    with RECOVERED_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=recovered_fields,
        )
        w.writeheader()
        w.writerows(recovered)

    unresolved_fields = list(
        failures[0].keys()
    ) + [
        "minutes_url",
        "recovery_error",
    ]

    with RECOVERY_FAILURES.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=unresolved_fields,
        )
        w.writeheader()
        w.writerows(unresolved)

    #
    # Build canonical set from:
    #
    #   88 direct statement-page timestamps
    #   + recovered official-minutes timestamps
    #
    canonical = []

    for row in existing:
        c = dict(row)
        c["timestamp_source_url"] = row["url"]
        canonical.append(c)

    canonical.extend(recovered)

    canonical.sort(
        key=lambda r: (
            r["event_timestamp_utc"],
            r["url"],
        )
    )

    groups = defaultdict(list)

    for row in canonical:
        key = (
            row["source_agency"],
            row["event_family"],
            row["event_timestamp_utc"],
        )
        groups[key].append(row)

    duplicates = []

    for items in groups.values():
        if len(items) > 1:
            duplicates.extend(items)

    with CANONICAL_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=recovered_fields,
        )
        w.writeheader()
        w.writerows(canonical)

    with CANONICAL_DUPLICATES.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=recovered_fields,
        )
        w.writeheader()
        w.writerows(duplicates)

    year_counts = Counter(
        int(r["source_local_date"][:4])
        for r in canonical
    )

    time_counts = Counter(
        r["source_local_time"]
        for r in canonical
    )

    parser_counts = Counter(
        r["timestamp_parser"]
        for r in canonical
    )

    print()
    print("Recovery summary")
    print()
    print(f"Direct events     : {len(existing)}")
    print(f"Recovered         : {len(recovered)}")
    print(f"Unresolved        : {len(unresolved)}")
    print(f"Canonical events  : {len(canonical)}")
    print(f"Duplicate rows    : {len(duplicates)}")
    print()

    print(f"{'YEAR':<6} {'FOMC':>6}")
    print("-" * 14)

    for year in range(2010, 2027):
        print(
            f"{year:<6} "
            f"{year_counts[year]:>6}"
        )

    print()
    print("Release-time distribution:")

    for t, n in sorted(time_counts.items()):
        print(f"  {t:<10} {n:>4}")

    print()
    print("Timestamp provenance:")

    for parser_name, n in sorted(
        parser_counts.items()
    ):
        print(
            f"  {parser_name:<34} {n:>4}"
        )

    print()
    print(f"Recovered  : {RECOVERED_OUTPUT}")
    print(f"Failures   : {RECOVERY_FAILURES}")
    print(f"Canonical  : {CANONICAL_OUTPUT}")
    print(f"Duplicates : {CANONICAL_DUPLICATES}")
    print()

    if unresolved:
        print(
            "RESULT: INCOMPLETE - inspect recovery failures."
        )
        return 1

    if duplicates:
        print(
            "RESULT: INCOMPLETE - duplicate canonical keys exist."
        )
        return 1

    if len(canonical) != 136:
        print(
            "RESULT: INCOMPLETE - expected 136 canonical events, "
            f"found {len(canonical)}."
        )
        return 1

    print(
        "RESULT: PASS - all 136 FOMC statement timestamps "
        "have authoritative provenance."
    )
    print("No database writes were performed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
