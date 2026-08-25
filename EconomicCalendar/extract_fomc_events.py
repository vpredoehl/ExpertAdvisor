#!/usr/bin/env python3

from __future__ import annotations

import csv
import html
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("EconomicCalendar/raw/federal_reserve")

MANIFEST = ROOT / "manifest_filtered.csv"
STATEMENT_DIR = ROOT / "statements"

OUTPUT = ROOT / "fomc_events_extracted.csv"
FAILURES = ROOT / "fomc_timestamp_parse_failures.csv"
DUPLICATES = ROOT / "fomc_duplicate_candidates.csv"

EASTERN = ZoneInfo("America/New_York")

MONTH_WORD = (
    r"January|February|March|April|May|June|"
    r"July|August|September|October|November|December"
)

#
# Federal Reserve statement pages use several generations of release-line
# formatting. Keep separate patterns so we can report exactly which parser
# matched each event.
#
RELEASE_PATTERNS = [
    (
        "for_release_at",
        re.compile(
            r"""
            \bFor\s+release\s+at\s+
            (?P<hour>\d{1,2})
            (?:
                :(?P<minute>\d{2})
            )?
            \s*
            (?P<ampm>a\.?m\.?|p\.?m\.?)
            \s*
            (?P<tz>EST|EDT)?
            """,
            re.I | re.X,
        ),
    ),
    (
        "for_immediate_release_at",
        re.compile(
            r"""
            \bFor\s+immediate\s+release
            .*?
            \bat\s+
            (?P<hour>\d{1,2})
            (?:
                :(?P<minute>\d{2})
            )?
            \s*
            (?P<ampm>a\.?m\.?|p\.?m\.?)
            \s*
            (?P<tz>EST|EDT)?
            """,
            re.I | re.X | re.S,
        ),
    ),
]

#
# Fallback explicit-clock scanner. This is deliberately NOT used blindly:
# it is only considered when the surrounding text looks like release metadata.
#
CLOCK_RE = re.compile(
    r"""
    (?P<hour>\d{1,2})
    :
    (?P<minute>\d{2})
    \s*
    (?P<ampm>a\.?m\.?|p\.?m\.?)
    \s*
    (?P<tz>EST|EDT)?
    """,
    re.I | re.X,
)

#
# Statement URLs encode the calendar date reliably:
#
#   .../monetary20190130a.htm
#   .../press/monetary/20100127a.htm
#
URL_DATE_RE = re.compile(
    r"(?:monetary)?(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})[a-z]?"
    r"\.(?:htm|html)$",
    re.I,
)


def strip_html(text: str) -> str:
    text = html.unescape(text)

    #
    # Preserve rough line boundaries before removing tags.
    #
    text = re.sub(
        r"</?(?:p|div|br|li|h1|h2|h3|h4|tr|td|span)[^>]*>",
        "\n",
        text,
        flags=re.I,
    )

    text = re.sub(r"<[^>]+>", " ", text)

    lines = []

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def page_title(text: str) -> str:
    #
    # Prefer meaningful H1/H2/H3 text.
    #
    for tag in ("h1", "h2", "h3"):
        values = re.findall(
            rf"<{tag}[^>]*>(.*?)</{tag}>",
            text,
            re.I | re.S,
        )

        for raw in values:
            candidate = html.unescape(
                re.sub(r"<[^>]+>", " ", raw)
            )
            candidate = re.sub(
                r"\s+",
                " ",
                candidate,
            ).strip()

            if not candidate:
                continue

            if candidate.lower() in {
                "press release",
                "federal reserve",
                "monetary policy",
            }:
                continue

            return candidate

    m = re.search(
        r"<title[^>]*>(.*?)</title>",
        text,
        re.I | re.S,
    )

    if m:
        candidate = html.unescape(
            re.sub(r"<[^>]+>", " ", m.group(1))
        )
        candidate = re.sub(
            r"\s+",
            " ",
            candidate,
        ).strip()

        candidate = re.sub(
            r"\s*-\s*Federal Reserve Board.*$",
            "",
            candidate,
            flags=re.I,
        ).strip()

        return candidate

    return ""


def release_date_from_url(url: str):
    filename = url.rstrip("/").split("/")[-1]

    m = URL_DATE_RE.search(filename)

    if not m:
        return None

    try:
        return datetime(
            int(m.group("year")),
            int(m.group("month")),
            int(m.group("day")),
        ).date()
    except ValueError:
        return None


def clock_to_24h(
    hour: int,
    minute: int,
    ampm: str,
):
    ampm = ampm.replace(".", "").upper()

    if not 1 <= hour <= 12:
        raise ValueError(
            f"invalid 12-hour clock hour: {hour}"
        )

    if not 0 <= minute <= 59:
        raise ValueError(
            f"invalid minute: {minute}"
        )

    if ampm == "PM" and hour != 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0

    return hour, minute


def parse_release_time(plain_text: str):
    #
    # First: strongly structured release-line formats.
    #
    for parser_name, rx in RELEASE_PATTERNS:
        m = rx.search(plain_text)

        if not m:
            continue

        hour = int(m.group("hour"))
        minute = int(m.group("minute") or "0")

        hour, minute = clock_to_24h(
            hour,
            minute,
            m.group("ampm"),
        )

        return {
            "hour": hour,
            "minute": minute,
            "timezone_token": (
                m.group("tz") or ""
            ).upper(),
            "parser": parser_name,
            "matched_text": re.sub(
                r"\s+",
                " ",
                m.group(0),
            ).strip(),
        }

    #
    # Second: examine line-local explicit clocks, but only where the line
    # strongly looks like publication/release metadata.
    #
    context_markers = (
        "for release",
        "immediate release",
        "release at",
        "released at",
        "embargo",
    )

    for line in plain_text.splitlines():
        lower = line.lower()

        if not any(
            marker in lower
            for marker in context_markers
        ):
            continue

        m = CLOCK_RE.search(line)

        if not m:
            continue

        hour = int(m.group("hour"))
        minute = int(m.group("minute"))

        hour, minute = clock_to_24h(
            hour,
            minute,
            m.group("ampm"),
        )

        return {
            "hour": hour,
            "minute": minute,
            "timezone_token": (
                m.group("tz") or ""
            ).upper(),
            "parser": "release_context_clock",
            "matched_text": line[:500],
        }

    return None


def timestamp_for(
    release_date,
    parsed_time,
):
    local_dt = datetime(
        release_date.year,
        release_date.month,
        release_date.day,
        parsed_time["hour"],
        parsed_time["minute"],
        tzinfo=EASTERN,
    )

    utc_dt = local_dt.astimezone(
        timezone.utc
    )

    return local_dt, utc_dt


def main():
    with MANIFEST.open(
        newline="",
        encoding="utf-8",
    ) as f:
        manifest_rows = list(
            csv.DictReader(f)
        )

    extracted = []
    failures = []

    parser_counts = Counter()

    for row in manifest_rows:
        path = STATEMENT_DIR / row["filename"]

        if not path.exists():
            failures.append({
                "discovery_year":
                    row["discovery_year"],
                "url":
                    row["url"],
                "filename":
                    row["filename"],
                "page_title":
                    row.get("page_title", ""),
                "error":
                    "missing_html",
                "release_text":
                    "",
            })
            continue

        raw_html = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        title = (
            row.get("page_title", "").strip()
            or page_title(raw_html)
        )

        release_date = release_date_from_url(
            row["url"]
        )

        if release_date is None:
            failures.append({
                "discovery_year":
                    row["discovery_year"],
                "url":
                    row["url"],
                "filename":
                    row["filename"],
                "page_title":
                    title,
                "error":
                    "release_date_not_derivable_from_url",
                "release_text":
                    "",
            })
            continue

        plain = strip_html(raw_html)

        parsed_time = parse_release_time(
            plain
        )

        if parsed_time is None:
            #
            # Do NOT fabricate the usual 2:00 p.m. time.
            #
            failures.append({
                "discovery_year":
                    row["discovery_year"],
                "url":
                    row["url"],
                "filename":
                    row["filename"],
                "page_title":
                    title,
                "error":
                    "release_time_not_found",
                "release_text":
                    "",
            })
            continue

        local_dt, utc_dt = timestamp_for(
            release_date,
            parsed_time,
        )

        parser_counts[
            parsed_time["parser"]
        ] += 1

        extracted.append({
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
                parsed_time["timezone_token"],
            "title":
                title,
            "url":
                row["url"],
            "filename":
                row["filename"],
            "discovery_year":
                row["discovery_year"],
            "discovery_source":
                row["discovery_source"],
            "timestamp_parser":
                parsed_time["parser"],
            "release_text":
                parsed_time["matched_text"],
        })

    #
    # Canonical key:
    #
    #   source_agency
    #   event_family
    #   event_timestamp_utc
    #
    groups = defaultdict(list)

    for row in extracted:
        key = (
            row["source_agency"],
            row["event_family"],
            row["event_timestamp_utc"],
        )

        groups[key].append(row)

    duplicates = []

    for key, items in groups.items():
        if len(items) > 1:
            duplicates.extend(items)

    #
    # Chronological canonical output.
    #
    extracted.sort(
        key=lambda r: (
            r["event_timestamp_utc"],
            r["url"],
        )
    )

    fields = [
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
    ]

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(extracted)

    failure_fields = [
        "discovery_year",
        "url",
        "filename",
        "page_title",
        "error",
        "release_text",
    ]

    with FAILURES.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=failure_fields,
        )
        w.writeheader()
        w.writerows(failures)

    with DUPLICATES.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(duplicates)

    year_counts = Counter(
        int(r["source_local_date"][:4])
        for r in extracted
    )

    time_counts = Counter(
        r["source_local_time"]
        for r in extracted
    )

    print(
        "Federal Reserve FOMC timestamp extraction"
    )
    print()
    print(
        f"Candidates        : "
        f"{len(manifest_rows)}"
    )
    print(
        f"Parsed timestamps : "
        f"{len(extracted)}"
    )
    print(
        f"Parse failures    : "
        f"{len(failures)}"
    )
    print(
        f"Duplicate rows    : "
        f"{len(duplicates)}"
    )
    print()

    print(
        f"{'YEAR':<6} {'FOMC':>6}"
    )
    print("-" * 14)

    for year in range(2010, 2027):
        print(
            f"{year:<6} "
            f"{year_counts[year]:>6}"
        )

    print()
    print("Release-time distribution:")

    for release_time, count in sorted(
        time_counts.items()
    ):
        print(
            f"  {release_time:<10} {count:>4}"
        )

    print()
    print("Timestamp parser distribution:")

    for parser_name, count in sorted(
        parser_counts.items()
    ):
        print(
            f"  {parser_name:<28} {count:>4}"
        )

    print()
    print(f"Events     : {OUTPUT}")
    print(f"Failures   : {FAILURES}")
    print(f"Duplicates : {DUPLICATES}")
    print()
    print(
        "No default FOMC release time was assumed."
    )
    print(
        "No database writes were performed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
