#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import html
import re
import time
from collections import Counter, defaultdict
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

ROOT = Path("EconomicCalendar/raw/census")

INDEX_DIR = ROOT / "index_pages"
RELEASE_ROOT = ROOT / "releases"

RETAIL_DIR = RELEASE_ROOT / "retail_sales"
DURABLE_DIR = RELEASE_ROOT / "durable_goods"

MANIFEST = ROOT / "manifest.csv"
FAILURES = ROOT / "http_failures.csv"
GAPS = ROOT / "coverage_gaps.csv"
COVERAGE = ROOT / "coverage_by_year.csv"

THROUGH = date(2026, 8, 24)

RETAIL_HISTORY = (
    "https://www.census.gov/retail/marts/"
    "historic_releases.html"
)
RETAIL_CURRENT = (
    "https://www.census.gov/retail/sales.html"
)
RETAIL_SCHEDULE = (
    "https://www.census.gov/retail/"
    "release_schedule.html"
)

DURABLE_HISTORY = (
    "https://www.census.gov/manufacturing/m3/adv/"
    "historical_data/index.html"
)
DURABLE_CURRENT = (
    "https://www.census.gov/manufacturing/m3/adv/"
    "current/index.html"
)
DURABLE_SCHEDULE = (
    "https://www.census.gov/manufacturing/m3/"
    "release_schedule.html"
)

REQUEST_DELAY = 0.20

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
    "ExpertAdvisor EconomicCalendar Census research downloader"
)

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

MONTH_NAMES = {
    value: key.capitalize()
    for key, value in MONTHS.items()
}

MONTH_YEAR_RE = re.compile(
    r"\b("
    + "|".join(MONTHS)
    + r")\s+(20\d{2})\b",
    re.I,
)

#
# The requested event window begins 2010-01-01.
#
# December 2009 reference-period releases can therefore be needed
# because they are normally published in January 2010.
#
FIRST_REFERENCE = (2009, 12)

#
# Conservative known reference-period endpoints as of 2026-08-24:
#
# Retail:
#   July 2026 was released 2026-08-14.
#
# Durable Goods:
#   June 2026 was released 2026-07-27.
#   July 2026 is scheduled after our through-date.
#
LAST_REFERENCE = {
    "RETAIL_SALES": (2026, 7),
    "DURABLE_GOODS": (2026, 6),
}


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._href = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return

        attrs = dict(attrs)
        self._href = attrs.get("href")
        self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag.lower() != "a":
            return

        if self._href is not None:
            text = html.unescape(
                " ".join(self._text)
            )
            text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

            self.links.append(
                (
                    self._href,
                    text,
                )
            )

        self._href = None
        self._text = []


def fetch(url: str):
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/pdf,"
                "application/vnd.ms-excel,"
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet,*/*"
            ),
        },
    )

    try:
        with urlopen(
            req,
            timeout=30,
        ) as response:
            return (
                response.status,
                response.geturl(),
                response.headers.get(
                    "Content-Type",
                    "",
                ),
                response.read(),
            )
    except HTTPError as exc:
        return (
            exc.code,
            url,
            exc.headers.get(
                "Content-Type",
                "",
            ),
            exc.read(),
        )
    except URLError as exc:
        raise RuntimeError(
            f"{url}: {exc}"
        ) from exc


def archive_index(
    name: str,
    url: str,
):
    status, final_url, content_type, body = fetch(
        url
    )

    path = INDEX_DIR / f"{name}.html"

    if status == 200:
        path.write_bytes(body)

    print(
        f"{name:<24} "
        f"HTTP {status} "
        f"{len(body):>8} bytes"
    )

    return {
        "name": name,
        "url": url,
        "final_url": final_url,
        "status": status,
        "content_type": content_type,
        "body": body,
        "path": path,
    }


def month_iter(
    first: tuple[int, int],
    last: tuple[int, int],
):
    year, month = first

    while (year, month) <= last:
        yield year, month

        month += 1

        if month == 13:
            month = 1
            year += 1


def reference_from_anchor(
    anchor_text: str,
):
    m = MONTH_YEAR_RE.search(
        anchor_text
    )

    if not m:
        return None

    month = MONTHS[
        m.group(1).lower()
    ]
    year = int(m.group(2))

    return year, month


def retail_reference_from_url(
    url: str,
):
    #
    # Historical MARTS filenames commonly resemble:
    #
    #   adv1012.pdf
    #
    # meaning December 2010.
    #
    name = urlparse(url).path.split("/")[-1]

    m = re.search(
        r"\badv(?P<yy>\d{2})(?P<mm>\d{2})"
        r"(?:\D|$)",
        name,
        re.I,
    )

    if not m:
        return None

    year = 2000 + int(
        m.group("yy")
    )
    month = int(
        m.group("mm")
    )

    if not 1 <= month <= 12:
        return None

    return year, month


def durable_reference_from_url(
    url: str,
):
    path = urlparse(url).path
    name = path.split("/")[-1]

    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    #
    # Examples:
    #
    #   dec10adv.pdf
    #   jan26adv.pdf
    #
    m = re.search(
        r"(?P<mon>"
        + "|".join(month_map)
        + r")(?P<yy>\d{2})adv",
        name,
        re.I,
    )

    if m:
        return (
            2000 + int(m.group("yy")),
            month_map[
                m.group("mon").lower()
            ],
        )

    #
    # Some newer paths contain a four-digit year directory.
    # Anchor text remains the preferred fallback.
    #
    return None


def extension_score(url: str):
    path = urlparse(url).path.lower()

    if path.endswith(".pdf"):
        return 0

    if path.endswith(".xlsx"):
        return 1

    if path.endswith(".xls"):
        return 2

    if path.endswith(".html"):
        return 3

    if path.endswith(".htm"):
        return 3

    return 9


def supported_release_url(url: str):
    host = (
        urlparse(url).hostname
        or ""
    ).lower()

    if host not in {
        "www.census.gov",
        "census.gov",
        "www2.census.gov",
    }:
        return False

    path = urlparse(url).path.lower()

    return path.endswith(
        (
            ".pdf",
            ".xls",
            ".xlsx",
            ".htm",
            ".html",
        )
    )


def discover_links(
    family: str,
    page_url: str,
    body: bytes,
    discovery_source: str,
):
    text = body.decode(
        "utf-8",
        errors="replace",
    )

    parser = LinkParser()
    parser.feed(text)

    result = []

    for href, anchor in parser.links:
        if not href:
            continue

        url = urljoin(
            page_url,
            html.unescape(href),
        )

        if not supported_release_url(url):
            continue

        ref = reference_from_anchor(
            anchor
        )

        if ref is None:
            if family == "RETAIL_SALES":
                ref = retail_reference_from_url(
                    url
                )
            else:
                ref = durable_reference_from_url(
                    url
                )

        if ref is None:
            continue

        year, month = ref

        if (
            (year, month) < FIRST_REFERENCE
            or
            (year, month)
            > LAST_REFERENCE[family]
        ):
            continue

        result.append({
            "family": family,
            "reference_year": year,
            "reference_month": month,
            "anchor_text": anchor,
            "url": url,
            "discovery_source":
                discovery_source,
        })

    return result


def filename_for(
    family: str,
    year: int,
    month: int,
    url: str,
):
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()

    if not suffix:
        suffix = ".bin"

    digest = hashlib.sha1(
        url.encode("utf-8")
    ).hexdigest()[:8]

    prefix = (
        "retail"
        if family == "RETAIL_SALES"
        else "durable"
    )

    return (
        f"{prefix}_"
        f"{year:04d}_{month:02d}_"
        f"{digest}"
        f"{suffix}"
    )


def select_candidates(
    candidates,
):
    #
    # Many Census archive entries expose both PDF and Excel
    # versions of the same release.
    #
    # For raw archival / timestamp extraction, prefer the PDF.
    #
    grouped = defaultdict(list)

    for row in candidates:
        key = (
            row["family"],
            row["reference_year"],
            row["reference_month"],
        )

        grouped[key].append(row)

    selected = []

    for key in sorted(grouped):
        options = grouped[key]

        options = sorted(
            options,
            key=lambda row: (
                extension_score(
                    row["url"]
                ),
                0
                if (
                    row[
                        "discovery_source"
                    ]
                    == "historical_archive"
                )
                else 1,
                row["url"],
            ),
        )

        chosen = dict(options[0])

        chosen[
            "candidate_count"
        ] = len(options)

        selected.append(chosen)

    return selected


def main():
    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    RETAIL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    DURABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Census RETAIL_SALES / DURABLE_GOODS "
        "authoritative acquisition"
    )
    print()
    print(
        f"Through date      : {THROUGH}"
    )
    print(
        f"First reference   : "
        f"{FIRST_REFERENCE[0]}-"
        f"{FIRST_REFERENCE[1]:02d}"
    )
    print(
        "Retail last ref   : "
        f"{LAST_REFERENCE['RETAIL_SALES'][0]}-"
        f"{LAST_REFERENCE['RETAIL_SALES'][1]:02d}"
    )
    print(
        "Durable last ref  : "
        f"{LAST_REFERENCE['DURABLE_GOODS'][0]}-"
        f"{LAST_REFERENCE['DURABLE_GOODS'][1]:02d}"
    )
    print()

    pages = [
        (
            "retail_history",
            RETAIL_HISTORY,
            "RETAIL_SALES",
            "historical_archive",
        ),
        (
            "retail_current",
            RETAIL_CURRENT,
            "RETAIL_SALES",
            "current_release_page",
        ),
        (
            "retail_schedule",
            RETAIL_SCHEDULE,
            None,
            "release_schedule",
        ),
        (
            "durable_history",
            DURABLE_HISTORY,
            "DURABLE_GOODS",
            "historical_archive",
        ),
        (
            "durable_current",
            DURABLE_CURRENT,
            "DURABLE_GOODS",
            "current_release_page",
        ),
        (
            "durable_schedule",
            DURABLE_SCHEDULE,
            None,
            "release_schedule",
        ),
    ]

    discovered = []

    print(
        "Archiving authoritative Census index/"
        "schedule pages..."
    )

    for (
        name,
        url,
        family,
        discovery_source,
    ) in pages:
        result = archive_index(
            name,
            url,
        )

        if (
            result["status"] == 200
            and family is not None
        ):
            discovered.extend(
                discover_links(
                    family,
                    result["final_url"],
                    result["body"],
                    discovery_source,
                )
            )

            #
            # The Census "current" pages are themselves authoritative
            # release artifacts. The newest release often has not yet
            # migrated into the historical archive, so there may be no
            # month-identifiable PDF/Excel link for discover_links() to
            # select.
            #
            # LAST_REFERENCE is intentionally tied to THROUGH and has
            # already established which current reference period is
            # eligible for this acquisition snapshot.
            #
            # Add the current page itself as a candidate. If the same
            # reference period is also present in the historical archive,
            # select_candidates() will still prefer the archived PDF.
            #
            if discovery_source == "current_release_page":
                current_year, current_month = (
                    LAST_REFERENCE[family]
                )

                #
                # Retail exposes the authoritative current monthly
                # release PDF directly from the current release page.
                # Prefer that release artifact over the HTML wrapper.
                #
                # Durable Goods does not expose an equivalent
                # self-contained current release PDF here; its current
                # HTML page is itself the authoritative release
                # artifact.
                #
                if family == "RETAIL_SALES":
                    current_url = (
                        "https://www.census.gov/retail/"
                        "marts/www/marts_current.pdf"
                    )
                    current_source = (
                        "current_release_pdf"
                    )
                    current_anchor = (
                        "Authoritative Census current "
                        "release PDF"
                    )
                else:
                    current_url = result["final_url"]
                    current_source = (
                        "current_release_page_self"
                    )
                    current_anchor = (
                        "Authoritative Census current "
                        "release page"
                    )

                discovered.append({
                    "family":
                        family,
                    "reference_year":
                        current_year,
                    "reference_month":
                        current_month,
                    "anchor_text":
                        current_anchor,
                    "url":
                        current_url,
                    "discovery_source":
                        current_source,
                })

        time.sleep(
            REQUEST_DELAY
        )

    #
    # De-duplicate exact discovery URLs.
    #
    unique = {}

    for row in discovered:
        key = (
            row["family"],
            row["reference_year"],
            row["reference_month"],
            row["url"],
        )

        unique[key] = row

    candidates = list(
        unique.values()
    )

    selected = select_candidates(
        candidates
    )

    print()
    print(
        f"Discovered release-file links : "
        f"{len(candidates)}"
    )
    print(
        f"Selected monthly artifacts     : "
        f"{len(selected)}"
    )
    print()

    manifest_rows = []
    failure_rows = []

    for i, row in enumerate(
        selected,
        start=1,
    ):
        family = row["family"]
        year = row["reference_year"]
        month = row["reference_month"]
        url = row["url"]

        output_dir = (
            RETAIL_DIR
            if family == "RETAIL_SALES"
            else DURABLE_DIR
        )

        filename = filename_for(
            family,
            year,
            month,
            url,
        )

        path = (
            output_dir
            / filename
        )

        try:
            (
                status,
                final_url,
                content_type,
                body,
            ) = fetch(url)
        except Exception as exc:
            failure_rows.append({
                "family": family,
                "reference_year": year,
                "reference_month":
                    f"{month:02d}",
                "url": url,
                "http_status": "",
                "error": str(exc),
            })

            print(
                f"[{i:03d}/{len(selected):03d}] "
                f"ERROR "
                f"{family:<14} "
                f"{year}-{month:02d}: "
                f"{exc}"
            )

            continue

        outcome = (
            "downloaded"
            if status == 200
            else "http_failure"
        )

        if status == 200:
            path.write_bytes(body)
        else:
            failure_rows.append({
                "family": family,
                "reference_year": year,
                "reference_month":
                    f"{month:02d}",
                "url": url,
                "http_status": status,
                "error": "",
            })

        manifest_rows.append({
            "source_agency":
                "CENSUS",
            "event_family":
                family,
            "reference_year":
                year,
            "reference_month":
                f"{month:02d}",
            "reference_period":
                (
                    f"{MONTH_NAMES[month]} "
                    f"{year}"
                ),
            "anchor_text":
                row["anchor_text"],
            "url":
                url,
            "final_url":
                final_url,
            "filename":
                filename,
            "content_type":
                content_type,
            "http_status":
                status,
            "bytes":
                len(body),
            "candidate_count":
                row[
                    "candidate_count"
                ],
            "discovery_source":
                row[
                    "discovery_source"
                ],
            "outcome":
                outcome,
        })

        print(
            f"[{i:03d}/{len(selected):03d}] "
            f"{family:<14} "
            f"{year}-{month:02d} "
            f"HTTP {status} "
            f"{len(body):>8} "
            f"{filename}"
        )

        time.sleep(
            REQUEST_DELAY
        )

    manifest_fields = [
        "source_agency",
        "event_family",
        "reference_year",
        "reference_month",
        "reference_period",
        "anchor_text",
        "url",
        "final_url",
        "filename",
        "content_type",
        "http_status",
        "bytes",
        "candidate_count",
        "discovery_source",
        "outcome",
    ]

    with MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=
                manifest_fields,
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    failure_fields = [
        "family",
        "reference_year",
        "reference_month",
        "url",
        "http_status",
        "error",
    ]

    with FAILURES.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=
                failure_fields,
        )

        writer.writeheader()
        writer.writerows(
            failure_rows
        )

    #
    # Reference-period coverage.
    #
    successful = {
        (
            row["event_family"],
            int(
                row["reference_year"]
            ),
            int(
                row["reference_month"]
            ),
        )
        for row in manifest_rows
        if str(
            row["http_status"]
        ) == "200"
    }

    gap_rows = []

    for family in (
        "RETAIL_SALES",
        "DURABLE_GOODS",
    ):
        for year, month in month_iter(
            FIRST_REFERENCE,
            LAST_REFERENCE[family],
        ):
            key = (
                family,
                year,
                month,
            )

            if key in successful:
                continue

            gap_rows.append({
                "source_agency":
                    "CENSUS",
                "event_family":
                    family,
                "reference_year":
                    year,
                "reference_month":
                    f"{month:02d}",
                "reference_period":
                    (
                        f"{MONTH_NAMES[month]} "
                        f"{year}"
                    ),
                "status":
                    "missing_release_artifact",
            })

    gap_fields = [
        "source_agency",
        "event_family",
        "reference_year",
        "reference_month",
        "reference_period",
        "status",
    ]

    with GAPS.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=gap_fields,
        )

        writer.writeheader()
        writer.writerows(
            gap_rows
        )

    #
    # Year/reference-month coverage report.
    #
    actual_counts = Counter()

    for (
        family,
        year,
        month,
    ) in successful:
        actual_counts[
            (
                family,
                year,
            )
        ] += 1

    coverage_rows = []

    for family in (
        "RETAIL_SALES",
        "DURABLE_GOODS",
    ):
        for year in range(
            2009,
            2027,
        ):
            expected_months = [
                (y, m)
                for y, m in month_iter(
                    FIRST_REFERENCE,
                    LAST_REFERENCE[
                        family
                    ],
                )
                if y == year
            ]

            if not expected_months:
                continue

            expected = len(
                expected_months
            )
            actual = actual_counts[
                (
                    family,
                    year,
                )
            ]

            coverage_rows.append({
                "source_agency":
                    "CENSUS",
                "event_family":
                    family,
                "reference_year":
                    year,
                "expected_reference_months":
                    expected,
                "downloaded_reference_months":
                    actual,
                "status":
                    (
                        "complete"
                        if actual == expected
                        else "gap"
                    ),
            })

    coverage_fields = [
        "source_agency",
        "event_family",
        "reference_year",
        "expected_reference_months",
        "downloaded_reference_months",
        "status",
    ]

    with COVERAGE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=
                coverage_fields,
        )

        writer.writeheader()
        writer.writerows(
            coverage_rows
        )

    family_counts = Counter(
        row["event_family"]
        for row in manifest_rows
        if str(
            row["http_status"]
        ) == "200"
    )

    gap_counts = Counter(
        row["event_family"]
        for row in gap_rows
    )

    print()
    print(
        "Census acquisition / coverage summary"
    )
    print()
    print(
        f"Retail artifacts downloaded : "
        f"{family_counts['RETAIL_SALES']}"
    )
    print(
        f"Durable artifacts downloaded: "
        f"{family_counts['DURABLE_GOODS']}"
    )
    print(
        f"HTTP/download failures      : "
        f"{len(failure_rows)}"
    )
    print()
    print(
        f"Retail coverage gaps        : "
        f"{gap_counts['RETAIL_SALES']}"
    )
    print(
        f"Durable coverage gaps       : "
        f"{gap_counts['DURABLE_GOODS']}"
    )
    print()
    print(
        f"Manifest : {MANIFEST}"
    )
    print(
        f"Failures : {FAILURES}"
    )
    print(
        f"Gaps     : {GAPS}"
    )
    print(
        f"Coverage : {COVERAGE}"
    )
    print(
        f"Retail   : {RETAIL_DIR}"
    )
    print(
        f"Durable  : {DURABLE_DIR}"
    )
    print()

    if failure_rows:
        print(
            "RESULT: ACQUISITION INCOMPLETE - "
            "HTTP/download failures require inspection."
        )
        return 1

    if gap_rows:
        print(
            "RESULT: COVERAGE INCOMPLETE - "
            "inspect coverage_gaps.csv before timestamp extraction."
        )
        return 1

    print(
        "RESULT: PASS - Census raw release "
        "artifact coverage is complete for the "
        "requested reference-period window."
    )
    print(
        "No database writes were performed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
