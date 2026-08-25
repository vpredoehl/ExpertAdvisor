#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

BASE = "https://www.bea.gov"
ARCHIVE = f"{BASE}/news/archive"

ROOT = Path("EconomicCalendar/raw/bea")
INPUT = ROOT / "manifest_filtered.csv"

RECOVERY_ROOT = ROOT / "recovery"
RECOVERY_RELEASES = RECOVERY_ROOT / "releases"
RECOVERY_SEARCHES = RECOVERY_ROOT / "search_pages"

GAP_REPORT = RECOVERY_ROOT / "gap_report.csv"
RECOVERY_MANIFEST = RECOVERY_ROOT / "recovery_manifest.csv"
RECOVERY_FAILURES = RECOVERY_ROOT / "recovery_failures.csv"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
    "AppleWebKit/537.36 "
    "ExpertAdvisor EconomicCalendar BEA gap recovery"
)

REQUEST_DELAY_SECONDS = 0.25
MAX_SEARCH_PAGES = 8

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
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

QUARTER_RE = re.compile(
    r"""
    (?:
        (?P<qnum>[1-4])(?:st|nd|rd|th)\s+quarter
        |
        (?P<word>first|second|third|fourth)\s+quarter
    )
    (?:\s+and\s+(?:annual|year))?
    \s+
    (?P<year>20\d{2})
    """,
    re.I | re.X,
)

GDP_ESTIMATE_PATTERNS = [
    ("advance", re.compile(r"\badvance\s+estimate\b", re.I)),
    ("initial", re.compile(r"\binitial\s+estimate\b", re.I)),
    ("second", re.compile(r"\bsecond\s+estimate\b", re.I)),
    ("third", re.compile(r"\bthird\s+estimate\b", re.I)),
    ("updated", re.compile(r"\bupdated\s+estimate\b", re.I)),
]

PCE_RE = re.compile(
    r"""
    personal\s+income\s+and\s+outlays
    [,:]?
    (?:\s+data\s+update\s*,?)?
    \s*
    (?P<month>
        January|February|March|April|May|June|
        July|August|September|October|November|December
    )
    \s+
    (?P<year>20\d{2})
    """,
    re.I | re.X,
)


def fetch(url: str) -> tuple[int, bytes]:
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    try:
        with urlopen(req, timeout=30) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()
    except URLError as exc:
        raise RuntimeError(f"{url}: {exc}") from exc


def clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def slug_filename(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}.html"


def parse_gdp_quarter(title: str):
    m = QUARTER_RE.search(title)
    if not m:
        return None

    if m.group("qnum"):
        quarter = int(m.group("qnum"))
    else:
        quarter = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
        }[m.group("word").lower()]

    year = int(m.group("year"))

    return year, quarter


def parse_gdp_estimate(title: str):
    for name, rx in GDP_ESTIMATE_PATTERNS:
        if rx.search(title):
            return name
    return None


def normalize_gdp_slot(estimate: str):
    """
    Normalize recent BEA naming into conceptual release slots.

    Traditional:
        advance -> advance
        second  -> second
        third   -> third

    Recent exceptional naming:
        initial -> advance-like first release
        updated -> second-like subsequent update

    We retain original estimate wording elsewhere.
    """
    if estimate == "advance":
        return "advance"
    if estimate == "initial":
        return "advance"
    if estimate == "second":
        return "second"
    if estimate == "updated":
        return "second"
    if estimate == "third":
        return "third"
    return None


def parse_pce_reference(title: str):
    m = PCE_RE.search(title)
    if not m:
        return None

    month = MONTHS[m.group("month").lower()]
    year = int(m.group("year"))

    return year, month


def quarter_end(year: int, quarter: int) -> date:
    if quarter == 1:
        return date(year, 3, 31)
    if quarter == 2:
        return date(year, 6, 30)
    if quarter == 3:
        return date(year, 9, 30)
    return date(year, 12, 31)


def days_since(d: date, through: date):
    return (through - d).days


def expected_gdp_slots(
    start_release_year: int,
    through: date,
):
    """
    Conservative availability rules.

    We only call a slot "expected" once enough time has elapsed after
    quarter-end that the release should ordinarily have occurred:

        advance: 30 days
        second : 60 days
        third  : 90 days

    These thresholds are deliberately conservative and are used only
    to find recovery candidates. They do NOT manufacture event dates.
    """
    result = []

    first_reference_year = start_release_year - 1

    for year in range(first_reference_year, through.year + 1):
        for quarter in range(1, 5):
            qe = quarter_end(year, quarter)
            elapsed = days_since(qe, through)

            if elapsed >= 30:
                result.append((year, quarter, "advance"))

            if elapsed >= 60:
                result.append((year, quarter, "second"))

            if elapsed >= 90:
                result.append((year, quarter, "third"))

    return result


def month_iter(start_year: int, start_month: int, end_year: int, end_month: int):
    y = start_year
    m = start_month

    while (y, m) <= (end_year, end_month):
        yield y, m

        m += 1
        if m == 13:
            m = 1
            y += 1


def expected_pce_months(start_release_year: int, through: date):
    """
    PCE for a reference month normally releases in the following month.

    Use a conservative 35-day threshold after month-end before declaring
    a reference month expected.
    """
    start_ref_year = start_release_year - 1
    start_ref_month = 12

    result = []

    for year, month in month_iter(
        start_ref_year,
        start_ref_month,
        through.year,
        through.month,
    ):
        if month == 12:
            month_end = date(year, 12, 31)
        else:
            month_end = date(year, month + 1, 1)
            month_end = date.fromordinal(month_end.toordinal() - 1)

        if (through - month_end).days >= 35:
            result.append((year, month))

    return result


def extract_archive_entries(page_html: str):
    pattern = re.compile(
        r'<a[^>]+href="(?P<href>/news/(?P<year>\d{4})/[^"]+)"[^>]*>'
        r'(?P<title>.*?)</a>',
        re.I | re.S,
    )

    seen = set()

    for m in pattern.finditer(page_html):
        href = html.unescape(m.group("href"))
        archive_year = int(m.group("year"))
        title = clean_text(m.group("title"))

        if not title:
            continue

        url = urljoin(BASE, href)
        key = (url, title)

        if key in seen:
            continue

        seen.add(key)

        yield archive_year, title, url


def is_national_gdp_title(title: str):
    t = title.lower()

    reject = (
        "by state",
        "personal income by state",
        "local area",
        "metropolitan",
        "county",
        "puerto rico",
        "guam",
        "american samoa",
        "virgin islands",
        "northern mariana",
        "cnmi",
    )

    if any(x in t for x in reject):
        return False

    parsed_quarter = parse_gdp_quarter(title)
    estimate = parse_gdp_estimate(title)

    return parsed_quarter is not None and estimate is not None


def build_archive_url(keyword: str, page: int):
    params = {
        "created_1": "All",
        "field_related_product_target_id": "All",
        "title": keyword,
        "page": str(page),
    }

    return ARCHIVE + "?" + urlencode(params)


def search_archive(keyword: str, search_id: str):
    found = []

    for page in range(MAX_SEARCH_PAGES):
        url = build_archive_url(keyword, page)

        try:
            status, body = fetch(url)
        except Exception as exc:
            return found, f"{url}: {exc}"

        if status != 200:
            return found, f"{url}: HTTP {status}"

        path = RECOVERY_SEARCHES / f"{search_id}_page_{page:02d}.html"
        path.write_bytes(body)

        text = body.decode("utf-8", errors="replace")
        entries = list(extract_archive_entries(text))

        found.extend(entries)

        if not entries:
            break

        time.sleep(REQUEST_DELAY_SECONDS)

    unique = {}

    for archive_year, title, url in found:
        unique[url] = (archive_year, title, url)

    return list(unique.values()), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--through",
        default="2026-08-24",
        help="Only consider releases expected on/before this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--start-release-year",
        type=int,
        default=2010,
    )

    args = parser.parse_args()

    through = datetime.strptime(args.through, "%Y-%m-%d").date()

    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    RECOVERY_RELEASES.mkdir(parents=True, exist_ok=True)
    RECOVERY_SEARCHES.mkdir(parents=True, exist_ok=True)

    with INPUT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    observed_gdp = defaultdict(list)
    observed_pce = defaultdict(list)

    for row in rows:
        title = row["title"]

        if row["family"] == "GDP":
            qp = parse_gdp_quarter(title)
            estimate = parse_gdp_estimate(title)

            if qp and estimate:
                slot = normalize_gdp_slot(estimate)

                if slot:
                    observed_gdp[(qp[0], qp[1], slot)].append(row)

        elif row["family"] == "PCE":
            ref = parse_pce_reference(title)

            if ref:
                observed_pce[ref].append(row)

    expected_gdp = expected_gdp_slots(
        args.start_release_year,
        through,
    )

    expected_pce = expected_pce_months(
        args.start_release_year,
        through,
    )

    missing_gdp = [
        slot
        for slot in expected_gdp
        if slot not in observed_gdp
    ]

    missing_pce = [
        ref
        for ref in expected_pce
        if ref not in observed_pce
    ]

    gap_rows = []

    for year, quarter, slot in missing_gdp:
        gap_rows.append({
            "family": "GDP",
            "reference_year": year,
            "reference_period": f"Q{quarter}",
            "slot": slot,
            "status": "missing",
        })

    for year, month in missing_pce:
        gap_rows.append({
            "family": "PCE",
            "reference_year": year,
            "reference_period": f"{month:02d}",
            "slot": "monthly",
            "status": "missing",
        })

    with GAP_REPORT.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "family",
            "reference_year",
            "reference_period",
            "slot",
            "status",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(gap_rows)

    print("BEA gap detection")
    print()
    print(f"Through date       : {through}")
    print(f"Observed GDP slots : {len(observed_gdp)}")
    print(f"Observed PCE months: {len(observed_pce)}")
    print(f"Missing GDP slots  : {len(missing_gdp)}")
    print(f"Missing PCE months : {len(missing_pce)}")
    print()

    if missing_gdp:
        print("Missing GDP quarter/estimate slots:")
        for year, quarter, slot in missing_gdp:
            print(f"  {year} Q{quarter} {slot}")

    print()

    if missing_pce:
        print("Missing PCE reference months:")
        for year, month in missing_pce:
            print(f"  {year}-{month:02d}")

    print()
    print("Searching BEA archive only for missing slots...")
    print()

    recovery_rows = []
    failure_rows = []

    known_urls = {row["url"] for row in rows}

    #
    # GDP recovery
    #
    for year, quarter, wanted_slot in missing_gdp:
        qword = {
            1: "First Quarter",
            2: "Second Quarter",
            3: "Third Quarter",
            4: "Fourth Quarter",
        }[quarter]

        search_terms = [
            f"Gross Domestic Product {qword} {year}",
            f"GDP {qword} {year}",
        ]

        search_candidates = {}

        for attempt, keyword in enumerate(search_terms, start=1):
            search_id = f"gdp_{year}_q{quarter}_{wanted_slot}_{attempt}"

            entries, error = search_archive(keyword, search_id)

            if error:
                failure_rows.append({
                    "family": "GDP",
                    "reference_year": year,
                    "reference_period": f"Q{quarter}",
                    "slot": wanted_slot,
                    "search_keyword": keyword,
                    "url": "",
                    "error": error,
                })
                continue

            for archive_year, title, url in entries:
                search_candidates[url] = (archive_year, title, url)

        matches = []

        for archive_year, title, url in search_candidates.values():
            if not is_national_gdp_title(title):
                continue

            qp = parse_gdp_quarter(title)
            estimate = parse_gdp_estimate(title)

            if qp != (year, quarter):
                continue

            normalized_slot = normalize_gdp_slot(estimate)

            if normalized_slot != wanted_slot:
                continue

            matches.append(
                (archive_year, title, url, estimate)
            )

        if not matches:
            failure_rows.append({
                "family": "GDP",
                "reference_year": year,
                "reference_period": f"Q{quarter}",
                "slot": wanted_slot,
                "search_keyword": " | ".join(search_terms),
                "url": "",
                "error": "no_matching_BEА_archive_result",
            })
            continue

        for archive_year, title, url, original_estimate in matches:
            if url in known_urls:
                continue

            filename = slug_filename(url)
            path = RECOVERY_RELEASES / filename

            try:
                status, body = fetch(url)
            except Exception as exc:
                failure_rows.append({
                    "family": "GDP",
                    "reference_year": year,
                    "reference_period": f"Q{quarter}",
                    "slot": wanted_slot,
                    "search_keyword": "",
                    "url": url,
                    "error": str(exc),
                })
                continue

            if status != 200:
                failure_rows.append({
                    "family": "GDP",
                    "reference_year": year,
                    "reference_period": f"Q{quarter}",
                    "slot": wanted_slot,
                    "search_keyword": "",
                    "url": url,
                    "error": f"HTTP {status}",
                })
                continue

            path.write_bytes(body)

            recovery_rows.append({
                "family": "GDP",
                "reference_year": year,
                "reference_period": f"Q{quarter}",
                "slot": wanted_slot,
                "original_estimate_wording": original_estimate,
                "archive_year": archive_year,
                "title": title,
                "url": url,
                "filename": filename,
                "http_status": status,
                "bytes": len(body),
            })

            known_urls.add(url)

            print(
                f"RECOVERED GDP {year} Q{quarter} {wanted_slot}: "
                f"{title}"
            )

            time.sleep(REQUEST_DELAY_SECONDS)

    #
    # PCE recovery
    #
    for year, month in missing_pce:
        month_name = MONTH_NAMES[month]

        keyword = f"Personal Income and Outlays {month_name} {year}"
        search_id = f"pce_{year}_{month:02d}"

        entries, error = search_archive(keyword, search_id)

        if error:
            failure_rows.append({
                "family": "PCE",
                "reference_year": year,
                "reference_period": f"{month:02d}",
                "slot": "monthly",
                "search_keyword": keyword,
                "url": "",
                "error": error,
            })
            continue

        matches = []

        for archive_year, title, url in entries:
            ref = parse_pce_reference(title)

            if ref != (year, month):
                continue

            if "data update" in title.lower():
                continue

            matches.append((archive_year, title, url))

        if not matches:
            failure_rows.append({
                "family": "PCE",
                "reference_year": year,
                "reference_period": f"{month:02d}",
                "slot": "monthly",
                "search_keyword": keyword,
                "url": "",
                "error": "no_matching_BEA_archive_result",
            })
            continue

        for archive_year, title, url in matches:
            if url in known_urls:
                continue

            filename = slug_filename(url)
            path = RECOVERY_RELEASES / filename

            try:
                status, body = fetch(url)
            except Exception as exc:
                failure_rows.append({
                    "family": "PCE",
                    "reference_year": year,
                    "reference_period": f"{month:02d}",
                    "slot": "monthly",
                    "search_keyword": keyword,
                    "url": url,
                    "error": str(exc),
                })
                continue

            if status != 200:
                failure_rows.append({
                    "family": "PCE",
                    "reference_year": year,
                    "reference_period": f"{month:02d}",
                    "slot": "monthly",
                    "search_keyword": keyword,
                    "url": url,
                    "error": f"HTTP {status}",
                })
                continue

            path.write_bytes(body)

            recovery_rows.append({
                "family": "PCE",
                "reference_year": year,
                "reference_period": f"{month:02d}",
                "slot": "monthly",
                "original_estimate_wording": "",
                "archive_year": archive_year,
                "title": title,
                "url": url,
                "filename": filename,
                "http_status": status,
                "bytes": len(body),
            })

            known_urls.add(url)

            print(
                f"RECOVERED PCE {year}-{month:02d}: "
                f"{title}"
            )

            time.sleep(REQUEST_DELAY_SECONDS)

    manifest_fields = [
        "family",
        "reference_year",
        "reference_period",
        "slot",
        "original_estimate_wording",
        "archive_year",
        "title",
        "url",
        "filename",
        "http_status",
        "bytes",
    ]

    with RECOVERY_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        w.writerows(recovery_rows)

    failure_fields = [
        "family",
        "reference_year",
        "reference_period",
        "slot",
        "search_keyword",
        "url",
        "error",
    ]

    with RECOVERY_FAILURES.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=failure_fields)
        w.writeheader()
        w.writerows(failure_rows)

    print()
    print("Recovery complete.")
    print()
    print(f"Gap report        : {GAP_REPORT}")
    print(f"Recovered manifest: {RECOVERY_MANIFEST}")
    print(f"Recovery failures : {RECOVERY_FAILURES}")
    print(f"Recovered HTML    : {RECOVERY_RELEASES}")
    print()
    print(f"Recovered pages   : {len(recovery_rows)}")
    print(f"Unresolved/search errors: {len(failure_rows)}")
    print()
    print("No database writes were performed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
