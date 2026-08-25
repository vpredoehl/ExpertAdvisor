#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import html
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

BASE = "https://www.federalreserve.gov"

START_YEAR = 2010
END_YEAR = 2026

ROOT = Path("EconomicCalendar/raw/federal_reserve")
INDEX_DIR = ROOT / "index_pages"
STATEMENT_DIR = ROOT / "statements"

MANIFEST = ROOT / "manifest.csv"
FAILURES = ROOT / "http_failures.csv"

REQUEST_DELAY = 0.25

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
    "ExpertAdvisor EconomicCalendar FOMC research downloader"
)


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
            text = re.sub(
                r"\s+",
                " ",
                html.unescape(" ".join(self._text)),
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


def slug_filename(url: str):
    slug = url.rstrip("/").split("/")[-1]

    if not slug:
        slug = "index"

    if not slug.lower().endswith((".htm", ".html")):
        slug += ".html"

    stem = re.sub(
        r"\.(?:htm|html)$",
        "",
        slug,
        flags=re.I,
    )

    digest = hashlib.sha1(
        url.encode("utf-8")
    ).hexdigest()[:8]

    return f"{stem}_{digest}.html"


def historical_index_url(year: int):
    return (
        f"{BASE}/monetarypolicy/"
        f"fomchistorical{year}.htm"
    )


def is_statement_link(href: str, text: str):
    t = re.sub(r"\s+", " ", text).strip().lower()

    if not t:
        return False

    reject_text = (
        "longer-run goals",
        "implementation note",
        "minutes",
        "press conference",
        "projection",
        "transcript",
    )

    if any(x in t for x in reject_text):
        return False

    if t in {"statement", "fomc statement"}:
        return True

    if t.startswith("statement:"):
        return True

    h = href.lower()

    if re.search(
        r"/newsevents/pressreleases/"
        r"monetary20\d{6}[a-z]?\.htm",
        h,
    ):
        return True

    return False


def extract_statement_links(page_url: str, body: bytes):
    text = body.decode(
        "utf-8",
        errors="replace",
    )

    parser = LinkParser()
    parser.feed(text)

    result = {}

    for href, anchor_text in parser.links:
        if not is_statement_link(href, anchor_text):
            continue

        url = urljoin(
            page_url,
            html.unescape(href),
        )

        if not url.startswith(BASE + "/"):
            continue

        result[url] = anchor_text

    return result


def infer_year_from_url(url: str):
    m = re.search(
        r"monetary(20\d{2})\d{4}",
        url,
        re.I,
    )

    if m:
        return int(m.group(1))

    m = re.search(
        r"/(20\d{2})/",
        url,
    )

    if m:
        return int(m.group(1))

    return None


def main():
    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    STATEMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    discovered = {}

    print("Downloading FOMC historical index pages...")

    for year in range(
        START_YEAR,
        2021,
    ):
        url = historical_index_url(year)

        try:
            status, final_url, body = fetch(url)
        except Exception as exc:
            print(
                f"INDEX ERROR {year}: {exc}"
            )
            continue

        path = (
            INDEX_DIR
            / f"fomchistorical{year}.html"
        )

        if status == 200:
            path.write_bytes(body)

        print(
            f"{year}: HTTP {status} "
            f"{len(body):7d} bytes"
        )

        if status != 200:
            continue

        links = extract_statement_links(
            final_url,
            body,
        )

        for statement_url, anchor in links.items():
            discovered.setdefault(
                statement_url,
                {
                    "discovery_year": year,
                    "anchor_text": anchor,
                    "url": statement_url,
                    "discovery_source": "historical_year_page",
                },
            )

        time.sleep(REQUEST_DELAY)

    calendar_url = (
        f"{BASE}/monetarypolicy/"
        "fomccalendars.htm"
    )

    print()
    print("Downloading current FOMC calendar...")

    status, final_url, body = fetch(
        calendar_url
    )

    calendar_path = (
        INDEX_DIR
        / "fomccalendars.html"
    )

    if status == 200:
        calendar_path.write_bytes(body)

    print(
        f"calendar: HTTP {status} "
        f"{len(body):7d} bytes"
    )

    if status == 200:
        links = extract_statement_links(
            final_url,
            body,
        )

        for statement_url, anchor in links.items():
            year = infer_year_from_url(
                statement_url
            )

            if year is None:
                continue

            if not (
                2021 <= year <= END_YEAR
            ):
                continue

            discovered.setdefault(
                statement_url,
                {
                    "discovery_year": year,
                    "anchor_text": anchor,
                    "url": statement_url,
                    "discovery_source": "current_calendar",
                },
            )

    print()
    print(
        f"Discovered {len(discovered)} "
        "unique candidate FOMC statement pages."
    )
    print("Downloading statement pages...")

    manifest_rows = []
    failure_rows = []

    ordered = sorted(
        discovered.values(),
        key=lambda r: (
            r["discovery_year"],
            r["url"],
        ),
    )

    for i, item in enumerate(
        ordered,
        start=1,
    ):
        url = item["url"]
        filename = slug_filename(url)
        path = STATEMENT_DIR / filename

        try:
            status, final_url, body = fetch(
                url
            )
        except Exception as exc:
            failure_rows.append({
                "discovery_year":
                    item["discovery_year"],
                "url": url,
                "error": str(exc),
                "http_status": "",
            })

            print(
                f"[{i:03d}/{len(ordered):03d}] "
                f"ERROR {url}: {exc}"
            )

            continue

        if status == 200:
            path.write_bytes(body)
            outcome = "downloaded"
        else:
            outcome = "http_failure"

            failure_rows.append({
                "discovery_year":
                    item["discovery_year"],
                "url": url,
                "error": "",
                "http_status": status,
            })

        manifest_rows.append({
            "discovery_year":
                item["discovery_year"],
            "discovery_source":
                item["discovery_source"],
            "anchor_text":
                item["anchor_text"],
            "url": url,
            "final_url": final_url,
            "filename": filename,
            "http_status": status,
            "bytes": len(body),
            "outcome": outcome,
        })

        print(
            f"[{i:03d}/{len(ordered):03d}] "
            f"{item['discovery_year']} "
            f"HTTP {status} "
            f"{len(body):7d} bytes "
            f"{url}"
        )

        time.sleep(REQUEST_DELAY)

    fields = [
        "discovery_year",
        "discovery_source",
        "anchor_text",
        "url",
        "final_url",
        "filename",
        "http_status",
        "bytes",
        "outcome",
    ]

    with MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    failure_fields = [
        "discovery_year",
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
            fieldnames=failure_fields,
        )
        writer.writeheader()
        writer.writerows(failure_rows)

    counts = {}

    for row in manifest_rows:
        if row["http_status"] != 200:
            continue

        year = int(row["discovery_year"])
        counts[year] = (
            counts.get(year, 0) + 1
        )

    print()
    print("Finished.")
    print()
    print(f"Manifest : {MANIFEST}")
    print(f"Failures : {FAILURES}")
    print(f"HTML     : {STATEMENT_DIR}")
    print()

    print(
        f"{'YEAR':<6} {'STATEMENTS':>10}"
    )
    print("-" * 18)

    for year in range(
        START_YEAR,
        END_YEAR + 1,
    ):
        print(
            f"{year:<6} "
            f"{counts.get(year, 0):>10}"
        )

    print()
    print(
        f"Successful statement pages: "
        f"{sum(counts.values())}"
    )
    print(
        f"Failures                  : "
        f"{len(failure_rows)}"
    )
    print()
    print("No database writes were performed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
