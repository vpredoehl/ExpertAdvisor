#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import html
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

BASE = "https://www.bea.gov"
ARCHIVE = f"{BASE}/news/archive"

START_YEAR = 2010
END_YEAR = 2026

ROOT = Path("EconomicCalendar/raw/bea")
RELEASE_DIR = ROOT / "releases"
ARCHIVE_DIR = ROOT / "archive_pages"

MANIFEST = ROOT / "manifest.csv"
FAILURES = ROOT / "http_failures.csv"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
    "AppleWebKit/537.36 "
    "ExpertAdvisor EconomicCalendar research downloader"
)

# BEA archive is paginated. We intentionally allow a generous upper bound
# and stop after multiple consecutive pages contain no archive result links.
MAX_ARCHIVE_PAGES = 100

# Be polite to BEA.
REQUEST_DELAY_SECONDS = 0.25


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


def classify_title(title: str) -> str | None:
    t = title.lower()

    # Monthly PCE release family.
    if "personal income and outlays" in t:
        return "PCE"

    # National quarterly GDP releases.
    #
    # Exclude state, county, territory, metro, industry-only, etc.
    if "gross domestic product" in t or re.search(r"\bgdp\b", t):
        excluded = (
            "state",
            "county",
            "metropolitan",
            "guam",
            "american samoa",
            "virgin islands",
            "puerto rico",
            "northern mariana",
            "by industry",
            "industries",
        )

        if any(x in t for x in excluded):
            # Modern combined national GDP releases can include "Industries"
            # after the national GDP estimate. Keep those only if they also
            # clearly identify an estimate / quarter.
            national_markers = (
                "advance estimate",
                "second estimate",
                "third estimate",
                "updated estimate",
                "initial estimate",
                "quarter",
                "q1",
                "q2",
                "q3",
                "q4",
            )
            if not any(x in t for x in national_markers):
                return None

            territorial = (
                "guam",
                "american samoa",
                "virgin islands",
                "puerto rico",
                "northern mariana",
                "county",
                "metropolitan",
            )
            if any(x in t for x in territorial):
                return None

        return "GDP"

    return None


def extract_archive_entries(page_html: str):
    """
    Extract:
        href
        visible title
        published date/year if available

    BEA archive result links point into /news/YYYY/...
    """
    pattern = re.compile(
        r'<a[^>]+href="(?P<href>/news/(?P<year>\d{4})/[^"]+)"[^>]*>'
        r'(?P<title>.*?)</a>',
        re.I | re.S,
    )

    seen = set()

    for m in pattern.finditer(page_html):
        href = html.unescape(m.group("href"))
        year = int(m.group("year"))
        title = clean_text(m.group("title"))

        if not title:
            continue

        key = (href, title)
        if key in seen:
            continue
        seen.add(key)

        yield year, title, urljoin(BASE, href)


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    discovered: dict[str, dict] = {}

    empty_pages = 0

    print("Crawling BEA news archive...")

    for page in range(MAX_ARCHIVE_PAGES):
        url = (
            f"{ARCHIVE}"
            f"?created_1=All"
            f"&field_related_product_target_id=All"
            f"&title="
            f"&page={page}"
        )

        try:
            status, body = fetch(url)
        except Exception as exc:
            print(f"ARCHIVE ERROR page={page}: {exc}")
            break

        if status != 200:
            print(f"ARCHIVE HTTP {status}: page={page}")
            break

        archive_path = ARCHIVE_DIR / f"archive_page_{page:03d}.html"
        archive_path.write_bytes(body)

        text = body.decode("utf-8", errors="replace")
        entries = list(extract_archive_entries(text))

        relevant_this_page = 0

        for year, title, release_url in entries:
            if year < START_YEAR or year > END_YEAR:
                continue

            family = classify_title(title)
            if family is None:
                continue

            relevant_this_page += 1

            discovered.setdefault(
                release_url,
                {
                    "family": family,
                    "year": year,
                    "title": title,
                    "url": release_url,
                },
            )

        print(
            f"archive page {page:02d}: "
            f"{len(entries):3d} release links, "
            f"{relevant_this_page:2d} relevant, "
            f"{len(discovered):3d} total unique"
        )

        if not entries:
            empty_pages += 1
        else:
            empty_pages = 0

        # Once we're beyond the archive and see 3 empty pages, stop.
        if empty_pages >= 3:
            break

        time.sleep(REQUEST_DELAY_SECONDS)

    print()
    print(f"Discovered {len(discovered)} candidate BEA releases.")
    print("Downloading release pages...")

    manifest_rows = []
    failure_rows = []

    for i, item in enumerate(
        sorted(discovered.values(), key=lambda x: (x["year"], x["family"], x["url"])),
        start=1,
    ):
        url = item["url"]
        filename = slug_filename(url)
        path = RELEASE_DIR / filename

        if path.exists() and path.stat().st_size > 0:
            status = 200
            body = path.read_bytes()
            outcome = "existing"
        else:
            try:
                status, body = fetch(url)
            except Exception as exc:
                failure_rows.append(
                    {
                        "family": item["family"],
                        "year": item["year"],
                        "title": item["title"],
                        "url": url,
                        "http_status": "",
                        "error": str(exc),
                    }
                )
                print(f"[{i:03d}] ERROR {item['family']} {url}: {exc}")
                continue

            if status == 200:
                path.write_bytes(body)
                outcome = "downloaded"
            else:
                outcome = "http_failure"
                failure_rows.append(
                    {
                        "family": item["family"],
                        "year": item["year"],
                        "title": item["title"],
                        "url": url,
                        "http_status": status,
                        "error": "",
                    }
                )

        manifest_rows.append(
            {
                "family": item["family"],
                "year": item["year"],
                "title": item["title"],
                "url": url,
                "filename": filename,
                "http_status": status,
                "bytes": len(body),
                "outcome": outcome,
            }
        )

        print(
            f"[{i:03d}/{len(discovered):03d}] "
            f"{item['family']:3s} "
            f"{item['year']} "
            f"HTTP {status} "
            f"{len(body):7d} bytes "
            f"{item['title'][:72]}"
        )

        time.sleep(REQUEST_DELAY_SECONDS)

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "family",
            "year",
            "title",
            "url",
            "filename",
            "http_status",
            "bytes",
            "outcome",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    with FAILURES.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "family",
            "year",
            "title",
            "url",
            "http_status",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_rows)

    counts = {}
    for row in manifest_rows:
        if row["http_status"] == 200:
            counts[row["family"]] = counts.get(row["family"], 0) + 1

    print()
    print("Finished.")
    print(f"Manifest : {MANIFEST}")
    print(f"Failures : {FAILURES}")
    print(f"HTML     : {RELEASE_DIR}")
    print(f"PCE      : {counts.get('PCE', 0)} successful pages")
    print(f"GDP      : {counts.get('GDP', 0)} successful pages")
    print(f"Failures : {len(failure_rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
