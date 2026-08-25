#!/usr/bin/env python3

from __future__ import annotations

import csv
import html
import re
from collections import Counter
from pathlib import Path

ROOT = Path("EconomicCalendar/raw/federal_reserve")

INPUT = ROOT / "manifest.csv"
STATEMENT_DIR = ROOT / "statements"

OUTPUT = ROOT / "manifest_filtered.csv"
REJECTED = ROOT / "manifest_auxiliary.csv"


def strip_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def page_heading(text: str) -> str:
    h3s = re.findall(
        r"<h3[^>]*>(.*?)</h3>",
        text,
        re.I | re.S,
    )

    for raw in h3s:
        value = strip_html(raw)

        if value:
            return value

    h2s = re.findall(
        r"<h2[^>]*>(.*?)</h2>",
        text,
        re.I | re.S,
    )

    for raw in h2s:
        value = strip_html(raw)

        if not value:
            continue

        if value.lower() == "press release":
            continue

        return value

    return ""


def classify(title: str, url: str):
    t = title.lower()

    #
    # Reject auxiliary monetary-policy releases BEFORE looking for
    # generic "FOMC statement" wording. Some auxiliary pages, such as
    # the September 17, 2014 normalization release, contain that phrase
    # but are not the primary meeting policy statement.
    #
    auxiliary_markers = (
        "balance sheet",
        "normalization",
        "principles",
        "plans for reducing",
        "implementation",
        "longer-run goals",
        "longer run goals",
        "policy normalization",
        "securities holdings",
        "operating regime",
        "fima repo",
        "repo facility",
        "foreign and international monetary authorities",
        "temporary u.s. dollar liquidity arrangements",
    )

    if any(x in t for x in auxiliary_markers):
        return False, "auxiliary_monetary_release"

    #
    # Primary FOMC policy statements.
    #
    primary_markers = (
        "federal reserve issues fomc statement",
        "federal open market committee statement",
        "fomc statement:",
    )

    if any(x in t for x in primary_markers):
        return True, "primary_fomc_statement"

    #
    # Historical/current main statement URL convention.
    #
    # This is only a fallback after auxiliary content has been excluded.
    #
    if re.search(
        r"(?:monetary)?20\d{6}a\.htm$",
        url,
        re.I,
    ):
        return True, "primary_url_a"

    return False, "unclassified_monetary_release"


def main():
    with INPUT.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    kept = []
    rejected = []

    for row in rows:
        path = STATEMENT_DIR / row["filename"]

        if not path.exists():
            r = dict(row)
            r["page_title"] = ""
            r["filter_reason"] = "missing_html"
            rejected.append(r)
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        title = page_heading(text)

        keep, reason = classify(
            title,
            row["url"],
        )

        r = dict(row)
        r["page_title"] = title
        r["filter_reason"] = reason

        if keep:
            kept.append(r)
        else:
            rejected.append(r)

    fields = list(rows[0].keys()) + [
        "page_title",
        "filter_reason",
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
        w.writerows(kept)

    with REJECTED.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(rejected)

    kept_counts = Counter(
        int(r["discovery_year"])
        for r in kept
    )

    rejected_counts = Counter(
        int(r["discovery_year"])
        for r in rejected
    )

    print("FOMC manifest filter")
    print()
    print(f"Input rows : {len(rows)}")
    print(f"Kept       : {len(kept)}")
    print(f"Auxiliary  : {len(rejected)}")
    print()
    print(f"{'YEAR':<6} {'PRIMARY':>8} {'AUX':>6}")
    print("-" * 24)

    for year in range(2010, 2027):
        print(
            f"{year:<6} "
            f"{kept_counts[year]:>8} "
            f"{rejected_counts[year]:>6}"
        )

    print()
    print(f"Primary   : {OUTPUT}")
    print(f"Auxiliary : {REJECTED}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
