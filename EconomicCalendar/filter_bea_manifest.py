#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path("EconomicCalendar/raw/bea")
INPUT = ROOT / "manifest.csv"
OUTPUT = ROOT / "manifest_filtered.csv"
REJECTED = ROOT / "manifest_rejected.csv"


def is_national_gdp(title: str) -> tuple[bool, str]:
    t = title.lower()

    # Definite non-national GDP products.
    reject_terms = (
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

    for term in reject_terms:
        if term in t:
            return False, f"excluded:{term}"

    # Industry-only releases are not the national headline GDP release.
    if "by industry" in t:
        # Modern national third-estimate releases can also mention
        # GDP by Industry. Keep them when they explicitly contain
        # an estimate marker.
        estimate_markers = (
            "advance estimate",
            "initial estimate",
            "second estimate",
            "third estimate",
            "updated estimate",
            "(advance estimate)",
            "(initial estimate)",
            "(second estimate)",
            "(third estimate)",
            "(updated estimate)",
        )
        if not any(marker in t for marker in estimate_markers):
            return False, "excluded:industry_only"

    # A national GDP event should identify an estimate.
    estimate_markers = (
        "advance estimate",
        "initial estimate",
        "second estimate",
        "third estimate",
        "updated estimate",
        "(advance estimate)",
        "(initial estimate)",
        "(second estimate)",
        "(third estimate)",
        "(updated estimate)",
    )

    if any(marker in t for marker in estimate_markers):
        return True, "national_gdp_estimate"

    # Some newer BEA titles start with GDP (...) rather than
    # "Gross Domestic Product".
    if re.match(r"^gdp\s*\(", t):
        return True, "national_gdp_estimate"

    return False, "excluded:no_estimate_marker"


def classify(row: dict) -> tuple[bool, str]:
    family = row["family"]
    title = row["title"]

    if family == "PCE":
        t = title.lower()

        if "data update" in t:
            return False, "excluded:data_update"

        if "personal income and outlays" in t:
            return True, "personal_income_and_outlays"

        return False, "excluded:not_pio"

    if family == "GDP":
        return is_national_gdp(title)

    return False, "excluded:unknown_family"


def main() -> int:
    with INPUT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept = []
    rejected = []

    for row in rows:
        keep, reason = classify(row)
        row = dict(row)
        row["filter_reason"] = reason

        if keep:
            kept.append(row)
        else:
            rejected.append(row)

    fields = list(rows[0].keys()) + ["filter_reason"]

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)

    with REJECTED.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rejected)

    def counts(items):
        result = {}
        for row in items:
            result[row["family"]] = result.get(row["family"], 0) + 1
        return result

    kept_counts = counts(kept)
    rejected_counts = counts(rejected)

    print("Filtered BEA manifest")
    print()
    print(f"Input rows : {len(rows)}")
    print(f"Kept       : {len(kept)}")
    print(f"Rejected   : {len(rejected)}")
    print()
    print(f"Kept GDP   : {kept_counts.get('GDP', 0)}")
    print(f"Kept PCE   : {kept_counts.get('PCE', 0)}")
    print(f"Reject GDP : {rejected_counts.get('GDP', 0)}")
    print(f"Reject PCE : {rejected_counts.get('PCE', 0)}")
    print()
    print(f"Output     : {OUTPUT}")
    print(f"Rejected   : {REJECTED}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
