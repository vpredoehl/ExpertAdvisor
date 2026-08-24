# DOL/ETA Weekly Claims fixture provenance

These are minimal first-party excerpts retained for deterministic parser tests.
They contain only the occurrence header, publication identity where available,
title, and the first reference-week sentence needed by this ingestion phase.
No statistical values are imported.

- `2010-07-22.html`
  - Authoritative URL: https://oui.doleta.gov/press/2010/072210.asp
  - Retrieved/verified: 2026-08-23
  - Retained portion: DOL/ETA header, `USDL 10-990-NAT`, embargo time/date/zone,
    report title, and initial/insured-claims reference-week sentences.
  - The source establishes Thursday July 22, 2010 at 8:30 a.m. EDT and the
    initial-claims reference week ending July 17, 2010.
- `2025-01-02.txt`
  - Authoritative URL: https://oui.doleta.gov/press/2025/010225.pdf
  - Retrieved/verified: 2026-08-23
  - Retained portion: deterministic page-one text corresponding to the DOL PDF
    header, title, and first reference-week sentences.
  - The source establishes Thursday January 2, 2025 at 8:30 a.m. Eastern and
    the initial-claims reference week ending December 28, 2024. The modern PDF
    has no USDL release number in its occurrence header, so the adapter uses an
    artifact-type/path-qualified immutable-URL identity.

SHA-256 values are pinned in `manifest.tsv`; the manifest itself is the parser
handoff used by offline tests. Production PDF acquisition retains and hashes
the original PDF and records the exact `pdftotext` version separately.
