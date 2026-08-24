# Independent Review — Authoritative Economic Calendar Phase 6A

## Executive verdict

**Phase 6A itself is a PASS as a pre-population audit, but the project is correctly `NOT_READY_FOR_PHASE_6B`.**

The implementation demonstrates a strong shared ingestion architecture, conservative causal timestamp handling, deterministic parsing for the accepted subset, disposable-database idempotency, and production-safety discipline.

However, an independent review finds several important issues that should be addressed before Phase 6B:

1. **The review bundle does not contain enough evidence to independently reproduce the headline corpus counts.**
2. **BEA, Census, and DOL/ETA acquisition allow same-host redirects without proving the final URL is the canonical occurrence URL.**
3. **The manifest audit can miss duplicate identities when one parsed candidate has already failed individual validation.**
4. **Parser behavior changed materially while manifest `parser_version` remained `v1`, weakening reproducibility of accepted manifests.**
5. Existing Phase 6A blockers remain valid: BLS reproducibility, DOL/ETA historical coverage, Census identity collisions, same-date/date-only FOMC collision, and failed-closed artifacts.

I do **not** recommend weakening causal timestamps, source identity constraints, or fail-closed parser behavior to eliminate those blockers.

---

## Material reviewed

The independent-review archive was integrity-checked against its included `SHA256SUMS.txt`.

The bundle contains:

- current Phase 6A working-tree diff;
- acquisition scripts for BEA, Census, DOL/ETA, and Federal Reserve;
- BEA, Census, DOL/ETA, and Federal Reserve adapters;
- shared import validation/repository/service sources;
- migration 072;
- Phase 6A manifest-audit tooling and tests;
- selected adapter/acquisition tests;
- Xcode project and shared `main.cpp`;
- Phase 6A continuation and final audit reports;
- git branch, HEAD, status, log, and patch metadata.

The package intentionally does **not** contain the bulk acquired historical corpus, Phase 6A staging directory, complete accepted manifests/audit JSON outputs, database dumps, or command transcripts.

---

# Findings

## F1 — HIGH: Quantitative Phase 6A results are not independently reproducible from the review bundle

The report claims:

- 2,007 non-BLS authoritative artifacts enumerated;
- 2,007 hashes validated;
- 1,993 parsed;
- 14 parse failures;
- 12 whole-corpus validation failures;
- 1,981 accepted non-BLS occurrences;
- 1,981 first-pass inserts;
- 1,981 unchanged on exact repeat.

Those values are internally coherent, and the code structure supports that workflow. However, the archive does not contain the actual Phase 6A full manifests, accepted manifests, per-occurrence audit TSV/JSON files, acquisition logs, or DB result exports from which these counts can be recomputed.

Therefore an independent reviewer can validate the **implementation design** and the **reported methodology**, but cannot independently derive the 2,007 / 1,993 / 1,981 numbers from the supplied evidence.

### Recommendation

Before using Phase 6A as the formal release gate for Phase 6B, preserve and package at least:

- source manifests for each agency/family;
- accepted manifests;
- per-entry audit TSV or JSON;
- aggregate audit output by agency/family/year/confidence;
- import summary from first pass;
- import summary from exact repeat;
- conflict/rollback test output;
- acquisition command log and upper/lower coverage boundaries.

The raw 408 MB source corpus does not necessarily need to be committed, but the compact machine-readable audit artifacts should be retained as review evidence.

---

## F2 — HIGH: Same-host redirects weaken occurrence provenance for BEA, Census, and DOL/ETA

The Federal Reserve acquisition path is strict: `fetch()` reconstructs the final response URL and requires it to equal the requested canonical occurrence URL exactly.

By contrast:

- BEA accepts any HTTPS redirect that remains on `bea.gov` / `www.bea.gov`;
- Census accepts redirects anywhere on `www.census.gov` or `www2.census.gov`;
- DOL/ETA accepts redirects anywhere on its allowed host.

The manifest continues to record the original canonical occurrence URL rather than the final response URL.

This creates a provenance weakness: if an old occurrence URL is redirected to another release, archive page, generic landing page, or replacement artifact on the same government host, the downloaded bytes can be hashed and parsed while the manifest still claims they came from the original immutable occurrence URL.

Several parsers perform internal date/identity checks, which reduces the risk, but provenance should not depend on every parser detecting every possible same-host redirect mismatch.

### Recommendation

Apply a consistent redirect policy:

- occurrence downloads: require final URL == canonical requested URL, or explicitly validate an approved canonical-equivalence rule;
- archive/index pages: same-host redirect may be acceptable if the final path is also validated as the expected archive endpoint.

At minimum, record the final URL in acquisition metadata and fail closed when an occurrence redirects to a different path.

---

## F3 — MEDIUM/HIGH: Whole-batch collision detection ignores parsed candidates that failed individual validation

`apply_batch_validation()` currently includes only rows where:

```text
validation_status == "validated"
```

when constructing source-ID and agency/family/timestamp collision sets.

That means a candidate that parsed successfully but failed individual validation is ignored for collision purposes.

Example:

- occurrence A parses to `source_event_id=census:cbXX-YY` but fails local/UTC consistency;
- occurrence B parses to the same source ID and otherwise validates.

The current logic can leave B accepted because A is omitted from the duplicate identity set.

For immutable source identity auditing, a parsed-but-invalid occurrence can still be evidence that the source uses the same identity for two different artifacts. That ambiguity should normally prevent the other occurrence from being silently accepted until the collision is understood.

### Recommendation

Separate two concepts:

1. **candidate validity**;
2. **corpus identity ambiguity**.

For duplicate `source_event_id` analysis, consider every successfully parsed candidate that has a usable agency/source ID, even if another validation invariant failed.

For timestamp collision analysis, consider every parsed candidate whose family and timestamp fields are usable.

Then retain the original individual validation diagnostic in addition to the collision diagnostic.

---

## F4 — MEDIUM/HIGH: Parser implementations changed materially without changing `parser_version`

The acquisition manifests still declare:

- `bea_economic_release_v1`
- `census_economic_release_v1`
- `dol_eta_weekly_claims_v1`
- `federal_reserve_economic_release_v1`

Phase 6A materially changed BEA, Census, and Federal Reserve parsing behavior, including:

- historical BEA identity/header variants;
- combined-month and shutdown-era BEA handling;
- Census shutdown/date-only/header variants;
- broader Federal Reserve Beige Book URL support;
- FOMC minutes/reference association behavior.

Yet the parser version remains `v1`.

This matters because an accepted manifest stores raw artifact rows, not the fully normalized candidate. Re-running the same accepted manifest later with different parser code under the same parser version can produce different candidates while appearing to use the same versioned contract.

### Recommendation

Before Phase 6B, choose one of these approaches:

- bump parser versions when parser semantics materially change; or
- add an immutable parser/build provenance identifier to the audit artifact, such as:
  - git commit hash;
  - adapter source hash;
  - audit binary SHA-256;
  - explicit parser schema/version revision.

The production population report should record the exact parser implementation that generated the accepted candidates.

---

## F5 — MEDIUM: Audit timestamp metadata uses source-manifest filesystem mtime

The audit document derives:

```text
source_manifest_completed_at_utc
```

from `manifest.stat().st_mtime`.

Filesystem mtime is not durable acquisition provenance. Copying, unpacking, restoring, or touching the manifest can change that value without changing its contents.

### Recommendation

Treat manifest mtime as operational metadata only.

For durable provenance, prefer:

- manifest SHA-256;
- acquisition-completed timestamp written into the manifest or a signed/hashed sidecar;
- git/release artifact identity.

---

## F6 — MEDIUM: The audit summary label `failed=` does not represent total validation failures

The script prints:

```text
failed = occurrence_count - parsed_count
```

This reports parse/provenance failures but excludes candidates that parsed and later failed individual or batch validation.

The JSON/TSV fields retain the necessary detail, so this does not corrupt the accepted manifest. But the terminal summary can understate failures.

### Recommendation

Print separate counts:

- provenance_failed;
- parse_failed;
- validation_failed;
- batch_collision_failed;
- accepted.

This will make operational review less error-prone.

---

## F7 — MEDIUM: Acquisition size limits are inconsistent across agencies

Federal Reserve acquisition uses an explicit 25 MB maximum download and reads one byte beyond the limit to enforce it.

BEA, Census, and DOL/ETA currently call `response.read()` without equivalent byte caps.

Because these are first-party government sources, this is not a major security exposure, but it weakens fail-closed acquisition behavior and can allow an unexpected endpoint response to consume large memory/disk resources.

### Recommendation

Use a consistent bounded-download helper for all agencies.

---

# Existing blockers independently confirmed as legitimate

## B1 — BLS reproducibility remains a Phase 6B blocker

The package and report show BLS is already present in production and was audited read-only, but no committed reproducible BLS acquisition/import corpus is included in the current architecture.

That prevents a clean source-to-ledger reconstruction of those 813 rows.

**Do not proceed to Phase 6B until BLS provenance is reproducible or formally reconstructed and audited.**

---

## B2 — DOL/ETA 2010-current enumeration is incomplete

The DOL/ETA acquisition enumerates links from the authoritative archive rather than guessing weekly URLs. That is the right policy.

The report says the live archive exposed only 46 accepted 2025 releases, leaving 2010-2024 and 2026 unresolved.

That is a real coverage blocker.

**Do not solve this by synthesizing historical URLs unless a first-party source provides a deterministic published naming/index contract.**

---

## B3 — Census source-ID collisions require an explicit identity decision

Five Census source-ID collision groups affect 10 candidates.

The shared schema correctly requires `(source_agency, source_event_id)` uniqueness.

The independent review agrees that these should not be bypassed with arbitrary suffixes or timestamp-based identity.

The correct next step is to inspect the authoritative source semantics and determine whether:

- the publisher reused an identifier;
- two artifacts are revisions of one publication;
- the URL/path contains a stronger occurrence identity;
- one artifact is genuinely duplicated.

---

## B4 — September 17, 2014 same-family/date-only FOMC collision is a real schema/semantic edge case

Two distinct authoritative FOMC statement pages resolve conservatively to the same date-only causal boundary.

The existing schema also requires uniqueness of:

```text
(source_agency, event_family, event_timestamp_utc)
```

The implementation correctly refuses to invent a release time.

The independent review agrees this requires an explicit contract decision rather than a parser workaround.

Potential architecture directions for later review include:

- allowing multiple authoritative events at the same agency/family/timestamp when source identity differs;
- representing date-only availability with a separate causal-boundary field rather than using the boundary as occurrence uniqueness;
- another schema-level distinction justified by the source.

No schema change should be made solely to force these two rows in without reviewing downstream assumptions.

---

## B5 — Failed-closed artifacts should remain failed closed unless better first-party evidence exists

The report identifies 14 authoritative artifacts that fail because of contradictions or missing provenance.

This is not evidence that the parsers are too strict.

The project should retain a disposition ledger for each excluded artifact:

- genuine non-publication;
- contradictory source timezone;
- missing release time with safe `date_only` fallback unavailable;
- publisher/source error;
- unsupported source layout;
- other documented reason.

Only source-backed evidence should promote an excluded artifact into the accepted corpus.

---

# Positive findings

## P1 — Shared ingestion architecture remains sound

The implementation continues to use:

```text
authoritative acquisition
    -> agency adapter
    -> AuthoritativeEconomicEventCandidate
    -> shared validation
    -> shared service
    -> shared repository
    -> economic_event
```

No parallel Phase-6-specific SQL ingestion path was introduced.

---

## P2 — Causal timestamp discipline is strong

The shared validator enforces:

- exact/reconstructed/date_only confidence;
- America/New_York source timezone;
- local-to-UTC consistency;
- conservative next-day 00:00 New York availability boundary for `date_only`.

The report's 1,911 exact / 70 date-only / 0 reconstructed accepted non-BLS distribution is consistent with a conservative policy.

The review strongly supports keeping this behavior.

---

## P3 — Repository conflict semantics are appropriately strict

The repository:

- compares existing source identity and family/timestamp collisions before writing;
- rejects conflicting batches;
- converts would-be inserts into rejected dispositions on an atomic batch conflict;
- commits only when the complete batch is conflict-free.

This is a strong basis for later production population.

---

## P4 — Federal Reserve enumeration is materially improved

The acquisition code now enumerates:

- annual press archive indexes;
- FOMC statements;
- FOMC minutes;
- Beige Book archives;
- intermeeting statements when present in the authoritative archive.

Tests specifically include the May 9, 2010 intermeeting statement and exclude unrelated Board discount-rate minutes.

This addresses the key Phase 5 acquisition-readiness gap.

---

## P5 — Production safety boundaries were preserved

According to the supplied report:

```text
LSTM_DATABASE_READ_ONLY_ACCESSED=true
FOREX_DATABASE_ACCESSED=false
PRODUCTION_DATABASE_MUTATED=false

PRODUCTION_SCHEDULER_TOUCHED=false
PRODUCTION_EXPERIMENT_TOUCHED=false
PRODUCTION_WORKER_SIGNALED=false
PRODUCTION_DERIVEDDATA_TOUCHED=false
```

The disposable Phase 6A database was reported dropped.

Nothing in the reviewed source changes indicates a hidden production mutation path.

---

# Recommended next sequence

I would **not proceed directly to Phase 6B**.

Recommended order:

1. **Commit Phase 6A only after addressing F2-F4 or explicitly tracking them as follow-up defects.**
2. Preserve compact machine-readable Phase 6A evidence so the corpus counts can be independently reproduced.
3. Resolve BLS reproducible acquisition/import.
4. Resolve DOL/ETA historical enumeration.
5. Investigate Census source-ID collisions.
6. Make a schema/contract decision for same-family/date-only distinct events.
7. Classify all 14 failed-closed artifacts with explicit dispositions.
8. Re-run the entire Phase 6A audit into a fresh disposable database.
9. Require the rerun to produce `READY_FOR_PHASE_6B`.
10. Only then write a separate Phase 6B production-population prompt.

---

# Independent readiness decision

**`NOT_READY_FOR_PHASE_6B`**

This agrees with the Phase 6A report, but the independent review adds four implementation/evidence issues that should be considered before the next readiness run:

- insufficient preserved review evidence for quantitative corpus claims;
- inconsistent same-host redirect handling;
- collision auditing that ignores parsed-but-invalid candidates;
- unchanged parser version despite material parser semantic changes.

None of these findings justify weakening source provenance, causal timestamp rules, immutable identity, or fail-closed behavior.
