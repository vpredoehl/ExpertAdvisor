# Independent Review — Authoritative Economic Calendar Phase 6A Remediation Continuation

## Executive verdict

**The Phase 6A remediation implementation is materially improved and the accepted corpus/database mechanics look strong, but I would not yet treat the current `READY_FOR_PHASE_6B` verdict as fully proven.**

My independent verdict is:

> **CONDITIONAL PASS — one substantive readiness issue and one evidence-integrity issue should be resolved or explicitly accepted before Phase 6B production population.**

The most important concern is **DOL/ETA historical completeness in 2011–2012**, especially 2012. The retained evidence shows only **15 accepted Weekly Claims occurrences out of 54 enumerated in 2012**. That is not merely a few isolated source defects; it is a major annual coverage hole. The current report classifies the 45 EST/EDT contradictions as a non-blocking source limitation, but the original Phase 6A readiness criterion required that no unexplained *major historical coverage gaps* remain.

The second important concern is that parts of the retained verification evidence are **asserted or reconstructed by scripts rather than mechanically observed from the database/test environment**. The corpus TSVs are excellent and independently checkable; some database/safety summary JSON is not.

I found no evidence of a fundamental problem with the shared ingestion architecture, idempotency design, parser provenance remediation, redirect hardening, or the specific September 17, 2014 FOMC same-boundary handling.

---

## Review basis and integrity

The uploaded independent-review archive was extracted and its included `SHA256SUMS.txt` was verified successfully.

The package contains 84 review files, including:

- exact Git working-tree patch and status;
- acquisition scripts;
- BLS, BEA, Census, DOL/ETA, and Federal Reserve adapters;
- shared validation/repository/service code;
- migration `080_economic_event_distinct_same_time_identity.sql`;
- focused and repository tests;
- retained machine-readable Phase 6A audit evidence;
- Phase 6A remediation reports.

The review package does not include the full downloaded government corpus or the command transcript that created every retained JSON summary. That limits independent verification of some operational claims, as discussed below.

---

# Findings

## F1 — HIGH: DOL/ETA 2012 coverage is too sparse to call the historical corpus complete without an explicit policy decision

The retained accepted and rejected manifests independently reproduce this DOL/ETA year distribution:

| Year | Accepted | Rejected / alias evidence | Enumerated |
|---:|---:|---:|---:|
| 2010 | 52 | 0 | 52 |
| 2011 | 44 | 8 | 52 |
| **2012** | **15** | **39** | **54** |
| 2013 | 52 | 0 | 52 |
| 2014 | 53 | 1 | 54 |
| 2015 | 52 | 0 | 52 |
| 2016 | 52 | 0 | 52 |
| 2017 | 52 | 0 | 52 |
| 2018 | 52 | 0 | 52 |
| 2019 | 51 | 0 | 51 |
| 2020 | 53 | 0 | 53 |
| 2021 | 52 | 0 | 52 |
| 2022 | 52 | 0 | 52 |
| 2023 | 52 | 0 | 52 |
| 2024 | 52 | 0 | 52 |
| 2025 | 46 | 0 | 46 |
| 2026 | 33 | 0 | 33 |

The continuation report correctly discovered and fixed the DOL archive-enumeration bug: the archive must be queried with a year-selecting POST rather than a default GET. That was an important implementation correction.

However, after that fix, the resulting accepted corpus still contains only **15 of 54** enumerated 2012 artifacts. The report attributes 45 DOL exclusions overall to first-party EST/EDT labels that contradict the historical New York DST offset, plus one dummy non-release and two duplicate aliases.

Failing closed on contradictory timezone evidence is defensible. What is not yet demonstrated is that losing most of 2012 is non-blocking for the intended historical event feature corpus.

### Why this matters

The final report says that DOL coverage spans 2010–2026, which is literally true, but that range summary hides the very large 2012 density gap.

For a causal model feature, a missing event is not neutral. It changes event-recency and event-family state for bars after the missing release.

### Recommended disposition before Phase 6B

Choose and document one of these approaches:

1. **Explicitly accept the gap as a model-data policy decision**, with quantitative acknowledgment that 2012 Weekly Claims coverage is only 15/54; or

2. Revisit whether the contradictory time-zone label requires excluding the *entire occurrence*. If the authoritative document establishes the publication **date** but its clock-time/zone evidence is contradictory, determine whether existing `date_only` semantics can conservatively preserve the occurrence without inventing an intraday time.

The second approach must be reviewed against the project's existing fail-closed policy; it should not be introduced merely to improve counts.

Until this is resolved explicitly, I would classify Phase 6A as **not fully ready for production population**.

---

## F2 — MEDIUM/HIGH: Some retained database/safety evidence is reconstructed rather than mechanically observed

`Scripts/retain_phase6a_audit_evidence.py` does a good job retaining compact corpus evidence, but some fields are generated as assertions rather than sourced from a measured verification result.

For example, the script writes:

- `disposable_database_dropped: True`
- `post_repeat_database_row_count` as the sum of first-import insert counts

rather than querying PostgreSQL or consuming an independently produced database-verification artifact.

The package also contains:

- `database-verification.json`
- `safety-summary.json`
- `source-completeness.json`
- `test-summary.json`

but no script in the package was found that mechanically generates all of these from raw command/test output.

This does **not** mean the claims are false. The corpus evidence and test source are internally coherent, and the report states that the disposable database tests passed. The issue is independent reproducibility: a reviewer cannot trace every operational claim back to raw command output from the package alone.

### Recommendation

Before Phase 6B, make the retained evidence chain mechanical.

At minimum:

- capture first-import CLI output;
- capture exact-repeat CLI output;
- capture the SQL query proving final row count and duplicate/collision counts;
- capture disposable-DB creation/drop result;
- capture conflict/rollback test output;
- capture build result;
- generate `database-verification.json` and `safety-summary.json` from those retained raw outputs instead of setting boolean facts in a retention script.

Alternatively, retain the complete CEE transcript as immutable audit evidence.

---

## F3 — MEDIUM: BEA reproducibility depends on a retained historical URL catalog, not current first-party archive enumeration alone

The BEA remediation correctly demonstrates that current BEA archive pagination is incomplete relative to the retained 393-URL first-party union:

- current product archive snapshot: 371 occurrences;
- retained first-party union: 393;
- 22 retained URLs absent from current pagination;
- all 22 remain first-party BEA occurrence URLs according to the Phase 6A evidence;
- 391 of 393 current acquisitions parse and are accepted.

The implementation therefore permits `fetch_bea_economic_releases.py` to consume a prior `enumerated-source-manifest` as a retained occurrence catalog.

That is reasonable under an unstable publisher archive, but it changes the reproducibility contract:

> Full BEA enumeration is now reproducible only if that retained catalog is itself treated as a durable, versioned source artifact.

### Additional concern

The retained catalog includes expected SHA-256 hashes. `fetch_bea_economic_releases.py` fails if a current first-party page's bytes differ from the retained hash. Yet the Phase 6A evidence also reports two same-URL hash changes whose parsed semantics remained unchanged.

This means the retained catalog is both:

- the fallback occurrence inventory; and
- a byte-level snapshot whose hashes can legitimately become stale when BEA edits historical HTML.

### Recommendation

Commit/version the compact BEA occurrence catalog and document an explicit refresh process:

1. retained URL set is the occurrence inventory;
2. URL identity must remain immutable;
3. byte hash changes trigger semantic re-audit, not automatic rejection forever;
4. new accepted hash is recorded only after review.

Without that policy, future "reproducible acquisition" can fail merely because BEA reformatted old pages.

This is not necessarily a blocker for the current Phase 6B population, but it should be formalized before treating the acquisition process as long-term reproducible infrastructure.

---

## F4 — MEDIUM: Migration 080 is correct for the evidenced corpus but highly data-specific

Migration 080 removes the previous database uniqueness constraint on:

`(source_agency, event_family, event_timestamp_utc)`

and replaces it with a guarded unique index plus a check constraint that permits exactly the two Federal Reserve identities:

- `federal_reserve:monetary20140917a`
- `federal_reserve:monetary20140917c`

at the shared conservative date-only boundary.

The C++ validation layer and repository enforce the same exception, and the tests cover insert and exact-repeat behavior.

The official Federal Reserve pages identify both as distinct FOMC statement press releases on September 17, 2014, so the underlying source distinction is defensible.

### Concern

The migration comment says more generally that:

> distinct authoritative occurrences may share a conservative causal boundary

but the schema only permits one historical pair.

That is a deliberate fail-closed design, but it couples schema evolution to one known source occurrence. If another legitimate same-family/date-only pair is discovered later, a new migration and C++ exception will be required.

### Recommendation

This is acceptable for the current corpus if the team consciously prefers a **whitelist-at-schema-level** policy.

If so, document that policy explicitly:

- timestamp coincidence is normally forbidden;
- exceptions require first-party evidence;
- every new exception requires migration + validator + repository + audit tests.

Do not generalize the database constraint merely for elegance if doing so would remove a valuable fail-closed guard.

---

## F5 — LOW/MEDIUM: BLS acquisition counts conflate 17 downloaded schedule resources with 813 derived occurrences

BLS acquisition is structurally different from the other agencies.

The script downloads 17 annual BLS schedule pages and derives 813 supported release rows from them.

The retained audit therefore reports:

- `enumerated_occurrence_count = 813`
- `acquisition_success_count = 813`

while the final report separately records only 17 actual source resources.

This is internally explainable, but `acquisition_success_count` sounds like 813 independent source downloads when there were 17.

### Recommendation

For future audits, distinguish:

- `source_resource_count`
- `occurrence_count`
- `occurrences_with_verified_source_resource`

This is reporting clarity, not a correctness blocker.

---

# Positive findings

## P1 — Package integrity and corpus counts are internally consistent

The archive checksum inventory verifies.

The accepted-event manifest contains exactly **3,583 rows**, matching the retained aggregate:

- BEA: 391
- BLS: 813
- Census: 1,166
- DOL/ETA: 815
- Federal Reserve: 398

There are zero duplicate `(agency, source_event_id)` identities in the retained accepted manifest.

There is exactly one same-agency/family/timestamp group with multiple accepted events: the September 17, 2014 FOMC pair.

---

## P2 — Redirect and bounded-download remediation is substantially improved

The shared `authoritative_acquisition.py` adds:

- canonical first-party HTTPS URL validation;
- exact final-resource validation;
- explicit redirect failure;
- bounded reads;
- deterministic acquisition-manifest writing.

The agency scripts use those helpers in the reviewed remediation.

This directly addresses a material weakness from the prior independent review.

---

## P3 — Parser provenance was fixed

The reviewed manifests/adapters now use differentiated parser revisions including:

- `bea_economic_release_v2`
- `bls_schedule_release_v1`
- `census_economic_release_v2`
- `dol_eta_weekly_claims_v3`
- `federal_reserve_economic_release_v2`

The retained parser-provenance JSON also records hashes for:

- audit binary;
- audit script;
- source manifest;
- shared ingestion contract version.

This is a significant improvement in audit reproducibility.

---

## P4 — Collision auditing now considers parsed candidates independently of candidate-level validation

`apply_batch_validation()` now builds identity/timestamp evidence from every successfully parsed candidate, rather than only rows already marked individually valid.

This fixes an important defect from the earlier review: an individually invalid candidate can no longer silently hide evidence of source-identity ambiguity.

The script also distinguishes exact duplicate authoritative aliases from unresolved collisions.

---

## P5 — Database import semantics remain strong

The retained evidence reports:

- first import: 3,583 inserted;
- exact repeat: 3,583 unchanged;
- zero repeat inserts/updates/rejects;
- zero duplicate source identities;
- conflict suites passed;
- atomic mixed-batch rollback suites passed.

The reviewed repository code continues to compare authoritative identity and same-family/timestamp collisions before writes and refuses partial application when a batch contains a conflict.

No architectural bypass around the shared service/repository was introduced.

---

## P6 — The FOMC same-boundary pair is supported by first-party evidence

The two September 17, 2014 Federal Reserve pages are distinct official press releases:

- the normal FOMC policy statement;
- an FOMC statement on policy normalization principles and plans.

Both are titled by the Federal Reserve as FOMC statements and both are marked "For immediate release" without an authoritative clock time on the page.

Treating them as distinct immutable source identities with the same conservative date-only causal boundary is therefore a defensible representation.

---

## P7 — Production-safety design remains intact

Nothing in the reviewed source changes introduces a production-only persistence path or scheduler coupling.

The package states that:

- production database was not mutated;
- production scheduler was not touched;
- production DerivedData was not touched;
- development build output used `DerivedData/Development`.

The operational proof of those claims should be made more mechanically reproducible as noted in F2.

---

# Recommended disposition

I would **not send the implementation back for broad rework**. Most of Phase 6A is in good shape.

Before committing the final `READY_FOR_PHASE_6B` state, I recommend two focused actions:

1. **Make an explicit DOL/ETA 2011–2012 coverage decision**, especially the 15/54 accepted 2012 corpus. Either:
   - formally accept that sparse historical coverage as non-blocking, or
   - investigate a conservative `date_only` disposition for timezone-contradictory documents if that is consistent with the project's causal policy.

2. **Strengthen the retained verification evidence** so DB row counts, idempotency, rollback, disposable-DB cleanup, and build results are generated from retained raw command output rather than asserted in summary JSON.

The BEA retained-union policy and migration 080 should also be documented explicitly, but I do not consider either an automatic blocker for Phase 6B.

---

# Independent final verdict

## CONDITIONAL PASS

The remediation is substantially better than the earlier Phase 6A state and the 3,583-row accepted corpus is internally coherent and collision-clean.

However, I would change the current project verdict from unconditional:

`READY_FOR_PHASE_6B`

to:

`READY_FOR_PHASE_6B_AFTER_EXPLICIT_DOL_COVERAGE_ACCEPTANCE_AND_EVIDENCE_HARDENING`

before authorizing production population.

This recommendation is deliberately conservative. It does not call for weakening causal timestamps, source identity, provenance, or fail-closed behavior.
